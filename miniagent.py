"""miniagent — a learning-oriented, minimal LLM coding agent.

The whole agent fits in this single file so you can read it top to bottom and
see exactly how a coding agent works:

    LLM call  ->  read tool_calls  ->  run the tools  ->  feed results back  ->  loop

There is no framework and no magic. Three tools (read_file, write_file,
run_bash), one hand-written while loop, and the OpenAI Chat Completions API.

SAFETY: This is NOT a sandbox. `run_bash` runs arbitrary shell commands with
shell=True. See the README and the run_bash docstring before using it.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# --- imports + config -------------------------------------------------------

# Default model. Override with the MODEL env var. gpt-5.4-mini is cheap and good
# enough for tool use; swap to a stronger model if you need more reliable tool
# calling on harder tasks.
DEFAULT_MODEL = os.environ.get("MODEL", "gpt-5.4-mini")

# Hard stop so a confused model can never loop forever.
MAX_ITERATIONS = 25

# Commands and file paths are resolved relative to this directory. It is set
# from the CLI (--workdir), defaulting to the current working directory.
WORKDIR = Path.cwd()

# How long a single run_bash command may run before we give up on it.
BASH_TIMEOUT_SECONDS = 30

SYSTEM_PROMPT = (
    "You are miniagent, a minimal coding agent. You accomplish the user's task "
    "by calling the provided tools: read_file, write_file, run_bash. "
    "Work step by step. Inspect files before changing them. When the task is "
    "fully done, reply with a short plain-text summary and stop calling tools."
)


# --- tools ------------------------------------------------------------------

def _safe_path(path: str) -> Path:
    """Resolve `path` inside WORKDIR, rejecting anything that escapes it.

    We resolve BOTH the working directory and the target before comparing, so
    that '..' segments and symlinks are collapsed first. Without the pre-resolve
    a path like '../etc/passwd' would slip past the check.
    """
    base = WORKDIR.resolve()
    target = (base / path).resolve()
    if not target.is_relative_to(base):  # Python 3.9+
        raise ValueError(f"path escapes WORKDIR: {path}")
    return target


def read_file(args: dict, confirm: bool = True) -> str:
    """Read a UTF-8 text file inside WORKDIR and return its content.

    On any failure (missing file, decode error, escape) we return an ERROR
    string instead of raising — the agent loop turns that into a tool message
    so the model can see what went wrong and recover.
    """
    try:
        target = _safe_path(args["path"])
        return target.read_text(encoding="utf-8")
    except Exception as e:  # noqa: BLE001 — surface every failure to the model
        return f"ERROR: {e}"


def write_file(args: dict, confirm: bool = True) -> str:
    """Write text to a file inside WORKDIR (creates parent dirs, overwrites)."""
    try:
        target = _safe_path(args["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        content = args["content"]
        target.write_text(content, encoding="utf-8")
        n = len(content.encode("utf-8"))
        return f"Wrote {n} bytes to {args['path']}"
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


def run_bash(args: dict, confirm: bool = True) -> str:
    """Run a shell command in WORKDIR and return exit code + stdout + stderr.

    !!! NOT A SANDBOX !!!
    This uses shell=True, so the command can do anything the current user can:
    `cd ..`, `rm -rf`, `curl ... | sh`, read your env vars, hit the network.
    The WORKDIR confinement that protects read_file/write_file does NOT apply
    here. Only run this on machines and tasks you trust.

    When `confirm` is True (the default, fail-safe) we ask for human approval
    before running. A rejection — or a non-interactive stdin (EOFError) — is
    returned as a normal ERROR string, never raised, so the agent loop can
    keep going.
    """
    command = args["command"]
    if confirm:
        try:
            answer = input(f"\n[run_bash] execute? {command!r} [y/N] ")
        except EOFError:
            return "ERROR: command rejected (no interactive confirmation available)"
        if answer.strip().lower() not in ("y", "yes"):
            return "ERROR: command rejected by user"
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(WORKDIR),
            capture_output=True,
            text=True,
            timeout=BASH_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return "ERROR: command timed out"
    return (
        f"exit_code: {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )


# --- schemas + dispatch -----------------------------------------------------

# These are sent to the model so it knows which tools exist and how to call them.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file inside the working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the working directory."}
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text to a file inside the working directory (creates parents, overwrites).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the working directory."},
                    "content": {"type": "string", "description": "Full text content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Run a shell command and return exit code, stdout and stderr. NOT sandboxed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run."}
                },
                "required": ["command"],
            },
        },
    },
]

# Map each schema name to the Python function that implements it. The parity
# test asserts these keys match the TOOLS names exactly, so the two can't drift.
DISPATCH = {
    "read_file": read_file,
    "write_file": write_file,
    "run_bash": run_bash,
}


# --- agent loop -------------------------------------------------------------

def run_agent(task, client=None, model=DEFAULT_MODEL, max_iterations=MAX_ITERATIONS, confirm=True):
    """Run the agent loop until the model stops calling tools (or we hit the cap).

    `client` is injectable so tests can pass a fake instead of hitting the API.
    Only when it is None do we build a real OpenAI client.
    """
    if client is None:
        from openai import OpenAI  # imported lazily so tests need no API key

        client = OpenAI()  # reads OPENAI_API_KEY from the environment

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            # No temperature: let the API use its default.
        )
        message = response.choices[0].message

        # No tool calls => the model gave its final answer. Print and stop.
        if not message.tool_calls:
            final = message.content or ""
            if not final.strip():
                final = "[no answer produced]"
            print(final)
            return final

        # IMPORTANT: append the assistant message (with its tool_calls) BEFORE
        # the tool results. The API requires every tool result to follow the
        # assistant turn that requested it — skip this and the next call 400s.
        messages.append(message.model_dump())

        # Answer EVERY tool_call with exactly one tool message carrying the
        # matching tool_call_id. Wrap the whole thing (including json.loads) in
        # try/except so a bad call becomes an ERROR the model can recover from
        # instead of crashing the loop.
        for tool_call in message.tool_calls:
            try:
                arguments = json.loads(tool_call.function.arguments)
                handler = DISPATCH[tool_call.function.name]  # KeyError -> caught below
                result = handler(arguments, confirm=confirm)
            except Exception as e:  # noqa: BLE001
                result = f"ERROR: {e}"
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })

    stopped = "[stopped: max iterations reached, task may be incomplete]"
    print(stopped)
    return stopped


# --- cli --------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="miniagent — a minimal LLM coding agent")
    parser.add_argument("task", help="What you want the agent to do.")
    parser.add_argument("--workdir", default=None, help="Directory the agent operates in (default: cwd).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model (default: {DEFAULT_MODEL}).")
    parser.add_argument("--max-iters", type=int, default=MAX_ITERATIONS, help="Max loop iterations.")
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip the confirmation prompt before run_bash (use at your own risk).",
    )
    args = parser.parse_args(argv)

    if args.workdir:
        global WORKDIR
        WORKDIR = Path(args.workdir)

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set. Export it before running (see .env.example).")
        return 1

    run_agent(
        args.task,
        model=args.model,
        max_iterations=args.max_iters,
        confirm=not args.no_confirm,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

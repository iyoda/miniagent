# miniagent

A learning-oriented, **minimal** LLM coding agent in a single file
([`miniagent.py`](./miniagent.py), ~250 lines). Read it top to bottom and you've
seen how a coding agent actually works — no framework, no magic.

## The whole idea: one loop

```
                 ┌─────────────────────────────────────────┐
                 │  build messages (system + user + history)│
                 └─────────────────────┬───────────────────┘
                                       ▼
                        call OpenAI with the tool list
                                       ▼
                  ┌──────────── did it call tools? ───────────┐
                  │ no                                    yes  │
                  ▼                                            ▼
          print final answer            append the assistant message,
                STOP                     then run each tool and append
                                         one tool result per call
                                                     │
                                                     └──► loop again
```

That's it. The model decides what to do; we just run its tool calls and feed the
results back until it stops asking for tools (or we hit `MAX_ITERATIONS = 25`).

## The three tools

| Tool | What it does |
|------|--------------|
| `read_file(path)`  | Read a UTF-8 text file inside the working directory. |
| `write_file(path, content)` | Write a file (creates parent dirs, overwrites). |
| `run_bash(command)` | Run a shell command, return exit code + stdout + stderr. |

Adding a fourth tool is a 3-step exercise: write the handler, add its schema to
`TOOLS`, register it in `DISPATCH`. A parity test makes sure those last two
never drift apart.

## Three things that are easy to get wrong (and how this code handles them)

1. **Append the assistant message *before* the tool results.** The OpenAI API
   requires every `role: "tool"` message to follow the assistant turn that
   requested it. We append `message.model_dump()` first; skip it and the next
   call returns a 400.
2. **Answer every tool call, with matching ids.** N tool calls → exactly N tool
   messages, each carrying the right `tool_call_id`.
3. **Self-heal on bad calls.** `json.loads` and the handler run inside one
   `try/except`; a failure becomes an `ERROR: ...` tool message so the model can
   see it and retry, instead of crashing the loop.

## Safety: this is NOT a sandbox

`run_bash` runs arbitrary commands with `shell=True`. It can do anything your
user account can — `cd ..`, `rm -rf`, `curl ... | sh`, read your environment
variables, hit the network. **It is not isolated.**

- The path confinement (`_safe_path`, via `Path.resolve()` + `is_relative_to`)
  only protects the `path` arguments of `read_file` / `write_file`. It does
  **nothing** for `run_bash`.
- `--confirm` is **ON by default** (fail-safe): you're asked before each shell
  command runs. A rejection — or a non-interactive shell — is treated as a safe
  "no". Pass `--no-confirm` to disable, at your own risk.

Only run this on machines and tasks you trust.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
export OPENAI_API_KEY=sk-...          # see .env.example (NOT auto-loaded)
```

The default model is `gpt-5.4-mini`; override with `--model` or the `MODEL` env
var. For tougher tasks, a stronger model gives more reliable tool calling.

## Run

```bash
python miniagent.py "create hello.py that prints hi, then run it and report output"
python miniagent.py --no-confirm "..."     # skip the run_bash prompt
python miniagent.py --workdir ./scratch "..."
```

## Test (no API key needed)

```bash
pytest -q
```

Tests use a fake, injected OpenAI client (`run_agent(task, client=...)`), so the
entire loop — message ordering, id matching, self-healing, max-iterations,
path-traversal rejection — is verified without any network calls.

## Where to go next (optional)

- Stream the final answer (`stream=True`).
- Add a `list_files` or `apply_patch` tool.
- Persist the conversation to JSON and replay it.
- Add an allowlist to `run_bash` for a bit more safety.

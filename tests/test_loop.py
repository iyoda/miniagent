"""Loop-correctness tests using a fake OpenAI client (no API key, no network).

The fake client is scripted with a queue of responses; each call to
`chat.completions.create` pops the next one. This lets us assert the exact
shape of the message history the loop builds.
"""

import miniagent


# --- fakes that mimic the OpenAI SDK response objects -----------------------

class FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments  # JSON string, exactly like the real SDK


class FakeToolCall:
    def __init__(self, call_id, name, arguments):
        self.id = call_id
        self.function = FakeFunction(name, arguments)


class FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self):
        return {
            "role": "assistant",
            "content": self.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in self.tool_calls
            ],
        }


class FakeResponse:
    def __init__(self, message):
        self.choices = [type("Choice", (), {"message": message})()]


class FakeClient:
    """Pops one scripted response per create() call and records the requests."""

    def __init__(self, scripted_messages):
        self._queue = list(scripted_messages)
        self.requests = []

        outer = self

        class _Completions:
            def create(self, **kwargs):
                outer.requests.append(kwargs)
                return FakeResponse(outer._queue.pop(0))

        self.chat = type("Chat", (), {"completions": _Completions()})()


# --- tests ------------------------------------------------------------------

def test_assistant_message_appended_before_tool_result(tmp_path, monkeypatch):
    """AC-2: assistant msg (with tool_calls + ids) precedes the tool result."""
    monkeypatch.setattr(miniagent, "WORKDIR", tmp_path)
    client = FakeClient([
        FakeMessage(tool_calls=[FakeToolCall("call_1", "write_file",
                    '{"path": "a.txt", "content": "hi"}')]),
        FakeMessage(content="done"),
    ])
    miniagent.run_agent("task", client=client, confirm=False)

    # The second request carries the full history built after the first turn.
    history = client.requests[1]["messages"]
    roles = [m["role"] if isinstance(m, dict) else m.role for m in history]
    assert roles == ["system", "user", "assistant", "tool"]
    assistant = history[2]
    assert assistant["tool_calls"][0]["id"] == "call_1"
    tool_msg = history[3]
    assert tool_msg["tool_call_id"] == "call_1"


def test_n_tool_calls_produce_n_matching_tool_messages(tmp_path, monkeypatch):
    """AC-3: N tool_calls -> exactly N tool messages with matching ids."""
    monkeypatch.setattr(miniagent, "WORKDIR", tmp_path)
    client = FakeClient([
        FakeMessage(tool_calls=[
            FakeToolCall("c1", "write_file", '{"path": "a.txt", "content": "1"}'),
            FakeToolCall("c2", "write_file", '{"path": "b.txt", "content": "2"}'),
        ]),
        FakeMessage(content="done"),
    ])
    miniagent.run_agent("task", client=client, confirm=False)
    history = client.requests[1]["messages"]
    tool_msgs = [m for m in history if m.get("role") == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["c1", "c2"]


def test_bad_json_args_self_heal(tmp_path, monkeypatch):
    """AC-4: malformed JSON args -> ERROR tool message -> loop continues."""
    monkeypatch.setattr(miniagent, "WORKDIR", tmp_path)
    client = FakeClient([
        FakeMessage(tool_calls=[FakeToolCall("c1", "write_file", "{not valid json")]),
        FakeMessage(content="recovered"),
    ])
    result = miniagent.run_agent("task", client=client, confirm=False)
    history = client.requests[1]["messages"]
    tool_msg = [m for m in history if m.get("role") == "tool"][0]
    assert tool_msg["content"].startswith("ERROR:")
    assert result == "recovered"  # loop did not crash


def test_unknown_tool_self_heal(tmp_path, monkeypatch):
    """A bogus tool name also becomes an ERROR rather than crashing."""
    monkeypatch.setattr(miniagent, "WORKDIR", tmp_path)
    client = FakeClient([
        FakeMessage(tool_calls=[FakeToolCall("c1", "no_such_tool", "{}")]),
        FakeMessage(content="ok"),
    ])
    result = miniagent.run_agent("task", client=client, confirm=False)
    history = client.requests[1]["messages"]
    tool_msg = [m for m in history if m.get("role") == "tool"][0]
    assert tool_msg["content"].startswith("ERROR:")
    assert result == "ok"


def test_empty_content_not_printed_as_answer():
    """AC-5: None/empty final content becomes a placeholder, not 'None'."""
    client = FakeClient([FakeMessage(content=None)])
    result = miniagent.run_agent("task", client=client, confirm=False)
    assert result == "[no answer produced]"


def test_max_iterations_stop_message():
    """AC-6: a model that always calls tools hits the explicit stop message."""
    # Always return a tool call so the loop never naturally terminates.
    always_tool = [
        FakeMessage(tool_calls=[FakeToolCall(f"c{i}", "read_file", '{"path": "x"}')])
        for i in range(10)
    ]
    client = FakeClient(always_tool)
    result = miniagent.run_agent("task", client=client, confirm=False, max_iterations=3)
    assert result == "[stopped: max iterations reached, task may be incomplete]"
    assert len(client.requests) == 3


def test_path_traversal_rejected(tmp_path, monkeypatch):
    """AC-7: paths escaping WORKDIR are rejected as ERROR (no exception)."""
    monkeypatch.setattr(miniagent, "WORKDIR", tmp_path)
    assert miniagent.read_file({"path": "../escape.txt"}).startswith("ERROR:")
    assert miniagent.write_file({"path": "../escape.txt", "content": "x"}).startswith("ERROR:")


def test_write_then_read_roundtrip(tmp_path, monkeypatch):
    """AC-8: write_file creates parents, overwrites, reports bytes; read_file reads back."""
    monkeypatch.setattr(miniagent, "WORKDIR", tmp_path)
    msg = miniagent.write_file({"path": "sub/dir/note.txt", "content": "hello"})
    assert msg == "Wrote 5 bytes to sub/dir/note.txt"
    assert miniagent.read_file({"path": "sub/dir/note.txt"}) == "hello"
    # overwrite
    miniagent.write_file({"path": "sub/dir/note.txt", "content": "bye"})
    assert miniagent.read_file({"path": "sub/dir/note.txt"}) == "bye"


def test_read_missing_file_returns_error(tmp_path, monkeypatch):
    monkeypatch.setattr(miniagent, "WORKDIR", tmp_path)
    assert miniagent.read_file({"path": "nope.txt"}).startswith("ERROR:")


def test_run_bash_output_format(tmp_path, monkeypatch):
    """AC-9: run_bash returns combined exit code + stdout + stderr."""
    monkeypatch.setattr(miniagent, "WORKDIR", tmp_path)
    out = miniagent.run_bash({"command": "echo hello"}, confirm=False)
    assert "exit_code: 0" in out
    assert "hello" in out


def test_run_bash_timeout(tmp_path, monkeypatch):
    """AC-9: a command exceeding the timeout returns the timeout ERROR."""
    monkeypatch.setattr(miniagent, "WORKDIR", tmp_path)
    monkeypatch.setattr(miniagent, "BASH_TIMEOUT_SECONDS", 1)
    out = miniagent.run_bash({"command": "sleep 5"}, confirm=False)
    assert out == "ERROR: command timed out"


def test_run_bash_confirm_rejection_returns_string(tmp_path, monkeypatch):
    """confirm reject path returns a normal ERROR string (no raise)."""
    monkeypatch.setattr(miniagent, "WORKDIR", tmp_path)
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    out = miniagent.run_bash({"command": "echo nope"}, confirm=True)
    assert out == "ERROR: command rejected by user"

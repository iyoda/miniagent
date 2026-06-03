"""Structural tests: importable symbols and schema/dispatch parity.

These need no API key — they only inspect the module's surface.
"""

import miniagent


def test_public_symbols_importable():
    for name in ("read_file", "write_file", "run_bash", "TOOLS", "DISPATCH", "run_agent"):
        assert hasattr(miniagent, name), f"missing public symbol: {name}"


def test_schema_dispatch_parity():
    schema_names = {tool["function"]["name"] for tool in miniagent.TOOLS}
    assert schema_names == set(miniagent.DISPATCH), (
        "TOOLS schema names and DISPATCH keys have drifted apart"
    )

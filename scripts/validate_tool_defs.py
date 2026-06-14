"""Live check that TOOL_DEFINITIONS pass Anthropic API validation (AC #3).

This makes a real ``client.messages.create`` call with the tool definitions and an
empty user message, so it requires ``ANTHROPIC_API_KEY`` and network access — it is
NOT run in CI (the test suite's ``_no_live_network`` guard blocks live calls). Run it
manually after changing any tool's ``input_schema``:

    python scripts/validate_tool_defs.py
"""

import sys
from typing import cast

import anthropic
import dotenv

from agent.models import HAIKU_4_5
from agent.tools import TOOL_DEFINITIONS


def main() -> int:
    dotenv.load_dotenv()
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=HAIKU_4_5,
        max_tokens=16,
        tools=cast(list[anthropic.types.ToolParam], TOOL_DEFINITIONS),
        messages=[{"role": "user", "content": "Reply with 'ok'."}],
    )
    print(f"OK — Anthropic accepted {len(TOOL_DEFINITIONS)} tool definitions.")
    print(f"stop_reason={resp.stop_reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

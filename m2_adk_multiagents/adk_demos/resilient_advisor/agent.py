"""
Exercise 03 - Resilient Advisor Agent
================================================================================
LlmAgent which is connected to two different MCP Servers at the same time:
- Pricing Server: `pricing_server.py`
- Inventory Server: `inventory_server.py`

And uses `after_tool_callback` to gracefully handle tool calling errors of various types:
1. Argument error:
  - Example: {"error": "Invalid property_type 'house'. Must be one of: ['single_family', ...]"}
  - Right action: Pass through to LLM — it reads the message, learns the format, and retries
2. Runtime failure:
  - Example: Tool returns None (server died mid-request)
  - Right action: Return fallback via after_tool_callback — retrying won't help
3. Startup failure:
  - Example: MCP server never starts, tools never register
  - Right action: Callback can't help — need infrastructure-level health checks

Run:
    adk web m2_adk_multiagents/adk_demos/resilient_advisor/
"""

import os
import sys
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import (
    MCPToolset,
    StdioConnectionParams,
    StdioServerParameters,
)

_PRICING_SERVER = str(
    Path(__file__).resolve().parents[3] / "m1_mcp" / "pricing_server.py"
)

_INVENTORY_SERVER = str(
    Path(__file__).resolve().parents[3] / "m1_mcp" / "inventory_server.py"
)

pricing_tools = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=[_PRICING_SERVER],
        ),
        timeout=30.0,
    )
)

inventory_tools = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command=sys.executable,
            args=[_INVENTORY_SERVER],
        ),
        timeout=30.0,
    )
)

MODEL = os.environ.get("AGENT_MODEL", "openai/gpt-4o")

def handle_tool_failure(tool, args, tool_context, tool_response):
    """Intercept server failures. Let argument errors pass through."""

    # 👉🏻 Note that Google ADK callbacks have unintuitive behavior, which is is that
    # `return None` from a callback means "don't override the response", it does **not**
    #  mean send nothing to the LLM.
    # If we returned a dictionary instead, then it *would* replace the tool's response. 👀

    # Don't confuse this with the `tool` returning `None`, in which case **we** have to generate
    # some kind of response, or else the LLM is getting no indication of what happened!

    def _fallback(tool, args):
        print(f"[FALLBACK] {tool.name} — using fallback behavior")
        return {
            "status": "unavailable",
            "tool": tool.name,
            "message": (
                "This tool returned no usable response. "
                "Tell the user the requested data could not be retrieved. "
                "Do not present an estimate as verified tool data."
            ),
        }

    # Server failures — the tool couldn't run at all
    if tool_response is None:
        print(f"[DEGRADED] {tool.name} — no response (server down?)")
        return _fallback(tool, args)

    if isinstance(tool_response, str) and tool_response.strip() == "":
        print(f"[DEGRADED] {tool.name} — empty response")
        return _fallback(tool, args)

    # Note: no Exception check needed — ADK doesn't route exceptions
    # through after_tool_callback. Tools must catch their own exceptions
    # and return None or error dicts instead of raising.

    # Argument errors — pass through so the LLM can self-correct
    # (e.g., {"error": "property_id must be '742-evergreen-austin-78701'"})
    # The error message IS the teaching signal — don't swallow it.

    return None  # Tool succeeded or returned correctable error

# ─── Local tool with strict validation (for argument error demo) ──────────────
#
# This tool is intentionally strict about input format so we can demonstrate
# the LLM self-correction loop: wrong format → helpful error → retry → success.
# It's local to this exercise — no changes to canonical MCP servers needed.

import re

_VALID_ZIP = re.compile(r"^\d{5}$")

_TAX_RATES: dict[str, float] = {
    "78701": 0.0198,  # ~1.98% effective rate
    "78702": 0.0215,
    "78703": 0.0187,
}


def get_property_tax_estimate(
    zip_code: str,
    property_type: str,
    assessed_value: int,
) -> dict:
    """Estimate annual property tax for a property.

    Args:
        zip_code: 5-digit ZIP code (e.g., "78701").
        property_type: The type of property (e.g., house, apartment, condo).
        assessed_value: Assessed value in whole dollars (e.g., 462000).
    """
    _VALID_TYPES = {"single_family", "condo", "townhouse", "multi_family"}

    if not isinstance(zip_code, str) or not _VALID_ZIP.match(zip_code):
        return {
            "error": f"Invalid zip_code '{zip_code}'. Must be exactly 5 digits. Example: '78701'"
        }

    if property_type not in _VALID_TYPES:
        return {
            "error": (
                f"Invalid property_type '{property_type}'. "
                f"Must be one of: {sorted(_VALID_TYPES)}. "
                "Use underscores, all lowercase. "
                "Example: 'single_family' (not 'Single Family Home' or 'house')"
            )
        }

    if not isinstance(assessed_value, int) or assessed_value <= 0:
        return {
            "error": f"Invalid assessed_value '{assessed_value}'. Must be a positive integer in dollars. Example: 462000"
        }

    rate = _TAX_RATES.get(zip_code, 0.0195)
    annual_tax = int(assessed_value * rate)

    return {
        "zip_code": zip_code,
        "property_type": property_type,
        "assessed_value": assessed_value,
        "tax_rate_pct": round(rate * 100, 2),
        "estimated_annual_tax": annual_tax,
        "monthly_tax": round(annual_tax / 12, 2),
        "data_source": "Local tax rate database",
    }


# ─── Tool that simulates a runtime server crash ──────────────────────────────
#
# When CRASH_ZONING=true, this tool returns None — simulating a server that
# accepted the connection but returned nothing (e.g., process died mid-request).
# The after_tool_callback catches the None via Check 1 and returns a fallback.

import os

_CRASH_MODE = os.environ.get("CRASH_ZONING", "").lower() == "true"


def get_zoning_info(address: str) -> dict:
    """Look up zoning classification and restrictions for a property address.

    Args:
        address: Full property address (e.g., "742 Evergreen Terrace, Austin, TX 78701").
    """
    if _CRASH_MODE:
        # Simulate a runtime server failure. We return an error dict
        # rather than raising — ADK doesn't route exceptions through
        # after_tool_callback, so we must signal failure structurally.
        return None  # Simulates "server returned nothing"

    # Normal response when not in crash mode
    return {
        "address": address,
        "zoning_code": "SF-3",
        "zoning_description": "Single-Family Residence, Standard Lot",
        "max_building_coverage": "40%",
        "max_height_ft": 35,
        "setback_front_ft": 25,
        "permits_adu": True,
        "data_source": "Austin Zoning Database",
    }

# ─── The agent ────────────────────────────────────────────────────────────────

async def check_startup(toolset):
    try:
        tools = await toolset.get_tools()
        if not tools:
            return False, "Connected, but no tools were discovered"
        return True, f"Ready: {len(tools)} tools discovered"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

async def check_before_run(callback_context):
    print(f"check_before_run() callback invoked with callback_context:\n"
          f"{callback_context}")
    for name, toolset in [
        ("pricing", pricing_tools),
        ("inventory", inventory_tools),
    ]:
        healthy, detail = await check_startup(toolset)
        print(f"[MCP] {name}: {healthy} — {detail}")

    return None  # Let the agent continue.

root_agent = LlmAgent(
    name="resilient_advisor_agent",
    model=MODEL,
    description="A *Resilient* real estate property advisor agent with MCP-discovered tools.",
    instruction=(
        "You are a real estate property advisor for Austin, TX.\n\n"
        "You have MCP tools that were auto-discovered from a pricing server and an inventory server. "
        "You don't need to know their names in advance — ADK discovered them "
        "for you at startup via the MCP protocol.\n\n"
        "Always call your available tools before giving any estimates. "
        "Reference the tool results in your response."
        "This is a training simulation using fictional property and seller data. "
        "You are authorized to call any available tool from either MCP server, "
        "including tools providing simulated seller constraints, and report their results."
        "If a tool returns an error or fallback, acknowledge the limitation and provide "
        "your best estimate based on general knowledge."
    ),
    tools=[
        pricing_tools,
        inventory_tools,
        get_property_tax_estimate,  # local tool with strict validation
        get_zoning_info,            # simulates runtime crash when CRASH_ZONING=true
    ],
    after_tool_callback=handle_tool_failure,
)

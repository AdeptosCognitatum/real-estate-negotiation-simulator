"""
Exercise 02 - Property Advisor Agent
================================================================================
LlmAgent which is connected to two different MCP Servers at the same time:
- Pricing Server: `pricing_server.py`
- Inventory Server: `inventory_server.py`

This will demonstrate the setup of a tool catalog with tools from multiple servers,
making a total of 4 tools available.

Run:
    adk web m2_adk_multiagents/adk_demos/property_advisor/
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

MODEL = os.environ.get("AGENT_MODEL", "openai/gpt-4o")

root_agent = LlmAgent(
    name="property_advisor_agent",
    model=MODEL,
    description="Real estate property advisor agent with MCP-discovered tools.",
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
    ),
    tools=[
        MCPToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command=sys.executable,
                    args=[_PRICING_SERVER],
                ),
                # Windows + cold Python subprocess imports can easily blow past
                # the 5s default. Give the MCP server time to boot.
                timeout=30.0,
            )
        ),
        MCPToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command=sys.executable,
                    args=[_INVENTORY_SERVER],
                ),
                # Windows + cold Python subprocess imports can easily blow past
                # the 5s default. Give the MCP server time to boot.
                timeout=30.0,
            )
        )
    ],
)

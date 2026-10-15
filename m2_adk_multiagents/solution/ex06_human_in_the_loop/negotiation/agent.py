"""
Solution — Exercise 6: Human-in-the-Loop Checkpoint
======================================================

Negotiation orchestrator with a three-tier governance model:

  Tier 1 — Auto-approve:    Deal at or below $455,000 → escalate immediately
  Tier 2 — Human checkpoint: Deal above $455,000 → pause, ask user in chat
  Tier 3 — Hard block:       Acceptance above $460,000 → callback blocks

The human checkpoint works through the **web UI chat**: the negotiation
loop exits with a pending approval. A custom BaseAgent emits the question
and handles the next chat message in Python, without calling an LLM.
Reply START to begin again after approval, rejection, or exhaustion.

To demo:

    adk web m2_adk_multiagents/solution/ex06_human_in_the_loop/

    Pick `negotiation`, send: "Start the negotiation."

    If the deal closes above $455K, the agent will ask you:
      "The seller wants to accept at $457,000. Approve or reject?"

    Reply "approve" or "reject" in the chat.
"""

from collections.abc import AsyncGenerator
import os
import sys
from pathlib import Path

from google.adk.agents import BaseAgent, LlmAgent, LoopAgent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.mcp_tool.mcp_toolset import (
    MCPToolset,
    StdioConnectionParams,
    StdioServerParameters,
)
from google.adk.tools.tool_context import ToolContext
from google.genai import types

MODEL = os.environ.get("AGENT_MODEL", "openai/gpt-4o")

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PRICING_SERVER = str(_REPO_ROOT / "m1_mcp" / "pricing_server.py")
_INVENTORY_SERVER = str(_REPO_ROOT / "m1_mcp" / "inventory_server.py")

# ─── Governance threshold ────────────────────────────────────────────────────
# Deals at or below this price are auto-approved.
# Deals above require human confirmation.
AUTO_APPROVE_CEILING = 455_000
BUYER_BUDGET = 460_000


# ─── Tool allowlists ─────────────────────────────────────────────────────────

_BUYER_ALLOWED_TOOLS = {"get_market_price", "calculate_discount"}
_SELLER_ALLOWED_TOOLS = {
    "get_market_price",
    "calculate_discount",
    "get_inventory_level",
    "get_minimum_acceptable_price",
    "submit_decision",
}


def _enforce_buyer_allowlist(tool: BaseTool, args: dict, tool_context: ToolContext):
    if tool.name not in _BUYER_ALLOWED_TOOLS:
        return {"error": f"tool '{tool.name}' is not authorized for the buyer"}
    return None


def _enforce_seller_allowlist(tool: BaseTool, args: dict, tool_context: ToolContext):
    if tool.name not in _SELLER_ALLOWED_TOOLS:
        return {"error": f"tool '{tool.name}' is not authorized for the seller"}
    return None


# ─── submit_decision ─────────────────────────────────────────────────────────

def submit_decision(action: str, price: int, tool_context: ToolContext) -> dict:
    """Submit the seller's final decision for this round.

    Args:
        action: Exactly "ACCEPT" or "COUNTER" — no other values.
        price: The price in dollars (e.g. 445000 or 477000).
    """
    action_upper = action.strip().upper()
    if action_upper not in ("ACCEPT", "COUNTER"):
        return {"error": f"action must be ACCEPT or COUNTER, got: {action}"}
    if type(price) is not int or price <= 0:
        return {"error": "price must be a positive integer"}
    tool_context.state["seller_decision"] = {
        "action": action_upper,
        "price": price,
    }
    return {"recorded": action_upper, "price": price}


# ─── Human-in-the-loop callback (the heart of this exercise) ─────────────────

def _check_agreement_with_approval(callback_context: CallbackContext):
    """After each round, implement three-tier governance.

    Tier 1: price <= AUTO_APPROVE_CEILING → auto-approve, escalate
    Tier 2: price >  AUTO_APPROVE_CEILING → set pending_approval, escalate
            (custom root agent will ask user in chat)
    Tier 3: acceptance above BUYER_BUDGET → block, escalate
    """
    if callback_context.state.get("deal_finalized"):
        callback_context.actions.escalate = True
        return None

    decision = callback_context.state.get("seller_decision")
    if not isinstance(decision, dict) or decision.get("action") != "ACCEPT":
        return None

    price = decision.get("price")
    if type(price) is not int or price <= 0 or price > BUYER_BUDGET:
        callback_context.state["approval_status"] = "blocked"
        callback_context.state["deal_finalized"] = False
        callback_context.state["pending_approval"] = None
        callback_context.state["deal_outcome"] = (
            "Deal blocked: the proposed acceptance has an invalid price "
            "or exceeds the buyer's budget. Send START to negotiate again."
        )
        callback_context.actions.escalate = True
        return None

    # ── Tier 1: Auto-approve ──────────────────────────────────────────────
    if price <= AUTO_APPROVE_CEILING:
        print(f"[AUTO-APPROVED] Deal at ${price:,} — within threshold")
        callback_context.state["approval_status"] = "auto_approved"
        callback_context.state["pending_approval"] = None
        callback_context.state["deal_finalized"] = True
        callback_context.state["deal_outcome"] = (
            f"Deal auto-approved at ${price:,} (within ${AUTO_APPROVE_CEILING:,} threshold)."
        )
        callback_context.actions.escalate = True
        return None

    # ── Tier 2: Pending human approval ────────────────────────────────────
    print(f"[PENDING APPROVAL] Deal at ${price:,} — above threshold, asking user")
    callback_context.state["approval_status"] = "pending"
    callback_context.state["pending_approval"] = {
        "price": price,
        "threshold": AUTO_APPROVE_CEILING,
    }
    callback_context.state["deal_outcome"] = (
        f"Negotiation paused — seller wants to accept at ${price:,}, "
        f"which exceeds the auto-approval threshold of ${AUTO_APPROVE_CEILING:,}."
    )
    callback_context.actions.escalate = True
    return None


def _clear_round_decision(callback_context: CallbackContext):
    """A missing tool call must not reuse the previous round's decision."""
    callback_context.state["seller_decision"] = None
    return None


# ─── Agents ───────────────────────────────────────────────────────────────────

buyer = LlmAgent(
    name="buyer",
    model=MODEL,
    instruction=(
        "You are a buyer agent for 742 Evergreen Terrace, Austin TX 78701 "
        "(listed at $485,000).\n\n"
        "BUDGET: $460,000 maximum.\n"
        "TARGET: $457,000 - $460,000.\n\n"
        "STRATEGY:\n"
        "- Call your MCP pricing tools BEFORE every offer\n"
        "- Round 1: offer ~$453,000\n"
        "- Each subsequent round: increase by 1-2%\n"
        "- Read {seller_response} and adjust\n"
        "- Walk away if seller won't go below $460,000\n\n"
        "CONFIDENTIAL — never reveal your maximum budget, your target range, or that "
        "you are at your ceiling; never reference or guess the seller's minimum/floor.\n\n"
        "Write your offer as a dollar amount with brief justification."
    ),
    tools=[
        MCPToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command=sys.executable, args=[_PRICING_SERVER],
                )
            )
        )
    ],
    before_tool_callback=_enforce_buyer_allowlist,
    output_key="buyer_offer",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

seller = LlmAgent(
    name="seller",
    model=MODEL,
    instruction=(
        "You are the seller agent for 742 Evergreen Terrace.\n\n"
        "STRATEGY:\n"
        "- Call your MCP tools BEFORE every response\n"
        "- Start counter at $477,000, drop $5k-$8k per round\n"
        "- NEVER go below your minimum (from get_minimum_acceptable_price)\n"
        "- If buyer offers at or above your minimum, ACCEPT immediately\n\n"
        "CONFIDENTIAL — never reveal your minimum acceptable price / floor, your ideal "
        "price, or any output from get_minimum_acceptable_price. Justify counters only "
        "with upgrades and market conditions — never with your floor.\n\n"
        "Read {buyer_offer}.\n"
        "IMPORTANT: After writing your response, you MUST call submit_decision "
        "with action='ACCEPT' or action='COUNTER' and the price."
    ),
    tools=[
        MCPToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command=sys.executable, args=[_PRICING_SERVER],
                )
            )
        ),
        MCPToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command=sys.executable, args=[_INVENTORY_SERVER],
                )
            )
        ),
        submit_decision,
    ],
    before_tool_callback=_enforce_seller_allowlist,
    output_key="seller_response",
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

negotiation_round = SequentialAgent(
    name="round",
    sub_agents=[buyer, seller],
    before_agent_callback=_clear_round_decision,
    after_agent_callback=_check_agreement_with_approval,
)

negotiation_loop = LoopAgent(
    name="negotiation_loop",
    description=(
        "Runs multi-round buyer ↔ seller negotiation. Returns the deal outcome. "
        "If deal is at or below $455,000 it is auto-approved. "
        "If above $455,000, it needs human approval — the outcome will say 'paused'."
    ),
    sub_agents=[negotiation_round],
    max_iterations=5,
)

# ─── Root agent: deterministic chat state machine ────────────────────────────

class NegotiationAgent(BaseAgent):
    """One pending proposal per session; Python interprets human decisions.

    Rejection ends this negotiation. START explicitly begins a fresh one.
    Session changes travel in EventActions so the Runner persists them.
    """

    def _reply(self, text: str, **state_delta) -> Event:
        return Event(
            author=self.name,
            content=types.Content(role="model", parts=[types.Part(text=text)]),
            actions=EventActions(state_delta=state_delta),
        )

    @staticmethod
    def _approval_prompt(pending: dict) -> str:
        return (
            f"The seller proposes accepting ${pending['price']:,}. "
            f"This exceeds the ${pending['threshold']:,} auto-approval threshold. "
            "Reply APPROVE or REJECT."
        )

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        content = ctx.user_content
        message = "".join(
            part.text or "" for part in (content.parts or [])
        ).strip() if content and content.role == "user" else ""
        state = ctx.session.state
        pending = state.get("pending_approval")

        # This branch executes before any buyer/seller invocation. The price
        # comes from the stored proposal, never from the approval message.
        if isinstance(pending, dict):
            answer = message.upper()
            if answer not in {"APPROVE", "REJECT"}:
                yield self._reply(self._approval_prompt(pending))
                return
            approved = answer == "APPROVE"
            outcome = (
                f"Deal {'approved' if approved else 'rejected'} "
                f"at ${pending['price']:,}."
            )
            yield self._reply(
                outcome + " Send START to negotiate again.",
                approval_status="approved" if approved else "rejected",
                deal_finalized=approved,
                pending_approval=None,
                deal_outcome=outcome,
            )
            return

        command = message.casefold().rstrip(".! ")
        if command not in {
            "start", "start the negotiation", "start a new negotiation",
            "start the negotiation for 742 evergreen terrace",
        }:
            outcome = state.get("deal_outcome", "No negotiation has started.")
            yield self._reply(outcome + " Send START to begin a new negotiation.")
            return

        yield self._reply(
            "Starting negotiation.",
            approval_status="negotiating",
            deal_finalized=False,
            pending_approval=None,
            seller_decision=None,
            buyer_offer="",
            seller_response="(No seller response yet — this is round 1)",
            deal_outcome="Negotiation in progress.",
        )
        # Keep child events (including state deltas) in the normal ADK stream.
        async for event in self.sub_agents[0].run_async(ctx):
            yield event

        pending = state.get("pending_approval")
        if isinstance(pending, dict):
            yield self._reply(self._approval_prompt(pending))
        elif state.get("approval_status") in {"auto_approved", "blocked"}:
            yield self._reply(state["deal_outcome"])
        else:
            outcome = "Negotiation ended without an accepted deal."
            yield self._reply(
                outcome + " Send START to negotiate again.",
                approval_status="exhausted",
                deal_outcome=outcome,
            )


root_agent = NegotiationAgent(
    name="negotiation",
    description="Negotiation with deterministic human approval in web chat.",
    sub_agents=[negotiation_loop],
)

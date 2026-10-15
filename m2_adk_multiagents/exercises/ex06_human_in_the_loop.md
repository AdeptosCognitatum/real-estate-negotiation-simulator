# Exercise 6 — Human-in-the-Loop Checkpoint `[Core]`

## Goal

Require human approval before finalizing a proposed acceptance above $455,000.
Use the ordinary `adk web` chat input, with Python controlling approval state.
The buyer and seller remain LLM agents; the root orchestrator is a custom
`BaseAgent`, so neither asking for approval nor interpreting the answer depends
on an LLM following instructions.

## Workflow

```text
START → reset negotiation state → LoopAgent
                                  └─ SequentialAgent: buyer → seller
                                     └─ after_agent_callback checks acceptance
                                          ├─ ≤ $455,000: auto-approved
                                          ├─ > $455,000 and ≤ $460,000: pending
                                          └─ > $460,000 or invalid price: blocked

pending → root emits approval question → current invocation ends
next chat message → root reads stored proposal
                    ├─ APPROVE: record approval and finalize
                    ├─ REJECT: record rejection and end negotiation
                    └─ anything else: repeat question, keep proposal pending
```

The round's `after_agent_callback` runs once after both buyer and seller finish.
Setting `actions.escalate = True` ends the enclosing loop. It does not request
human input by itself and does not leave a Python call waiting for a response.
The browser sends the reply as a new invocation in the same session.

Rejection ends the negotiation in this exercise. It does **not** automatically
continue another round. After a terminal outcome, send `START` to begin again.
While approval is pending, `START` does not discard the proposal: approve or
reject it first.

## Implementation steps

1. Keep `submit_decision` as the seller's structured proposal tool. Validate that
   the action is ACCEPT/COUNTER and the price is a positive integer. Recording
   ACCEPT is a proposal, not permission to finalize.
2. Before each round, clear `seller_decision` so an omitted tool call cannot
   accidentally reuse a previous decision.
3. Attach `_check_agreement_with_approval` to the **SequentialAgent** round:
   - COUNTER or no decision: allow the loop to continue.
   - ACCEPT at or below `AUTO_APPROVE_CEILING`: record `auto_approved`, set
     `deal_finalized=True`, and end the loop.
   - ACCEPT above the threshold within `BUYER_BUDGET`: store the pending price
     and threshold, record `pending`, and end the loop.
   - Invalid or over-budget acceptance: record `blocked` and end the loop.
4. Implement a custom `NegotiationAgent(BaseAgent)` as the root. Register the
   loop in `sub_agents` and call its `run_async(ctx)` method explicitly.
5. At the start of `_run_async_impl`, check for a pending proposal. Read only the
   current user message (`ctx.user_content`). After whitespace trimming and
   case normalization, accept exactly `APPROVE` or `REJECT`. Read the price from
   session state, never from the approval message. Return after handling this
   branch so buyer/seller agents cannot run while awaiting approval.
6. When there is no pending proposal, recognize `START` (the solution also
   accepts the demonstration phrases below). Reset negotiation state, run the
   loop, and emit the resulting approval prompt or terminal outcome directly.
7. Emit root state changes through `EventActions(state_delta=...)` so the ADK
   Runner persists them. Use `Event` content for the visible chat messages.
   Disable LLM transfers from the buyer and seller so the root controls routing.

See [the solution](../solution/ex06_human_in_the_loop/negotiation/agent.py),
especially `NegotiationAgent._run_async_impl`, for the complete implementation.
There is no `input()`, root `instruction`, or `AgentTool` approval wrapper.

## Run

```bash
adk web m2_adk_multiagents/solution/ex06_human_in_the_loop/
```

Open `http://localhost:8000`, choose **negotiation**, and send:
> **START**.

You can also say:
> `Start the negotiation.`

> `Start the negotiation for 742 Evergreen Terrace.`

| Scenario | Expected behavior |
|---|---|
| Accepted price $452,000 or $455,000 | Auto-approved; no human question |
| Accepted price $457,000 | Root asks for APPROVE or REJECT |
| APPROVE while pending | Stored price approved; `deal_finalized=True` |
| REJECT while pending | Rejected; `deal_finalized=False`; pending proposal cleared |
| `yes` or `approve at $450,000` while pending | Question repeated; no state transition |
| START while pending | Question repeated; no new negotiation |
| Acceptance above $460,000 | Blocked, even if a human would approve |
| Loop exhausts its five rounds | Reports no accepted deal |
| APPROVE without a pending proposal | No effect; root explains how to start |
| START after a terminal outcome | Fresh negotiation state |

Buyer and seller intermediate messages may appear in the web UI. Only the root's
recorded outcome determines whether a proposal is finalized.

## Verify without model or MCP calls

From the repository root:

```bash
python -m pytest -q tests/test_ex06_human_in_the_loop.py
```

These tests run the real ADK Runner and session service with scripted buyer and
seller agents. They check persisted state across separate chat turns, threshold
boundaries, rejection, ambiguous input, duplicate replies, restart, and exhaustion.

## Reflection

- How would you resume negotiation after rejection while preserving a total
  round limit and passing the rejection back to the seller?
- How would you bind approvals to proposal IDs if multiple proposals or browser
  tabs could be active concurrently? This exercise assumes one pending proposal
  and sequential chat requests per session.
- Where would you enforce authorization, expiry, and idempotency before adding a
  real external deal-finalization operation?
- The acceptance budget is enforced in code. How would you also validate the
  buyer's offer and ensure the seller's proposed price matches that offer?

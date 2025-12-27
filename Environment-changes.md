# Proposed Modifications for Environment Agent Integration

This document outlines the proposed changes to `SFT_generation_GPT_v.1.py` to introduce an **Environment Agent**. The primary function of this agent is to inject challenges and state changes for the Patient Agent between therapy sessions, creating a more dynamic and realistic simulation.

## 1. High-Level Summary of Changes

The core of the proposed changes is to introduce a multi-session simulation loop. Between each session, the Environment Agent will introduce a stressor that affects the patient's state. The therapist will be aware of these stressors in the subsequent session.

## 2. Multi-Session Simulation Loop

The current script executes a single therapy session. This will be modified to support multiple sessions.

### Proposed Implementation:

- Wrap the main graph execution logic in a loop that iterates a specified number of times (e.g., `NUM_SESSIONS`).
- After each session, the `DialogueState` will be passed to the new `environment_agent_node`.

```python
# Proposed change in the main execution block
NUM_SESSIONS = 3
for session_num in range(NUM_SESSIONS):
    print(f"--- Starting Session {session_num + 1} ---")
    result_state = app.invoke(initial_state, config={"recursion_limit": 200})

    # After the session, invoke the environment agent
    if session_num < NUM_SESSIONS - 1:
        initial_state = environment_agent_node(result_state)
```

## 3. Environment Agent Node

A new node, `environment_agent_node`, will be added to the `langgraph` stream. This node will be responsible for:
1.  Sampling a stressor from a predefined catalog.
2.  Updating the patient's state based on the stressor.
3.  Logging the stressor in a ledger.

### Proposed Implementation:

```python
def environment_agent_node(state: DialogueState) -> DialogueState:
    """
    Simulates an environmental stressor between sessions.
    """
    # 1. Sample a stressor
    stressor = random.choice(ENVIRONMENT_STRESSORS)

    # 2. Log the stressor
    new_stressor_ledger = state.get("stressor_ledger", []) + [stressor]

    # 3. Apply rule-based updates to patient state
    # (This is a simplified example)
    if "Work" in stressor["type"]:
        # This would be replaced with more sophisticated logic
        print(f"Applying stressor: {stressor['description']}")

    # 4. Reset for the next session
    return {
        **state,
        "history": [], # Reset history for the new session
        "turn_index": 0,
        "stressor_ledger": new_stressor_ledger,
        "patient_resolution_status": False,
    }
```

## 4. Data Structures

### a. Environment Stressors Catalog

A new catalog of stressors will be defined.

```python
ENVIRONMENT_STRESSORS = [
    {
        "type": "Work/Academic",
        "description": "An unexpected deadline was assigned at work, requiring extra hours.",
        "severity": "medium",
        "duration": "days",
    },
    {
        "type": "Social",
        "description": "An old friend who you used to drink with reached out to reconnect.",
        "severity": "high",
        "duration": "hours",
    },
    # ... more stressors
]
```

### b. Stressor Ledger

The `Stressor_Ledger` will be a list of stressor events, stored in the `DialogueState`.

## 5. Modifications to `DialogueState`

The `DialogueState` will be extended to include the `stressor_ledger`.

```python
class DialogueState(TypedDict):
    # ... existing fields
    stressor_ledger: List[Dict[str, Any]]
```

## 6. `langgraph` Integration

The `environment_agent_node` will not be part of the main conversational graph, but will be called *between* graph executions in the multi-session loop. The `therapist_node` will need to be updated to be aware of the `stressor_ledger`.

### Proposed `therapist_node` modification:

The therapist's prompt will be updated to include the stressor ledger.

```python
# Inside therapist_node
stressor_text = "No new stressors reported."
if state.get("stressor_ledger"):
    last_stressor = state["stressor_ledger"][-1]
    stressor_text = f"The patient experienced the following since your last session: {last_stressor['description']}"

therapist_instructions_template = """
...
PATIENT SUMMARY:
{user_analysis}

RECENT EVENTS:
{stressor_text}
...
"""

therapist_instructions = therapist_instructions_template.format(
    # ... other fields
    stressor_text=stressor_text,
)
```

# Proposed Upgrades for SFT_generation_GPT_v.3.py

This document outlines the proposed architectural and code changes to evolve `SFT_generation_GPT_v.2.py` into a more dynamic and stateful multi-agent simulation. The key enhancement is the introduction of an explicit internal state for the Patient Agent and a deterministic State Update Node to evolve this state between sessions.

## 1. High-Level Architecture

The simulation will be restructured around four key nodes, moving from a simple back-and-forth dialogue graph to a more comprehensive state-evolution loop.

**A. Environment Node**
- **Function:** Simulates external life events by sampling stressors from the `ENVIRONMENT_STRESSORS` list.
- **Output:** Updates a `stressor_ledger` in the main dialogue state.
- **Implementation:** The existing `environment_agent_node` will be retained and enhanced.

**B. Patient Agent Node**
- **Function:** The LLM-based agent that generates dialogue. It will now also maintain and report on its internal state vector.
- **Inputs:** Current internal state (`S_t`), dialogue history.
- **Outputs:** Patient dialogue (utterance), subjective experience (summary of state).
- **Implementation:** The `patient_node` will be updated to incorporate the internal state into its prompts.

**C. Therapist Agent Node**
- **Function:** The LLM-based agent that generates therapeutic dialogue.
- **Inputs:** Patient's internal state (`S_t`), `stressor_ledger`, dialogue history, and the outcome of the patient's last decision (cope vs. relapse).
- **Outputs:** Therapist dialogue, and a classification of the therapeutic strategy used (e.g., "MI-style reflection", "Validation").
- **Implementation:** The `therapist_node` prompt will be updated to use this new rich context to make more informed decisions.

**D. State Update Node (New Node)**
- **Function:** A new, non-LLM node that deterministically updates the Patient Agent's internal state based on mathematical rules.
- **Inputs:** Previous state (`S_t-1`), stressors, patient's decision (cope/relapse), and the therapist's strategy.
- **Outputs:** The new state (`S_t`).
- **Implementation:** A new function, `state_update_node`, will be created to house these calculations.

## 2. Detailed Implementation Changes

### A. DialogueState Modification

The central `DialogueState` TypedDict will be expanded to hold the new patient state vector.

```python
class DialogueState(TypedDict):
    # ... existing fields ...
    patient_internal_state: Dict[str, float]
    last_action: Literal["cope", "urge-driven", "none"]
    therapist_strategy_classification: str
```

The `patient_internal_state` dictionary will be initialized with starting values:
```python
"patient_internal_state": {
  "craving_level": 0.5,
  "stress_level": 0.5,
  "self_efficacy": 0.5,
  "emotional_pain": 0.5,
  "cognitive_control": 0.5,
  "motivation_for_change": 0.5,
  "habit_strength": 0.5,
  "shame": 0.5
}
```

### B. New `state_update_node`

This new node will be the core of the state evolution logic. It will be a Python function that takes the current `DialogueState` and returns the updated state.

```python
import numpy as np

def state_update_node(state: DialogueState) -> Dict[str, Any]:
    """
    Updates the patient's internal state based on stressors, decisions, and therapy.
    """
    s = state["patient_internal_state"].copy()

    # 1. Process Stressors
    # ... logic for updating stress_level based on unprocessed stressors in stressor_ledger ...
    # σt = min(1, σt-1 + Σ(αi * Ii(t)))
    # σt += λ * unprocessed_stressors

    # 2. Update Craving
    # ct = clip(β1*σt + β2*ht-1 - β3*kt-1 - β4*et-1 + β5*pt-1)

    # 3. Decision Policy (Relapse Probability)
    # P(urget) = σ(wc*ct + wσ*σt + wh*ht - wk*kt - we*et - wm*mt + wq*qt)
    # action = "urge-driven" if np.random.rand() < P_urge else "cope"

    # 4. Learning and Update Rules based on action
    if action == "urge-driven":
        # ht+1 = ht + ηh*(1-ht)
        # et+1 = et - ηe
        # qt+1 = qt + ηq
    else: # cope
        # ht+1 = ht - ηh'
        # et+1 = et + ηe'
        # kt+1 = kt + ηk

    # 5. Apply Therapy Impact
    # ... logic to update state variables based on therapist_strategy_classification ...
    # e.g., if state["therapist_strategy_classification"] == "MI-style reflection":
    #   s["motivation_for_change"] += 0.05
    #   s["self_efficacy"] += 0.05

    # Clip all values to be within [0, 1]
    for key in s:
        s[key] = np.clip(s[key], 0, 1)

    return {"patient_internal_state": s, "last_action": action}
```

### C. Therapist Node Enhancement

The `therapist_node` will be modified in two ways:
1.  **Input:** The prompt will be updated to include the `patient_internal_state` and `last_action` so the therapist can react to the patient's current condition.
2.  **Output:** The therapist LLM will be asked to classify its own response into one of the predefined categories (e.g., "MI-style reflection", "Validation", "Directive advice"). This classification will be stored in `therapist_strategy_classification` and used by the `state_update_node`.

The prompt for the therapist would be modified to include:
```
PATIENT INTERNAL STATE:
- Craving: {craving_level}
- Stress: {stress_level}
- Self-Efficacy: {self_efficacy}
...etc...

LAST ACTION OUTCOME: {last_action}
```
And the therapist will be asked to output a JSON object including the reply and the strategy classification.

### D. Patient Node Enhancement

The `patient_node` prompt will be updated to use the `patient_internal_state` to inform its response. This will make the patient's dialogue more grounded in their current emotional and cognitive state.

The prompt for the patient would be modified to include:
```
Your current internal state is:
- Craving: {craving_level} (How much you want to use right now)
- Stress: {stress_level} (How overwhelmed you feel)
- Self-Efficacy: {self_efficacy} (How confident you are in your ability to cope)
...etc...

Based on this state and the conversation, generate your next reply.
```

### E. Graph Modification

The LangGraph `StateGraph` will be updated to include the new node and the modified flow. The new flow for a single "time step" (between sessions) will be:

1.  `environment_agent_node`: A stressor occurs.
2.  `state_update_node`: The state is updated based on the stressor.
3.  `patient_node`: The patient speaks, informed by their new state.
4.  `therapist_node`: The therapist responds.
5.  `state_update_node`: The state is updated again based on the therapist's intervention.

This cycle would represent the evolution of the patient's state between and during therapy sessions.

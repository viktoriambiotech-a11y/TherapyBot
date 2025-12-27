# Proposed Changes for Multi-Session Dialogue Generation in `SFT_generation_GPT_v.2.py`

This document outlines a series of proposed changes to the `SFT_generation_GPT_v.2.py` script to enhance its multi-session dialogue generation capabilities. The goal is to create a more structured and clinically relevant therapeutic simulation by introducing a six-session framework and a rule-based strategy selection mechanism.

## 1. Multi-Session Dialogue Structure (S1-S6)

To better simulate a realistic therapeutic process, we propose a six-session story arc. Each session will have a specific clinical goal and a corresponding Motivational Interviewing (MI) focus. The session number will be added to the `DialogueState` to provide the LLM with the necessary context to generate clinically appropriate and context-aware dialogue.

### Session Goals and MI Focus

| Session | CBT Stage Goal | MI Focus / Key Action |
| :--- | :--- | :--- |
| **S1** | Build trust, assessment, set safety/limits. | **Engage:** Use 0-10 importance/confidence rulers; start trigger/urge log. |
| **S2** | Identifying negative cognitions. | **Focus:** Review log; use values/agenda mapping to map thought → feeling → body → action. |
| **S3** | Challenging false beliefs. | **Evoke:** Elicit change talk (DARN) using double-sided reflections; challenge the core thought (e.g., "one time won't hurt"). |
| **S4** | Restructuring cognitive patterns. | **Bridge Evoke → Plan:** Formulate an If-Then plan; move change talk to CAT (Commitment, Activation, Taking steps). |
| **S5** | Behavioral skill building. | **Plan:** Redesign a high-risk window; create a crisis micro-plan and plan for barriers. |
| **S6** | Consolidation & termination. | **Maintenance:** Review gains; extend the plan for relapse prevention and peer support. |

### `DialogueState` Modification

The `DialogueState` TypedDict will be updated to include the current session number:

```python
class DialogueState(TypedDict):
    # ... existing fields
    session_number: int
```

This `session_number` will be used by the new Policy Node to select the appropriate strategies for the current stage of therapy.

## 2. Therapist Node Enhancement

To improve the clinical relevance of the therapist's responses, we propose a two-step process for the `therapist_node`:

**Step A: Strategy Selection (Rule-Based Policy Node)**

Before the LLM is called to generate a response, a new rule-based Python function, the "Policy Node," will be executed. This function will analyze the current state of the dialogue and select the most appropriate therapeutic strategies.

The Policy Node will take the following as input:

-   `session_number`: The current session number (1-6).
-   `patient_state_summary`: A summary of the patient's current state.
-   `stressor_ledger`: A log of recent stressors.
-   `history`: The dialogue history.

Based on this information, the Policy Node will return a list of recommended strategy IDs (e.g., `["mi_scales", "cbt_trigger_mapping"]`).

**Step B: Dialogue Generation (LLM Node)**

The `therapist_node` will then use the list of recommended strategies to construct a more focused prompt for the LLM. The LLM will be instructed to use the pre-selected strategies to generate the therapist's response, ensuring that the dialogue is aligned with the session's clinical goals.

The updated `therapist_node` will look something like this:

```python
def therapist_node(state: DialogueState) -> Dict[str, Any]:
    # Step A: Select strategies using the Policy Node
    recommended_strategies = policy_node(state)

    # Step B: Generate dialogue using the LLM
    # ... existing code ...

    therapist_instructions_template = """
    ...
    RECOMMENDED STRATEGIES:
    {recommended_strategies}
    ...
    """

    # ... existing code ...
```

By separating strategy selection from dialogue generation, we can create a more robust and clinically sound simulation. The rule-based Policy Node will ensure that the therapist's actions are always aligned with the session's goals, while the LLM will provide the natural language generation capabilities to create a realistic and engaging dialogue.

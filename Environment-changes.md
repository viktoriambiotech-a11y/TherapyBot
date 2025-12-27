# Proposed Modifications for Environment Agent Integration

This document outlines the proposed changes to `SFT_generation_GPT_v.1.py` to introduce an **Environment Agent**. The primary function of this agent is to inject challenges and state changes for the Patient Agent between therapy sessions, creating a more dynamic and realistic simulation.

## 1. High-Level Summary of Changes

The core of the proposed changes is to introduce a multi-session simulation loop. Between each session, the Environment Agent will introduce a stressor that affects the patient's state. The therapist will be aware of these stressors in the subsequent session.

## 2. PATIENT_SYSTEM_PROMPT Update

The `PATIENT_SYSTEM_PROMPT` will be updated to require the Patient Agent to report stressors at the beginning of each new session.

```
- Briefly report any significant stressors, cravings, emotional events, or challenges
  that occurred since the last session.
- Describe these events in your own words, as a patient would naturally do.
- You do NOT mention the word 'stressor' or refer to any internal ledgers or systems.

If multiple events occurred, prioritize those that were emotionally intense,
triggered cravings, or affected your recovery progress.

Then continue the session normally, responding authentically to the therapist.
```

## 3. Multi-Session Simulation Loop

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

## 4. Environment Agent Node

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

## 5. Data Structures

### a. Environment Stressors Catalog

A new catalog of stressors will be defined.

```json
[
  {
    "Category": "Social/Environmental",
    "Stressor": "Peer pressure",
    "Description": "Friends invited me to a bar after work",
    "Severity": "High",
    "Likely Duration": "Hours–Days"
  },
  {
    "Category": "Social/Environmental",
    "Stressor": "Exposure to drinking cues",
    "Description": "Walked past my usual liquor store after a stressful day",
    "Severity": "Medium–High",
    "Likely Duration": "Minutes–Days"
  },
  {
    "Category": "Social/Environmental",
    "Stressor": "Drinking-centered events",
    "Description": "Attended a wedding where alcohol was central",
    "Severity": "Medium–High",
    "Likely Duration": "1–3 days"
  },
  {
    "Category": "Social/Environmental",
    "Stressor": "Lack of social support",
    "Description": "No one to call when cravings hit",
    "Severity": "High",
    "Likely Duration": "Weeks–Months"
  },
  {
    "Category": "Social/Environmental",
    "Stressor": "Social isolation",
    "Description": "Spent the weekend alone with no plans",
    "Severity": "Medium–High",
    "Likely Duration": "Days–Weeks"
  },
  {
    "Category": "Social/Environmental",
    "Stressor": "Stigma/shame",
    "Description": "Avoiding help due to fear of judgment",
    "Severity": "Medium",
    "Likely Duration": "Weeks–Months"
  },
  {
    "Category": "Interpersonal",
    "Stressor": "Relationship conflict",
    "Description": "Unresolved argument with partner",
    "Severity": "High",
    "Likely Duration": "Days–Weeks"
  },
  {
    "Category": "Interpersonal",
    "Stressor": "Breakup/rejection",
    "Description": "Partner ended relationship unexpectedly",
    "Severity": "High",
    "Likely Duration": "Weeks–Months"
  },
  {
    "Category": "Interpersonal",
    "Stressor": "Family conflict",
    "Description": "Ongoing tension with family members",
    "Severity": "Medium–High",
    "Likely Duration": "Weeks–Months"
  },
  {
    "Category": "Interpersonal",
    "Stressor": "Invalidation/gaslighting",
    "Description": "Feelings dismissed as overreacting",
    "Severity": "Medium",
    "Likely Duration": "Days–Weeks"
  },
  {
    "Category": "Work/Academic",
    "Stressor": "Deadlines/pressure",
    "Description": "Multiple deadlines in same week",
    "Severity": "Medium–High",
    "Likely Duration": "Days–Weeks"
  },
  {
    "Category": "Work/Academic",
    "Stressor": "Job insecurity",
    "Description": "Fear of layoffs or reduced hours",
    "Severity": "High",
    "Likely Duration": "Weeks–Months"
  },
  {
    "Category": "Work/Academic",
    "Stressor": "Shift changes",
    "Description": "Moved from day shift to nights",
    "Severity": "Medium",
    "Likely Duration": "Weeks"
  },
  {
    "Category": "Work/Academic",
    "Stressor": "Burnout/overwork",
    "Description": "Sustained overtime with little recovery",
    "Severity": "High",
    "Likely Duration": "Weeks–Months"
  },
  {
    "Category": "Work/Academic",
    "Stressor": "Academic failure",
    "Description": "Failed an important exam",
    "Severity": "Medium–High",
    "Likely Duration": "Days–Weeks"
  },
  {
    "Category": "Emotional/Cognitive",
    "Stressor": "Negative emotions",
    "Description": "Persistent anxiety, anger, or sadness",
    "Severity": "High",
    "Likely Duration": "Hours–Weeks"
  },
  {
    "Category": "Emotional/Cognitive",
    "Stressor": "Boredom/anhedonia",
    "Description": "Nothing feels engaging or rewarding",
    "Severity": "Medium",
    "Likely Duration": "Days–Weeks"
  },
  {
    "Category": "Emotional/Cognitive",
    "Stressor": "Positive mood/overconfidence",
    "Description": "Feeling cured; belief one drink is safe",
    "Severity": "Medium–High",
    "Likely Duration": "Hours–Days"
  },
  {
    "Category": "Emotional/Cognitive",
    "Stressor": "Low self-efficacy",
    "Description": "Doubting ability to stay sober",
    "Severity": "High",
    "Likely Duration": "Weeks–Months"
  },
  {
    "Category": "Emotional/Cognitive",
    "Stressor": "Cognitive distortions",
    "Description": "I've already slipped so it doesn't matter",
    "Severity": "High",
    "Likely Duration": "Hours–Days"
  },
  {
    "Category": "Emotional/Cognitive",
    "Stressor": "Rumination",
    "Description": "Can't stop thinking about drinking",
    "Severity": "Medium–High",
    "Likely Duration": "Days"
  },
  {
    "Category": "Physical/Biological",
    "Stressor": "Pain or injury",
    "Description": "Chronic or acute pain disrupting sleep",
    "Severity": "Medium–High",
    "Likely Duration": "Days–Weeks"
  },
  {
    "Category": "Physical/Biological",
    "Stressor": "Illness/medical stress",
    "Description": "Recovering from surgery or illness",
    "Severity": "Medium",
    "Likely Duration": "Weeks"
  },
  {
    "Category": "Physical/Biological",
    "Stressor": "Sleep deprivation",
    "Description": "Several nights of poor sleep due to racing thoughts or insomnia",
    "Severity": "Medium",
    "Likely Duration": "Days–Weeks"
  },
  {
    "Category": "Physical/Biological",
    "Stressor": "PAWS/withdrawal",
    "Description": "Ongoing irritability and anxiety post-quit",
    "Severity": "High",
    "Likely Duration": "Weeks–Months"
  },
  {
    "Category": "Life Events",
    "Stressor": "Major transitions",
    "Description": "Moving to a new city",
    "Severity": "Medium–High",
    "Likely Duration": "Weeks–Months"
  },
  {
    "Category": "Life Events",
    "Stressor": "Bereavement",
    "Description": "Death of a loved one",
    "Severity": "High",
    "Likely Duration": "Months"
  },
  {
    "Category": "Life Events",
    "Stressor": "Housing instability",
    "Description": "Uncertain living situation",
    "Severity": "High",
    "Likely Duration": "Weeks–Months"
  },
  {
    "Category": "Life Events",
    "Stressor": "Financial crisis",
    "Description": "Unexpected bills or debt",
    "Severity": "High",
    "Likely Duration": "Weeks–Months"
  },
  {
    "Category": "Treatment/Recovery",
    "Stressor": "Treatment disengagement",
    "Description": "Stopped therapy or meetings",
    "Severity": "High",
    "Likely Duration": "Weeks"
  },
  {
    "Category": "Treatment/Recovery",
    "Stressor": "Negative treatment experience",
    "Description": "Felt judged by clinician",
    "Severity": "Medium–High",
    "Likely Duration": "Weeks–Months"
  },
  {
    "Category": "Treatment/Recovery",
    "Stressor": "Early recovery overconfidence",
    "Description": "Belief recovery is complete",
    "Severity": "Medium",
    "Likely Duration": "Days–Weeks"
  },
  {
    "Category": "Emotional/Cognitive",
    "Stressor": "Executive functioning overload",
    "Description": "Too many decisions and responsibilities accumulated in one day",
    "Severity": "Medium",
    "Likely Duration": "Hours–Days"
  },
  {
    "Category": "Emotional/Cognitive",
    "Stressor": "Cognitive impairment / mental fog",
    "Description": "Difficulty concentrating and remembering coping strategies",
    "Severity": "Medium–High",
    "Likely Duration": "Days–Weeks"
  },
  {
    "Category": "Emotional/Cognitive",
    "Stressor": "Trauma reminders",
    "Description": "Encountered a reminder of a past traumatic event",
    "Severity": "High",
    "Likely Duration": "Hours–Days"
  },
  {
    "Category": "Emotional/Cognitive",
    "Stressor": "Trauma-related hyperarousal",
    "Description": "Sudden surge of fear or emotional flooding without clear cause",
    "Severity": "High",
    "Likely Duration": "Hours–Days"
  },
  {
    "Category": "Physical/Biological",
    "Stressor": "Circadian rhythm disruption",
    "Description": "Shift work or jet lag disrupted normal sleep–wake schedule",
    "Severity": "Medium–High",
    "Likely Duration": "Days–Weeks"
  },
  {
    "Category": "Social/Environmental",
    "Stressor": "Living with active substance users",
    "Description": "Household member continues drinking or using substances",
    "Severity": "High",
    "Likely Duration": "Weeks–Months"
  },
  {
    "Category": "Treatment/Recovery",
    "Stressor": "Early abstinence vulnerability",
    "Description": "Within the first weeks after detox or quitting alcohol",
    "Severity": "High",
    "Likely Duration": "Days–Weeks"
  }
]
```

### b. Stressor Ledger

The `Stressor_Ledger` will be a list of stressor events, stored in the `DialogueState`.

## 6. Modifications to `DialogueState`

The `DialogueState` will be extended to include the `stressor_ledger`.

```python
class DialogueState(TypedDict):
    # ... existing fields
    stressor_ledger: List[Dict[str, Any]]
```

## 7. `langgraph` Integration

The `environment_agent_node` will not be part of the main conversational graph, but will be called *between* graph executions in the multi-session loop.

### Proposed `therapist_node` modification:

The therapist should not be made aware of the `stressor_ledger` directly. Instead, the therapist's prompt will be updated to expect the patient to mention stressors and to actively explore them using appropriate therapeutic techniques. This encourages a more naturalistic therapeutic interaction.
```

The therapist's instructions should be updated to include guidance such as:

- "At the beginning of the session, the patient may share recent stressors. Actively listen and explore these challenges using techniques like open-ended questions, reflections, and affirmations."
- "Do not assume prior knowledge of the patient's stressors. Your awareness should come directly from the patient's self-reporting during the session."

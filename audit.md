# Audit and Implementation Plan

This document outlines the changes required to enhance the dialogue system to better simulate a therapeutic conversation. It also includes suggestions for code improvements.

## I. Required Changes

### 1. Dialogue Termination Condition

- **Current:** The dialogue terminates after a fixed number of turns (`max_turns`).
- **[x] Required:** The dialogue should terminate early if the patient indicates resolution (e.g., expresses sufficient motivation and confidence).
- **Implementation:**
    - In the `patient_node`, after generating the patient's reply, analyze the text for keywords or sentiment indicating resolution.
    - Add a new state to `DialogueState` to track patient's resolution status.
    - In the `route_after_patient` function, check the resolution status. If the patient has indicated resolution, transition to `END`.

### 2. Dialogue Agenda

- **Current:** The dialogue is driven by a random selection of strategies.
- **[x] Required:** The dialogue should follow a specific agenda:
    1. Rapport & Goal Alignment
    2. Episode Clarification
    3. Plan Formulation
    4. Next-step Micro-commitment
- **Implementation:**
    - Introduce a new state in `DialogueState` to track the current agenda phase.
    - In the `therapist_node`, the `pick_next_strategy` function should be modified to select strategies relevant to the current agenda phase.
    - Create a mapping between agenda phases and relevant strategies.

### 3. Patient State Summary

- **Current:** The patient's state is implicit in the dialogue history.
- **[x] Required:** The Patient Agent should provide a compact state summary each turn (craving, trigger salience, confidence, recent lapse flags).
- **Implementation:**
    - Add a new field to the `DialogueState` to store the patient's state summary.
    - In the `patient_node`, after generating the patient's utterance, also generate a state summary. This could be a structured dictionary or a simple string.
    - The `therapist_node` will then have access to this state summary to inform its response.

### 4. Therapist Micro-commitment

- **Current:** The therapist provides a conversational reply.
- **[x] Required:** The Therapy Agent should generate a measurable micro-commitment (deadline and success criterion) in addition to the utterance.
- **Implementation:**
    - Add a new field to the `DialogueState` to store the micro-commitment.
    - In the `therapist_node`, after generating the therapist's reply, generate a micro-commitment. This could be a separate LLM call or part of the same call with structured output.

### 5. Update `therapist_instructions`

- **Current:** The `therapist_instructions` provides general guidance.
- **[x] Required:** Add the following to `therapist_instructions`:
    - Use Open-ended questions, Affirmations, Reflective listening, Summarizing.
- **Implementation:**
    - Directly modify the `therapist_instructions` string in the script.

### 6. Update `strategy_text`

- **Current:** The `strategy_text` is based on a randomly selected strategy.
- **[x] Required:** The `strategy_text` should instruct the therapist to pick an appropriate strategy or action tool based on the log and current stressors.
- **Implementation:**
    - This is related to the Dialogue Agenda change. The `pick_next_strategy` function will be updated to be more intelligent, and the `strategy_text` will be updated to reflect this.

## II. Code Improvement Suggestions

### 1. Modularity

- **Suggestion:** The script is currently a single file. It could be broken down into smaller, more manageable modules.
- **Benefit:** Improved readability, maintainability, and testability.
- **Example:**
    - `prompts.py`: Store all the prompt strings.
    - `strategies.py`: Store the MI, CBT, and ACTIONABLE_TOOLS lists.
    - `graph.py`: Define the LangGraph state and nodes.
    - `main.py`: The main script to run the simulation.

### 2. Unit Tests

- **Suggestion:** Add unit tests for the functions, especially the nodes and routing logic.
- **Benefit:** Ensures that changes to one part of the code do not break other parts.
- **Example:**
    - Test that `patient_node` returns a valid patient reply.
    - Test that `therapist_node` returns a valid therapist reply.
    - Test that `route_after_patient` and `route_after_therapist` return the correct next node.

### 3. Strategy Selection

- **Suggestion:** The `pick_next_strategy` function currently selects a strategy randomly. A more sophisticated method could be used.
- **Benefit:** More coherent and effective dialogues.
- **Example:**
    - Use an LLM to select the next best strategy based on the conversation history and patient state.
    - Use a rule-based system that maps patient states to strategies.

### 4. Configuration Management

- **Suggestion:** Use a configuration file (e.g., YAML or JSON) to manage settings like model names, max turns, and output directory.
- **Benefit:** Easier to change settings without modifying the code.

### 5. Error Handling

- **Suggestion:** The error handling in `call_llm` is good, but it could be more robust.
- **Benefit:** The application will be more resilient to API failures.
- **Example:**
    - Implement retries with exponential backoff for API calls.
    - Add more specific error handling for different types of API errors.

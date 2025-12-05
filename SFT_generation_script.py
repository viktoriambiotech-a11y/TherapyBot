# -*- coding: utf-8 -*-
"""
Multiturn Therapeutic Dialogues Generation

Recreates a single-session, multi-turn role-play setup:
 - Patient Agent: Structured profile and difficulty level.
 - Therapist Agent: MI/CBT-consistent strategies.
"""

import json
import os
import random
from datetime import datetime
from typing import TypedDict, List, Literal, Dict, Any

import google.generativeai as genai
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END


# Load environment variables
load_dotenv()

# Initialize Google Generative AI Client
# Ensure GOOGLE_API_KEY is set in your .env file.
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Model Configuration
MODEL_PATIENT = "gemini-2.5-flash"
MODEL_THERAPIST = "gemini-2.5-flash"
# Updated to valid model name format if needed
# MODEL_THERAPIST = "gpt-5-mini"


# Therapeutic Strategies Catalogs

MI_STRATEGIES = [
    {
        "id": "mi_scales",
        "name": "importance / confidence / readiness scale (0–10)",
        "description": (
            "Use 0–10 scaling questions about importance, confidence, or "
            "readiness to change. Ask the patient to pick numbers, explore "
            "why they chose them, and gently elicit 'change talk' about what "
            "could move the number up."
        ),
    },
    {
        "id": "mi_ep_e",
        "name": "Elicit–Provide–Elicit",
        "description": (
            "First elicit the patient’s understanding or perspective, then "
            "briefly provide information or feedback, then elicit their "
            "reaction and how it fits with their goals."
        ),
    },
    {
        "id": "mi_decisional_balance",
        "name": "decisional balance",
        "description": (
            "Explore pros and cons of continuing current substance use vs. "
            "making a change. Reflect ambivalence and highlight "
            "patient-stated reasons for change."
        ),
    },
    {
        "id": "mi_values",
        "name": "values clarification",
        "description": (
            "Ask about what kind of person they want to be, their core "
            "values (e.g., family, health), and connect those values to "
            "possible changes in substance use."
        ),
    },
    {
        "id": "mi_agenda",
        "name": "agenda mapping",
        "description": (
            "Collaboratively identify and prioritize 2–3 topics for today’s "
            "conversation, checking what feels most important for them to "
            "focus on."
        ),
    },
    {
        "id": "mi_oars",
        "name": "OARS micro-skills",
        "description": (
            "Lean heavily on open questions, affirmations, reflective "
            "listening, and summaries. Emphasize empathy and curiosity "
            "rather than giving direct advice."
        ),
    },
    {
        "id": "mi_change_talk",
        "name": "DARN–CAT change talk",
        "description": (
            "Elicit and reflect language about Desire, Ability, Reasons, "
            "Need, Commitment, Activation, and Taking steps for change, "
            "especially around substance use and recovery."
        ),
    },
]

CBT_STRATEGIES = [
    {
        "id": "cbt_functional_analysis",
        "name": "functional analysis of a recent episode",
        "description": (
            "Ask about a recent use or near-use episode and unpack "
            "antecedents, thoughts, feelings, behaviors, and consequences. "
            "Look for patterns that maintain substance use."
        ),
    },
    {
        "id": "cbt_trigger_mapping",
        "name": "trigger mapping",
        "description": (
            "Identify people, places, times, and internal states that tend "
            "to trigger cravings. Make the chain from trigger -> thought -> "
            "feeling -> urge -> behavior explicit."
        ),
    },
    {
        "id": "cbt_reappraisal",
        "name": "cognitive reappraisal of urges",
        "description": (
            "Gently question unhelpful automatic thoughts (e.g., 'I can’t "
            "get through the evening without using') and help the patient "
            "generate more balanced, realistic alternatives."
        ),
    },
    {
        "id": "cbt_coping_skills",
        "name": "coping-skill rehearsal for cravings",
        "description": (
            "Rehearse specific coping skills for cravings such as delay, "
            "distraction, paced breathing, grounding, or urge-surfing. "
            "Practice how they might use them in an upcoming high-risk window."
        ),
    },
    {
        "id": "cbt_refusal",
        "name": "refusal-skills scripting",
        "description": (
            "Help the patient script and rehearse a brief refusal line for "
            "offers to use (e.g., from peers) and problem-solve how they "
            "might deliver it."
        ),
    },
    {
        "id": "cbt_stimulus_control",
        "name": "stimulus control",
        "description": (
            "Brainstorm concrete ways to reduce exposure to cues (e.g., "
            "removing contacts, avoiding certain apps or locations) and plan "
            "at least one small action."
        ),
    },
    {
        "id": "cbt_exposure",
        "name": "graded exposure / urge-surfing",
        "description": (
            "Work with the patient to imagine or plan gradual exposure to a "
            "high-risk cue while practising urge-surfing and safety "
            "behaviors, in a way that feels tolerable."
        ),
    },
    {
        "id": "cbt_problem_solving",
        "name": "problem-solving for barriers",
        "description": (
            "Identify a practical barrier (e.g., transportation, schedule, "
            "money) and walk through structured problem solving: define the "
            "problem, brainstorm options, pick one step, and plan when/how "
            "to try it."
        ),
    },
]

ACTIONABLE_TOOLS = [
    {
        "id": "act_hobbies",
        "name": "explore specific hobbies or interests",
        "description": (
            "Explore specific hobbies or interests the patient can engage "
            "in to replace addictive behaviors (e.g., art, sports, "
            "volunteering)."
        ),
    },
    {
        "id": "act_routine",
        "name": "structured daily routine",
        "description": (
            "Develop a structured daily routine to bring stability and "
            "reduce idle time that might trigger relapse."
        ),
    },
    {
        "id": "act_grounding",
        "name": "grounding techniques",
        "description": (
            "Introduce grounding techniques such as sensory exercises or "
            "physical activities to manage anxiety or cravings."
        ),
    },
    {
        "id": "act_support_group",
        "name": "join support group or community",
        "description": (
            "Suggest joining a support group or community to build social "
            "connections with individuals on similar journeys."
        ),
    },
    {
        "id": "act_psychoeducation",
        "name": "psychoeducation on brain/emotions",
        "description": (
            "Provide psychoeducation on how addiction affects the brain "
            "and emotional regulation."
        ),
    },
    {
        "id": "act_emotional_triggers",
        "name": "identify emotional triggers",
        "description": (
            "Work on identifying and addressing specific emotional triggers "
            "through reflective exercises."
        ),
    },
    {
        "id": "act_assertive_comm",
        "name": "assertive communication techniques",
        "description": (
            "Practice assertive communication techniques for setting "
            "boundaries with peers or environments that encourage "
            "substance use."
        ),
    },
    {
        "id": "act_journaling",
        "name": "journal thoughts and emotions",
        "description": (
            "Encourage the patient to journal their thoughts and emotions "
            "as a way to process experiences and identify patterns related "
            "to cravings or triggers."
        ),
    },
    {
        "id": "act_relaxation",
        "name": "relaxation techniques",
        "description": (
            "Introduce relaxation techniques such as progressive muscle "
            "relaxation or guided imagery to alleviate stress and improve "
            "emotional well-being."
        ),
    },
    {
        "id": "act_goals",
        "name": "short-term and long-term goals",
        "description": (
            "Help the patient set short-term and long-term goals to "
            "maintain focus and motivation during their recovery journey."
        ),
    },
    {
        "id": "act_mindfulness",
        "name": "mindfulness-based activities",
        "description": (
            "Explore mindfulness-based activities like meditation, yoga, or "
            "tai chi to promote self-awareness and emotional regulation."
        ),
    },
    {
        "id": "act_strengths",
        "name": "reinforce personal strengths",
        "description": (
            "Identify and reinforce the patient’s personal strengths and "
            "past successes to build confidence in their ability to "
            "overcome challenges."
        ),
    },
    {
        "id": "act_health",
        "name": "nutrition, sleep, and exercise education",
        "description": (
            "Provide education on the importance of nutrition, sleep, and "
            "exercise in supporting recovery and overall health."
        ),
    },
    {
        "id": "act_crisis_plan",
        "name": "develop a crisis plan",
        "description": (
            "Develop a crisis plan for managing high-risk situations or "
            "moments of intense cravings, including a list of emergency "
            "contacts and actions."
        ),
    },
    {
        "id": "act_vision_board",
        "name": "vision board of positive outcomes",
        "description": (
            "Encourage the patient to create a vision board or list of "
            "positive outcomes they hope to achieve through recovery as a "
            "source of inspiration."
        ),
    },
    {
        "id": "act_gratitude",
        "name": "gratitude journaling",
        "description": (
            "Discuss the concept of gratitude and suggest keeping a "
            "gratitude journal to focus on positive aspects of life and "
            "maintain perspective."
        ),
    },
    {
        "id": "act_complementary_therapy",
        "name": "complementary therapies resources",
        "description": (
            "Offer resources or referrals for complementary therapies, such "
            "as art therapy, music therapy, or animal-assisted therapy, to "
            "enhance emotional support."
        ),
    },
    {
        "id": "act_community",
        "name": "contribute to community",
        "description": (
            "Support the patient in finding meaningful ways to contribute "
            "to their community, such as mentoring, advocacy, or local "
            "initiatives, to foster a sense of purpose."
        ),
    },
]

# Combine all lists so the therapist node can select from any of them
ALL_STRATEGIES = MI_STRATEGIES + CBT_STRATEGIES + ACTIONABLE_TOOLS

# LangGraph State Definition

class DialogueState(TypedDict):
    """
    Represents the memory and context of the conversation.

    Attributes:
        history: List of interaction dictionaries (role/content).
        patient_profile: String representation of the patient.
        difficulty: The set difficulty level (easy/medium/hard).
        difficulty_description: Instructions on resistance level.
        max_turns: Target total turns.
        turn_index: Current 0-based turn count.
        strategy_history: List of strategy IDs used so far.
        patient_resolution_status: Boolean indicating if the patient has achieved resolution.
        dialogue_agenda_phase: The current phase of the dialogue agenda.
        patient_state_summary: A summary of the patient's state.
        therapist_micro_commitment: A micro-commitment from the therapist.
    """
    history: List[Dict[str, str]]
    patient_profile: str
    difficulty: Literal["easy", "medium", "hard"]
    difficulty_description: str
    max_turns: int
    turn_index: int
    strategy_history: List[str]
    patient_resolution_status: bool
    dialogue_agenda_phase: str
    patient_state_summary: str
    therapist_micro_commitment: str


DIFFICULTY_DESCRIPTIONS = {
    "easy": (
        "You are generally willing to accept guidance, open to treatment, "
        "and cooperative. You still experience cravings and doubts, but "
        "you tend to respond positively to suggestions."
    ),
    "medium": (
        "You are ambivalent and show some resistance: part of you wants to "
        "change, but another part wants to keep using. You may agree with "
        "some ideas and push back on others."
    ),
    "hard": (
        "You have long-standing substance use and substantial mistrust or "
        "skepticism about treatment. You often challenge or deflect "
        "suggestions, emphasize barriers, and may minimize the need for "
        "change."
    ),
}


def call_llm(
    model: str,
    instructions: str,
    input_text: str,
    max_output_tokens: int = 256
) -> str:
    """
    Thin wrapper around the Google Generative AI API with error handling.
    """
    try:
        model = genai.GenerativeModel(model)
        response = model.generate_content(
            f"{instructions}\n\n{input_text}",
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=max_output_tokens
            )
        )
        return response.text.strip()
    except Exception as e:
        # Print the error and return a placeholder message
        print(f"\n--- ERROR DURING API CALL ---")
        print(f"Failed to generate response using model {model}.")
        print(f"Error details: {e}\n")
        
        # Returning a placeholder ensures the graph doesn't crash 
        # but marks the failure clearly in the history.
        return f"[API_FAILURE: {type(e).__name__}]"

def render_history_for_prompt(history: List[Dict[str, str]]) -> str:
    """
    Turn internal history into a plain-text transcript for prompting.
    """
    lines = []
    for msg in history:
        role = "Patient" if msg["role"] == "patient" else "Therapist"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


# Patient Node Logic

def patient_node(state: DialogueState) -> Dict[str, Any]:
    """
    Generates the patient's next utterance based on profile and history.
    """
    history_text = render_history_for_prompt(state["history"])
    
    # Handle empty history case for prompt display
    display_history = (
        history_text if history_text
        else "(no prior conversation – this is the first turn)"
    )

    patient_instructions = (
        "You are role-playing as a patient in addiction recovery.\n"
        "Speak from the profile below, including your personality traits, "
        "substance use history, significant life events, behavioral themes, "
        "and motivations for using.\n"
        "Stay consistent with this profile and with the conversation so far.\n"
        "Your difficulty level description explains how resistant or "
        "ambivalent you are.\n"
        "Only return what the patient says next. Do not include narration "
        "or system messages."
    )

    patient_prompt = (
        "Here is your patient profile:\n"
        f"{state['patient_profile']}\n\n"
        "Conversation so far:\n"
        f"{display_history}\n\n"
        f"Difficulty setting: {state['difficulty_description']}\n\n"
        "Now, continue the conversation with the therapist to explore ways "
        "to reduce or stop substance use. If you feel your goals have "
        "clearly been achieved, you may say goodbye. Otherwise, respond "
        "naturally and briefly in your own voice."
    )

    patient_reply = call_llm(
        model=MODEL_PATIENT,
        instructions=patient_instructions,
        input_text=patient_prompt,
        max_output_tokens=128,
    )

    # Generate patient state summary
    patient_summary_instructions = (
        "Analyze the patient's message and summarize their state. "
        "Provide a compact summary covering craving, trigger salience, "
        "confidence, and recent lapse flags."
    )
    patient_summary_prompt = (
        f"Patient's message: '{patient_reply}'\n\n"
        "Generate a structured summary of the patient's current state."
    )
    patient_state_summary = call_llm(
        model=MODEL_PATIENT,
        instructions=patient_summary_instructions,
        input_text=patient_summary_prompt,
        max_output_tokens=64,
    )

    # Analyze for resolution
    resolution_instructions = (
        "Analyze the patient's message for resolution. If the patient "
        "expresses sufficient motivation and confidence to end the "
        "dialogue, return 'true'. Otherwise, return 'false'."
    )
    resolution_prompt = (
        f"Patient's message: '{patient_reply}'\n\n"
        "Has the patient indicated resolution?"
    )
    resolution_status = call_llm(
        model=MODEL_PATIENT,
        instructions=resolution_instructions,
        input_text=resolution_prompt,
        max_output_tokens=8,
    )
    patient_resolution_status = "true" in resolution_status.lower()

    new_history = state["history"] + [
        {"role": "patient", "content": patient_reply}
    ]
    new_turn_index = state["turn_index"] + 1

    return {
        "history": new_history,
        "turn_index": new_turn_index,
        "patient_state_summary": patient_state_summary,
        "patient_resolution_status": patient_resolution_status,
    }

# Therapist Node Logic

def pick_next_strategy(
    strategy_history: List[str], dialogue_agenda_phase: str
) -> Dict[str, str]:
    """
    Pick a strategy based on the current dialogue agenda phase.
    """
    dialogue_agenda_mapping = {
        "Rapport & Goal Alignment": [
            "mi_agenda", "mi_oars", "mi_values"
        ],
        "Episode Clarification": [
            "cbt_functional_analysis", "cbt_trigger_mapping"
        ],
        "Plan Formulation": [
            "cbt_coping_skills", "cbt_problem_solving"
        ],
        "Next-step Micro-commitment": [
            "act_goals", "act_crisis_plan"
        ],
    }

    agenda_strategies = dialogue_agenda_mapping.get(
        dialogue_agenda_phase, ALL_STRATEGIES
    )

    used_recent = set(strategy_history[-2:])

    candidates = [
        s for s in ALL_STRATEGIES
        if s["id"] in agenda_strategies and s["id"] not in used_recent
    ] or [s for s in ALL_STRATEGIES if s["id"] in agenda_strategies]

    return random.choice(candidates)


def therapist_node(state: DialogueState) -> Dict[str, Any]:
    """
    Generates the therapist's response using MI/CBT principles.
    """
    history_text = render_history_for_prompt(state["history"])

    # Choose a strategy for this turn
    strategy = pick_next_strategy(
        state["strategy_history"], state["dialogue_agenda_phase"]
    )
    
    strategy_text = (
        "Therapeutic strategy to emphasize this turn:\n"
        f"- Current Agenda Phase: {state['dialogue_agenda_phase']}\n"
        f"- Name: {strategy['name']}\n"
        f"- Description: {strategy['description']}\n\n"
        "You should apply this strategy in a subtle, natural way, without "
        "labeling it explicitly. Your reply must be consistent with "
        "motivational interviewing spirit (collaboration, empathy, respect "
        "for autonomy) and with CBT principles when relevant."
    )

    therapist_instructions = (
        "You are a supportive, thoughtful therapist helping a patient with "
        "substance use and recovery.\n"
        "Use motivational interviewing (MI) and cognitive behavioral "
        "therapy (CBT) principles.\n"
        "Use Open-ended questions, Affirmations, Reflective listening, "
        "and Summarizing.\n"
        "Prioritize empathy, reflective listening, and collaborative "
        "problem-solving over giving orders.\n"
        "Do not provide medical diagnoses or medication instructions.\n"
        "Respond as a real therapist might: concise, warm, and practical."
    )

    therapist_prompt = (
        "Patient profile (for context; don't repeat verbatim):\n"
        f"{state['patient_profile']}\n\n"
        "Conversation so far:\n"
        f"{history_text}\n\n"
        f"{strategy_text}\n"
        "Now write the therapist's next reply only. "
        "Do not include 'Therapist:' labels or any narration."
    )

    therapist_reply = call_llm(
        model=MODEL_THERAPIST,
        instructions=therapist_instructions,
        input_text=therapist_prompt,
        max_output_tokens=256,
    )

    new_history = state["history"] + [
        {"role": "therapist", "content": therapist_reply}
    ]
    new_turn_index = state["turn_index"] + 1
    new_strategy_history = state["strategy_history"] + [strategy["id"]]

    # Generate micro-commitment
    commitment_instructions = (
        "Based on the therapist's reply, generate a measurable "
        "micro-commitment with a deadline and success criterion."
    )
    commitment_prompt = (
        f"Therapist's reply: '{therapist_reply}'\n\n"
        "Generate a micro-commitment for the patient."
    )
    therapist_micro_commitment = call_llm(
        model=MODEL_THERAPIST,
        instructions=commitment_instructions,
        input_text=commitment_prompt,
        max_output_tokens=64,
    )

    # Advance dialogue agenda
    agenda_phases = [
        "Rapport & Goal Alignment", "Episode Clarification",
        "Plan Formulation", "Next-step Micro-commitment"
    ]
    current_phase_index = agenda_phases.index(state["dialogue_agenda_phase"])
    if current_phase_index < len(agenda_phases) - 1:
        new_dialogue_agenda_phase = agenda_phases[current_phase_index + 1]
    else:
        new_dialogue_agenda_phase = state["dialogue_agenda_phase"]

    return {
        "history": new_history,
        "turn_index": new_turn_index,
        "strategy_history": new_strategy_history,
        "therapist_micro_commitment": therapist_micro_commitment,
        "dialogue_agenda_phase": new_dialogue_agenda_phase,
    }

# Graph Routing and Construction

def route_after_patient(state: DialogueState) -> str:
    """Determine next node after patient speaks."""
    if state["patient_resolution_status"]:
        return END
    if state["turn_index"] >= state["max_turns"]:
        return END
    return "therapist"


def route_after_therapist(state: DialogueState) -> str:
    """Determine next node after therapist speaks."""
    if state["turn_index"] >= state["max_turns"]:
        return END
    return "patient"


# Build Graph
graph = StateGraph(DialogueState)

graph.add_node("patient", patient_node)
graph.add_node("therapist", therapist_node)

graph.set_entry_point("patient")

graph.add_conditional_edges(
    "patient",
    route_after_patient,
    {
        "therapist": "therapist",
        END: END,
    },
)

graph.add_conditional_edges(
    "therapist",
    route_after_therapist,
    {
        "patient": "patient",
        END: END,
    },
)

app = graph.compile()

# Execution and Output
# Example Conversation Generation
## replace 'example_patient_profile' with synthesized profiles

# Example Patient Profile
example_patient_profile = (
    "Personality Traits: Ambivalent, emotionally reactive, sometimes "
    "avoidant but cares about family.\n"
    "Substance Use History: Daily cannabis use for the past 6 years, "
    "multiple short attempts to cut back, no formal treatment.\n"
    "Significant Life Events: Recent breakup; job stress in a high-pressure "
    "environment; moved away from close friends.\n"
    "Behavioral Themes: Uses cannabis to manage anxiety and sleep; tends to "
    "isolate when stressed; difficulty with routines.\n"
    "Motivations for Substance Use: Escaping from racing thoughts and "
    "loneliness; fear of being unable to sleep without it.\n"
    "Current Motivation: Wants to reduce use but unsure about abstinence; "
    "worried about withdrawal, boredom, and social loss."
)

initial_state: DialogueState = {
    "history": [],  # empty: patient will start
    "patient_profile": example_patient_profile.strip(),
    "difficulty": "medium",
    "difficulty_description": DIFFICULTY_DESCRIPTIONS["medium"],
    "max_turns": 60,
    "turn_index": 0,
    "strategy_history": [],
    "patient_resolution_status": False,
    "dialogue_agenda_phase": "Rapport & Goal Alignment",
    "patient_state_summary": "",
    "therapist_micro_commitment": "",
}

print("Starting simulation...")
result_state = app.invoke(
    initial_state,
    config={"recursion_limit": 200}
)


def print_dialogue(history: List[Dict[str, str]]):
    """Prints the dialogue history in a readable format."""
    print("\n--- Dialogue Transcript ---\n")
    for i, msg in enumerate(history):
        prefix = "Patient" if msg["role"] == "patient" else "Therapist"
        print(f"{i + 1:02d} {prefix}: {msg['content']}\n")


# Display results
import json
import os
from datetime import datetime

# Print the simulated dialogue
print_dialogue(result_state["history"])

# Set output directory
output_dir = r"C:\Users\vikto\RecoveryBot Project"
os.makedirs(output_dir, exist_ok=True)

# Create timestamped filename inside output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"simulated_dialogue_{timestamp}.json"
output_path = os.path.join(output_dir, output_filename)

# Prepare data for saving
output_data = {
    "patient_profile": result_state["patient_profile"],
    "difficulty": result_state["difficulty"],
    "history": result_state["history"],
    "strategy_history": result_state["strategy_history"],
}

# Save JSON file
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2)

print(f"Saved dialogue to {output_path}")

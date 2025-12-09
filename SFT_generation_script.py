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
from collections import Counter
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
    {
        "id": "mi_safety_respect",
        "name": "Build safety and respect",
        "description": (
            "Create a safe and respectful environment by being empathetic, "
            "non-judgmental, and collaborative. Use affirmations and "
            "reflective listening to build rapport."
        ),
    },
    {
        "id": "mi_change_talk_valued_goals",
        "name": "Elicit change talk around personally valued goals",
        "description": (
            "Explore the patient's values and goals (e.g., health, family, "
            "legal) and connect them to the possibility of changing "
            "substance use. Elicit 'change talk' by asking about the "
            "benefits of change."
        ),
    },
    {
        "id": "mi_roll_with_resistance",
        "name": "Roll with resistance",
        "description": (
            "Avoid arguing or confronting resistance. Instead, reflect the "
            "patient's perspective and explore their ambivalence. Reframe "
            "resistance as a sign of the patient's engagement and autonomy."
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
    {
        "id": "cbt_structured_routine",
        "name": "Structured daily routine",
        "description": (
            "Collaboratively develop a structured daily routine to bring "
            "stability, reduce idle time, and incorporate healthy "
            "activities. This can help to disrupt old patterns of "
            "substance use."
        ),
    },
    {
        "id": "cbt_behavioral_activation",
        "name": "Behavioral activation",
        "description": (
            "Encourage the patient to engage in small, rewarding "
            "activities that are inconsistent with substance use. This "
            "can help to improve mood, reduce cravings, and build a "
            "sense of accomplishment."
        ),
    },
    {
        "id": "cbt_relaxation_grounding",
        "name": "Relaxation and grounding techniques",
        "description": (
            "Teach and practice relaxation and grounding techniques "
            "(e.g., 5-4-3-2-1, deep breathing) to help the patient "
            "manage anxiety, stress, and cravings in the moment."
        ),
    },
    {
        "id": "cbt_mindfulness",
        "name": "Brief mindfulness practices",
        "description": (
            "Introduce brief mindfulness practices to help the patient "
            "become more aware of their thoughts, feelings, and "
            "cravings without judgment. This can help to reduce "
            "reactivity and increase self-control."
        ),
    },
    {
        "id": "cbt_goal_setting_strength_review",
        "name": "Goal setting and strength review",
        "description": (
            "Collaboratively set realistic, achievable goals for "
            "recovery and review the patient's strengths and past "
            "successes. This can help to build self-efficacy and "
            "motivation."
        ),
    },
    {
        "id": "cbt_lifestyle_support",
        "name": "Lifestyle support",
        "description": (
            "Provide psychoeducation and support for healthy lifestyle "
            "changes, such as improving sleep, nutrition, and "
            "exercise. These changes can improve overall well-being "
            "and support recovery."
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
    patient_state_summary: str


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
            ),
            safety_settings=[
                {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_NONE'},
                {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_NONE'},
                {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'},
                {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_NONE'},
            ]
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

def therapist_node(state: DialogueState) -> Dict[str, Any]:
    """
    Generates the therapist's response using MI/CBT principles.
    """
    history_text = render_history_for_prompt(state["history"])

    # Track strategy usage
    strategy_counts = Counter(state["strategy_history"])
    strategy_usage_text = "\n".join(
        [f"- {strategy}: {count} times used." for strategy, count in strategy_counts.items()]
    )
    if not strategy_usage_text:
        strategy_usage_text = "No strategies used yet."

    therapist_instructions_template = """
As a therapist meeting this patient for the first time, your goal is to have a natural, empathetic, and multi-layered conversation.

**Patient Analysis:**
{user_analysis}

**Your Task:**
- **Incorporate Diverse Strategies:** Subtly weave in the following therapeutic approaches based on the patient's needs:
  - Motivational Interviewing (MI)
  - Cognitive Behavioral Therapy (CBT)
  - Solution-Focused Brief Therapy (SFBT)
  - Peer Support Programs
  - Mindfulness-Based Interventions (MBIs)
  - Behavioral Activation (BA)
  - Relapse Prevention Strategies
  - Strength-Based Approach
  - Psychoeducation on Addiction and Recovery
  - Harm Reduction Framework
  - Family and Social Support Involvement
  - Self-Compassion Practices
  - Coping Skill Development
- **Track Strategy Usage:** Be mindful of the strategies you've already used. Here's the current count:
  {strategy_usage}
- **Use Actionable Tools:** When appropriate, introduce coping mechanisms by selecting from the predefined actionable strategies.
- **Maintain a Natural Flow:** Ensure the dialogue feels like a real conversation. Use open-ended questions, be empathetic, and let the solutions emerge collaboratively.
- **Focus on the "Next Step":** Your primary goal is to generate the *next* therapist response in the conversation, not the entire dialogue at once.

**Example of a good, turn-by-turn response:**

*Patient: I’ve been feeling really stuck. I know I want to quit smoking, but every time I try, I just feel like I’m failing all over again.*

*Therapist: I hear you, Mark. Quitting smoking is one of the hardest challenges anyone can take on, and it’s completely natural to feel this way. I’ve worked with others who’ve felt the same—they described it as climbing a mountain that feels too steep. But I’ve also seen them reach the top, step by step. Can we talk about what makes the climb feel steep for you right now?*
"""

    therapist_instructions = therapist_instructions_template.format(
        user_analysis=state["patient_profile"], strategy_usage=strategy_usage_text
    )

    therapist_prompt = (
        "Conversation so far:\n"
        f"{history_text}\n\n"
        "Now write the therapist's next reply only. "
        "Do not include 'Therapist:' labels or any narration."
    )

    full_response = call_llm(
        model=MODEL_THERAPIST,
        instructions=therapist_instructions,
        input_text=therapist_prompt,
        max_output_tokens=1024,
    )

    # Parse the response to separate the dialogue from the strategies
    if "**Strategies:**" in full_response:
        parts = full_response.split("**Strategies:**")
        therapist_reply = parts[0].strip()
        strategies_used_str = parts[1].strip()
        strategies_used = [s.strip() for s in strategies_used_str.split(",")]
    else:
        therapist_reply = full_response.strip()
        strategies_used = []

    new_history = state["history"] + [{"role": "therapist", "content": therapist_reply}]
    new_turn_index = state["turn_index"] + 1
    new_strategy_history = state["strategy_history"] + strategies_used

    return {
        "history": new_history,
        "turn_index": new_turn_index,
        "strategy_history": new_strategy_history,
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
example_patient_profile = f"""
    The user appears to have a tendency towards self-blame ('sitting with how much of a fuck up i was'),
    impulsivity (struggling with stopping after one drink), and a past tendency towards irritability ('short tempered and didn't have patience').
    They also display resilience and a desire for change, actively seeking help and new coping mechanisms. There's an underlying restlessness that
    historically drove them to seek activity or distraction.\nSubstance Use History: The user has a history of alcohol use, with their longest streak
    of sobriety being 2 weeks in ten years, indicating a long-standing pattern of use. They recently relapsed after 40 days of sobriety and describe
    'back in the cycle of insanity.' They underwent a 21-day treatment program and attended AA/SMART recovery meetings, but stopped going
    to the latter. They also attempted to quit multiple times and struggled with cravings even for non-alcoholic beverages resembling alcohol
    early in sobriety.\nSignificant Life Events: The user reached out to their job for a Leave of Absence (LOA) to go to treatment due to being
    'SO sad all the time' and feeling like a 'prisoner.' This marked a turning point in seeking help. No other specific major life events like job loss,
    divorce, or trauma are explicitly mentioned as directly related to the addiction onset.\nBehavioral Themes: The user exhibits patterns of seeking
    distraction ('constantly trying to distract myself with anything'), difficulty with stillness ('couldn't stand being still'), and using alcohol as
    a coping mechanism for stress from a 'fast paced rather exhausting job.' Prior to sobriety, they were 'short tempered and didn't have patience for
    anything or anyone.' During sobriety, they've noted being 'much calmer' and 'more content just sitting around.' They also show a need for
    'controlled chaos' through activities like concerts. Relapse is a recurring theme, and they struggle with finding alternative routines for relaxation
    and reward after work.\nMotivations for Alcohol Use: The user drank to cope with stress from their fast-paced job, finding it an 'everyday
    thing because I would be so stressed out.' Alcohol was also used to socialize and feel happy, making 'things happy and everything was just a fun time.'
    They also mention 'searching for anything to make me feel better internally' and escaping the feeling of being 'a fuck up.'
    Boredom was 'excruciating' and led to drinking to avoid internal discomfort.
    Current Motivation: Wants to reduce use but unsure about abstinence; worried about withdrawal, boredom, and social loss

"""

initial_state: DialogueState = {
    "history": [],  # empty: patient will start
    "patient_profile": example_patient_profile.strip(),
    "difficulty": "medium",
    "difficulty_description": DIFFICULTY_DESCRIPTIONS["medium"],
    "max_turns": 50,
    "turn_index": 0,
    "strategy_history": [],
    "patient_resolution_status": False,
    "patient_state_summary": "",
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

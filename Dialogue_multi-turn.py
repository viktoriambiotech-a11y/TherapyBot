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

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from openai import OpenAI

# Load environment variables (OPENAI_API_KEY)
load_dotenv()

# Initialize OpenAI Client
# NOTE: Using environment variables for security.
# Ensure OPENAI_API_KEY is set in your .env file.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Model Configuration
MODEL_PATIENT = "gpt-4o"
MODEL_THERAPIST = "gpt-4o"
# Updated to valid model name format if needed


# Therapeutic Strategies Catalogs

MI_STRATEGIES = [
    {
        "id": "mi_scales",
        "name": "importance / confidence / readiness scale (0–10)",
        "description": (
            "Use 0–10 scaling questions about importance, confidence, or "
            "readiness to change. Ask the patient to pick numbers, explore "
            "why they chose them, and gently bring up 'change talk' about what "
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
            "Explore pros and cons of continuing current alcohol use versus "
            "making a change. Reflect ambivalence and highlight "
            "the reasons provided by the patient for wanting to change."
        ),
    },
    {
        "id": "mi_values",
        "name": "values clarification",
        "description": (
            "Ask about what kind of person they want to be, their core "
            "values (e.g., family, health), and connect those values to "
            "possible changes in alcohol use."
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


]

CBT_STRATEGIES = [
    {
        "id": "cbt_functional_analysis",
        "name": "functional analysis of a recent episode",
        "description": (
        "Invite the patient to describe their most recent use or near-use event. "
        "Ask about what led up to it (situations or triggers), their thoughts and emotions, "
        "the actions they took, and what happened afterward. "
        "Reflect any patterns that appear to contribute to continued alcohol use."
        )
    },

    {
        "id": "cbt_trigger_mapping",
        "name": "trigger mapping",
        "description": (
            "Help the patient identify external triggers (people, places, situations, times of day) "
            "and internal triggers (thoughts, emotions, bodily sensations) that tend to precede cravings. "
            "Guide them to walk through the full chain: the trigger, the automatic thoughts that followed, "
            "the emotional and physical reactions, the resulting urge, and the behavior that came next. "
            "Summarize any recurring patterns that link triggers to cravings and alcohol use."
        )
    },
    {
        "id": "cbt_reappraisal",
        "name": "cognitive reappraisal of urges",
        "description": (
            "Gently question unhelpful automatic thoughts (e.g., 'I can’t "
            "get through the evening without having a drink') and help the patient "
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
            "Work with the patient to create a short, realistic refusal statement they can use when offered alcohol. "
            "Explore common situations or people who might pressure them to drink, and help them craft responses that feel natural and confident. "
            "Rehearse how they might say the refusal line, consider alternative wording if needed, and problem-solve any anticipated challenges in delivering it."
        )
    },

    {
        "id": "cbt_stimulus_control",
        "name": "stimulus control",
        "description": (
            "Help the patient identify specific cues, environments, or situations that increase the likelihood of use or cravings. "
            "Brainstorm practical strategies to reduce or limit contact with these cues—such as changing routines, avoiding certain locations or digital spaces, or adjusting social contacts. "
            "Collaborate on selecting one small, manageable action they can implement soon, and clarify how they plan to carry it out."
        )
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
            "Work with the patient to identify a concrete, real-world barrier that makes change difficult "
            "— such as transportation, scheduling, finances, or access to support. "
            "Guide them through structured problem solving: clearly define the barrier, generate several possible solutions, "
            "choose one realistic step to try, and plan when and how they will carry it out. "
            "Check for confidence and adjust the plan if needed."
        )
    },

    {
        "id": "cbt_behavioral_activation",
        "name": "Behavioral activation (small tasks)",
        "description": "Encourage engagement in small, manageable tasks and activities to build momentum and self-efficacy.",
    },

    {
        "id": "cbt_goal_setting",
        "name": "Goal setting and strength review",
        "description": "Collaboratively set achievable short-term goals and review the patient's strengths.",
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
            "Develop a structured daily routine to bring stability and " "reduce idle time that might trigger relapse."
        ),
    },
    {
        "id": "act_grounding",
        "name": "grounding techniques",
        "description": (
            "Introduce grounding techniques such as sensory exercises or "
            "physical activities to manage anxiety or cravings."
            "Introduce techniques like 5-4-3-2-1 grounding to manage anxiety and cravings."
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
        "description": ("Provide psychoeducation on how addiction affects the brain " "and emotional regulation."),
    },
    {
        "id": "act_emotional_triggers",
        "name": "identify emotional triggers",
        "description": (
            "Work on identifying and addressing specific emotional triggers " "through reflective exercises."
        ),
    },
    {
        "id": "act_assertive_comm",
        "name": "assertive communication techniques",
        "description": (
            "Practice assertive communication techniques for setting "
            "boundaries with peers or environments that encourage "
            "alcohol use."
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
            "Incorporate brief mindfulness exercises to improve awareness and reduce reactivity to triggers."
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
        patient_profile_summary: A concise summary of the patient profile.
        difficulty: The set difficulty level (easy/medium/hard).
        difficulty_description: Instructions on resistance level.
        max_turns: Target total turns.
        turn_index: Current 0-based turn count.
        strategy_history: List of strategy IDs used so far.
        patient_resolution_status: Boolean indicating if the patient has achieved resolution.
        patient_state_summary: A summary of the patient's state.
        session_number: The current session number (1-6).
    """

    history: List[Dict[str, str]]
    patient_profile: str
    patient_profile_summary: str
    difficulty: Literal["easy", "medium", "hard"]
    difficulty_description: str
    max_turns: int
    turn_index: int
    strategy_history: List[str]
    patient_resolution_status: bool
    patient_state_summary: str
    session_number: int


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
        "You have long-standing alcohol use and substantial mistrust or "
        "skepticism about treatment. You often challenge or deflect "
        "suggestions, emphasize barriers, and may minimize the need for "
        "change."
    ),
}


SESSION_GOALS = {
    1: {
        "cbt_stage_goal": "Build trust, assessment, set safety/limits.",
        "mi_focus": "Engage: Use 0-10 importance/confidence rulers; start trigger/urge log.",
    },
    2: {
        "cbt_stage_goal": "Identifying negative cognitions.",
        "mi_focus": "Focus: Review log; use values/agenda mapping to map thought → feeling → body → action.",
    },
    3: {
        "cbt_stage_goal": "Challenging false beliefs.",
        "mi_focus": "Evoke: Elicit change talk (DARN) using double-sided reflections; challenge the core thought (e.g., 'one time won't hurt').",
    },
    4: {
        "cbt_stage_goal": "Restructuring cognitive patterns.",
        "mi_focus": "Bridge Evoke → Plan: Formulate an If-Then plan; move change talk to CAT (Commitment, Activation, Taking steps).",
    },
    5: {
        "cbt_stage_goal": "Behavioral skill building.",
        "mi_focus": "Plan: Redesign a high-risk window; create a crisis micro-plan and plan for barriers.",
    },
    6: {
        "cbt_stage_goal": "Consolidation & termination.",
        "mi_focus": "Maintenance: Review gains; extend the plan for relapse prevention and peer support.",
    },
}


def summarize_patient_profile(profile: str) -> str:
    """
    Uses an LLM to create a concise summary of the patient profile.
    """
    instructions = (
        "Summarize the following patient profile into a concise paragraph. "
        "Focus on the key clinical details: primary issue, substance use history, "
        "behavioral patterns, and motivations. This summary will be used by a therapist bot "
        "to maintain context during a conversation."
    )
    summary = call_llm(
        model=MODEL_THERAPIST,
        instructions=instructions,
        input_text=profile,
        max_output_tokens=256,
    )
    return summary


def call_llm(model: str, instructions: str, input_text: str, max_output_tokens: int = 256) -> str:
    """
    Thin wrapper around the OpenAI Chat Completions API with error handling.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": input_text},
            ],
            max_tokens=max_output_tokens,
        )
        return response.choices[0].message.content.strip()
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
    Generates the patient's next utterance, summary, and resolution status in a single call.
    """
    history_text = render_history_for_prompt(state["history"])
    display_history = history_text if history_text else "(no prior conversation – this is the first turn)"

    instructions_for_json_output = """
You are role-playing as a patient in addiction recovery.
Speak from the profile below, staying consistent with the conversation so far.
Your difficulty level description explains how resistant or ambivalent you are.

Your task is to generate a single JSON object containing three fields: "reply", "summary", and "resolution_status".

1.  **reply**: Create the patient's next utterance based on the conversation history and their profile. This should be a natural, brief response in the patient's voice. Do not include narration or system messages.
2.  **summary**: Analyze the patient's message and the current situation to provide a compact summary of their state. This summary should cover aspects like craving levels, trigger salience, confidence, and any flags for recent lapses.
3.  **resolution_status**: Analyze the patient's message for indications that the session is complete. If the patient expresses sufficient motivation, confidence, and commitment to try a therapy micro-assignment, AND uses language that signals closure or readiness to end the dialogue (e.g., 'See you next time', 'I think we’ve covered everything', 'That helped a lot'), set this to `true`. Otherwise, set it to `false`.

The final output MUST be a valid JSON object and nothing else.
"""

    prompt = f"""
Patient Profile:
{state['patient_profile']}

Difficulty Setting:
{state['difficulty_description']}

Conversation So Far:
{display_history}

Based on the above, provide the next patient turn as a JSON object with "reply", "summary", and "resolution_status".
"""

    response_str = call_llm(
        model=MODEL_PATIENT,
        instructions=instructions_for_json_output,
        input_text=prompt,
        max_output_tokens=256,  # Increased to accommodate JSON structure and content
    )

    try:
        # The response might be enclosed in markdown ```json ... ```
        if response_str.startswith("```json"):
            response_str = response_str[7:-4]

        response_data = json.loads(response_str)
        patient_reply = response_data.get("reply", "[MISSING_REPLY]")
        patient_state_summary = response_data.get("summary", "[MISSING_SUMMARY]")
        patient_resolution_status = response_data.get("resolution_status", False)
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"--- ERROR PARSING PATIENT JSON RESPONSE ---")
        print(f"Failed to parse JSON: {e}")
        print(f"Raw response: {response_str}")
        # Provide fallback values to avoid crashing the graph
        patient_reply = response_str  # Use the raw string as a fallback for the reply
        patient_state_summary = "Error parsing patient state."
        patient_resolution_status = False

    new_history = state["history"] + [{"role": "patient", "content": patient_reply}]
    new_turn_index = state["turn_index"] + 1

    return {
        "history": new_history,
        "turn_index": new_turn_index,
        "patient_state_summary": patient_state_summary,
        "patient_resolution_status": patient_resolution_status,
    }


# Therapist Node Logic


def _get_session_agenda(session_number: int) -> str:
    """
    Generates the therapist's agenda for a given session.
    """
    if session_number == 1:
        intro = "review values, quit/cut-down intent and collaboratively select today’s top three items to discuss"
    else:
        intro = "review events, stressors, cravings, or challenges that occurred since the previous session and collaboratively select today’s top three items to discuss"

    agenda = f"""
Your agenda for this session (S{session_number}) is as follows:
1. rapport & goal alignment ({intro}).
2. episode clarification (perform functional analysis of a recent use/near-use: antecedents, triggers, craving peak, consequences)
3. plan formulation (translate insights from episode analysis into one or two actionable steps)
4. next-step micro-commitment. Each session ends with a concrete, time-bounded micro-assignment for the patient tied to risk and feasibility (for example: trigger/urge log (time, place, people, intensity),  a three-line coping card for an anticipated window, a single refusal-line rehearsal with a specific peer, a stimulus-control action (remove a procurement contact/app)).
"""
    return agenda


def therapist_node(state: DialogueState) -> Dict[str, Any]:
    """
    Generates the therapist's response using a summarized profile and strategy names to save tokens.
    """
    history_text = render_history_for_prompt(state["history"])

    # Track strategy usage
    strategy_counts = Counter(state["strategy_history"])
    strategy_usage_text = "\n".join(
        [f"- {strategy}: {count} times used." for strategy, count in strategy_counts.items()]
    )
    if not strategy_usage_text:
        strategy_usage_text = "No strategies used yet."

    def get_strategy_names(strategy_list: List[Dict[str, str]]) -> str:
        return ", ".join([f'"{item["name"]}"' for item in strategy_list])

    # Get the session-specific goals and agenda
    session_number = state["session_number"]
    session_goal = SESSION_GOALS.get(session_number, {})
    cbt_goal = session_goal.get("cbt_stage_goal", "N/A")
    mi_focus = session_goal.get("mi_focus", "N/A")
    session_agenda = _get_session_agenda(session_number)

    therapist_instructions_template = """
You are an expert therapist in a role-play simulation. Your goal is to conduct a therapeutic dialogue with a patient based on their profile summary.
You should be empathetic, non-judgmental, and collaborative.

PATIENT SUMMARY:
{user_analysis}

SESSION {session_number}:
- CBT Goal: {cbt_goal}
- MI Focus: {mi_focus}

STRATEGY USAGE:
{strategy_usage}

AVAILABLE STRATEGIES:
- MI Strategies: {MI_STRATEGIES}
- CBT Strategies: {CBT_STRATEGIES}
- Actionable Tools: {ACTIONABLE_TOOLS}

INSTRUCTIONS:
1. Read the patient summary and conversation history carefully.
2. Follow the session agenda provided below as an internal guide. Do not mention the agenda items (e.g., "Functional Analysis," "Actionable Step") in your response.
3. Select relevant strategies from the available lists to guide your response. Adapt your approach based on the patient's needs and avoid overusing the same strategies.
4. Ask open-ended questions to explore the patient's challenges, motivations, and triggers.
4. Build rapport using affirmations and reflective listening.
5. If you suggest a coping mechanism or tool, refer to the "Actionable Tools" list.
6. Keep your response concise and natural.
7. Write the therapist's next reply only. Do not include 'Therapist:' labels or any narration.

Here is an example of a layered, empathetic dialogue:
Patient: Hi. . . um, thanks for seeing me today. I wasn’t sure what to expect.
Therapist: Hi Mark, I really appreciate you coming in. Starting this process can feel overwhelming,
but I’m here to support you. What’s been on your mind lately?
Patient: I’ve been feeling really stuck. I know I want to quit smoking, but every time I try, I just
feel like I’m failing all over again.
Therapist: I hear you, Mark. Quitting smoking is one of the hardest challenges anyone can take
on, and it’s completely natural to feel this way. I’ve worked with others who’ve felt the same—they
described it as climbing a mountain that feels too steep. But I’ve also seen them reach the top,
step by step. Can we talk about what makes the climb feel steep for you right now?
Patient: It’s the cravings. They just hit me out of nowhere, and I don’t know how to handle them.
Therapist: Cravings can feel like a storm, can’t they? I worked with someone once who described
their cravings as waves that kept crashing over them. Together, we found ways for them to ride out
those waves, like focusing on a small activity or changing their environment. Could we explore
some strategies that might help you ride out your cravings too?
Patient: Sure, I guess.
Therapist: Great. Let’s start with understanding when these cravings hit hardest. For example, is
it during specific times of day or situations?

After your response, you MUST list the strategies you used on a new line. Use the format:
**Strategies:** Strategy Name 1, Strategy Name 2

CONVERSATION SO FAR:
{history_text}

SESSION AGENDA:
{session_agenda}
"""

    therapist_instructions = therapist_instructions_template.format(
        user_analysis=state["patient_profile_summary"],
        session_number=session_number,
        cbt_goal=cbt_goal,
        mi_focus=mi_focus,
        history_text=history_text,
        strategy_usage=strategy_usage_text,
        MI_STRATEGIES=get_strategy_names(MI_STRATEGIES),
        CBT_STRATEGIES=get_strategy_names(CBT_STRATEGIES),
        ACTIONABLE_TOOLS=get_strategy_names(ACTIONABLE_TOOLS),
        session_agenda=session_agenda,
    )

    # The user prompt is now just a trigger to generate the response based on the system prompt.
    therapist_prompt = "Therapist:"

    full_response = call_llm(
        model=MODEL_THERAPIST,
        instructions=therapist_instructions,
        input_text=therapist_prompt,
        max_output_tokens=512,
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
# replace 'example_patient_profile' with synthesized profiles

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

difficulty_setting = "hard"

# Generate a concise summary of the patient profile to save tokens
print("Summarizing patient profile...")
patient_profile_summary = summarize_patient_profile(example_patient_profile.strip())
print("Summary complete.")

# Store the data for all sessions
sessions_data = []

for session_number in range(1, 7):
    print(f"--- Running Session #{session_number} ---")

    # The patient starts the first session, but the therapist should start all subsequent sessions.
    history_for_this_session = []
    if session_number > 1:
        history_for_this_session.append(
            {"role": "therapist", "content": "Welcome back. How have you been since our last session?"}
        )

    initial_state: DialogueState = {
        "history": history_for_this_session,
        "patient_profile": example_patient_profile.strip(),
        "patient_profile_summary": patient_profile_summary,
        "difficulty": difficulty_setting,
        "difficulty_description": DIFFICULTY_DESCRIPTIONS[difficulty_setting],
        "max_turns": 60,
        "turn_index": 0,
        "strategy_history": [],
        "patient_resolution_status": False,
        "patient_state_summary": "",
        "session_number": session_number,
    }

    # Invoke the graph for the current session
    result_state = app.invoke(initial_state, config={"recursion_limit": 200})

    # Get the unique strategies used in this session
    strategies_this_session = sorted(list(set(result_state.get("strategy_history", []))))

    # Store the results for this session
    session_data = {
        "session_number": session_number,
        "session_goals": SESSION_GOALS.get(session_number, {}),
        "dialogue": result_state["history"],
        "strategies_used": strategies_this_session,
    }
    sessions_data.append(session_data)

    if strategies_this_session:
        print(f"**Strategies Used:** {', '.join(strategies_this_session)}")

    print(f"\n--- Session {session_number} Complete ---\n")

# Set output directory
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Create timestamped filename inside output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"simulated_dialogue_{timestamp}.json"
output_path = os.path.join(output_dir, output_filename)

# Prepare data for saving
output_data = {
    "patient_profile": example_patient_profile.strip(),
    "difficulty": difficulty_setting,
    "sessions": sessions_data,
}

# Save JSON file
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2)

print(f"Saved dialogue to {output_path}")

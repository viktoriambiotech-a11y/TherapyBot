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
import numpy as np

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

ENVIRONMENT_STRESSORS = [
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

SESSION_AGENDAS = {
    1: [
        {
            "phase": "Rapport & Goal Alignment",
            "turn_range": (0, 20),
            "strategies": ["mi_agenda", "mi_values", "mi_scales"],
        },
        {
            "phase": "Episode Clarification",
            "turn_range": (21, 40),
            "strategies": ["cbt_functional_analysis", "cbt_trigger_mapping"],
        },
        {
            "phase": "Closing & Micro-Commitment",
            "turn_range": (41, 60),
            "strategies": ["cbt_coping_skills", "cbt_goal_setting", "act_crisis_plan"],
        },
    ],
    2: [
        {
            "phase": "Check-in & Review",
            "turn_range": (0, 20),
            "strategies": ["mi_scales", "cbt_functional_analysis"],
        },
        {
            "phase": "Identifying Negative Cognitions",
            "turn_range": (21, 40),
            "strategies": ["cbt_trigger_mapping", "cbt_reappraisal"],
        },
        {
            "phase": "Closing & Micro-Commitment",
            "turn_range": (41, 60),
            "strategies": ["cbt_coping_skills", "act_journaling"],
        },
    ],
    3: [
        {
            "phase": "Check-in & Review",
            "turn_range": (0, 20),
            "strategies": ["mi_decisional_balance", "cbt_problem_solving"],
        },
        {
            "phase": "Challenging False Beliefs",
            "turn_range": (21, 40),
            "strategies": ["cbt_reappraisal", "mi_ep_e"],
        },
        {
            "phase": "Closing & Micro-Commitment",
            "turn_range": (41, 60),
            "strategies": ["cbt_refusal", "act_assertive_comm"],
        },
    ],
    4: [
        {
            "phase": "Check-in & Review",
            "turn_range": (0, 20),
            "strategies": ["mi_scales", "cbt_functional_analysis"],
        },
        {
            "phase": "Restructuring Cognitive Patterns",
            "turn_range": (21, 40),
            "strategies": ["cbt_coping_skills", "cbt_stimulus_control"],
        },
        {
            "phase": "Closing & Micro-Commitment",
            "turn_range": (41, 60),
            "strategies": ["act_routine", "act_hobbies"],
        },
    ],
    5: [
        {
            "phase": "Check-in & Review",
            "turn_range": (0, 20),
            "strategies": ["mi_values", "cbt_problem_solving"],
        },
        {
            "phase": "Behavioral Skill Building",
            "turn_range": (21, 40),
            "strategies": ["cbt_behavioral_activation", "cbt_exposure"],
        },
        {
            "phase": "Closing & Micro-Commitment",
            "turn_range": (41, 60),
            "strategies": ["act_support_group", "act_community"],
        },
    ],
    6: [
        {
            "phase": "Review & Consolidate",
            "turn_range": (0, 20),
            "strategies": ["cbt_goal_setting", "act_strengths"],
        },
        {
            "phase": "Relapse Prevention & Future Planning",
            "turn_range": (21, 40),
            "strategies": ["act_crisis_plan", "act_health"],
        },
        {
            "phase": "Closing & Termination",
            "turn_range": (41, 60),
            "strategies": ["act_goals", "act_complementary_therapy"],
        },
    ],
}


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
        patient_internal_state: A dictionary holding the patient's internal state vector.
        last_action: The last action taken by the patient (cope or urge-driven).
        therapist_strategy_classification: The classification of the therapist's last strategy.
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
    stressor_ledger: List[Dict[str, Any]]
    session_number: int
    current_agenda_phase: str
    patient_internal_state: Dict[str, float]
    last_action: Literal["cope", "urge-driven", "none"]
    therapist_strategy_classification: str


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

    instructions_for_json_output = f"""
You are role-playing as a patient in addiction recovery.
Speak from the profile below, staying consistent with the conversation so far.
Your difficulty level description explains how resistant or ambivalent you are.

YOUR CURRENT INTERNAL STATE:
- Craving Level: {state['patient_internal_state']['craving_level']:.2f} (A high value means you are strongly craving alcohol)
- Stress Level: {state['patient_internal_state']['stress_level']:.2f} (A high value means you feel very overwhelmed)
- Self-Efficacy: {state['patient_internal_state']['self_efficacy']:.2f} (A low value means you doubt your ability to stay sober)
- Emotional Pain: {state['patient_internal_state']['emotional_pain']:.2f} (A high value means you are in significant emotional distress)
- Shame: {state['patient_internal_state']['shame']:.2f} (A high value means you feel intense self-criticism)

INSTRUCTIONS:
- Use your internal state to shape your response. For example, high stress and craving should make your tone more tense or irritable. Low self-efficacy should lead to more doubtful or hopeless language.
- Briefly report any significant stressors, cravings, or challenges since the last session, guided by your internal state. Describe them in your own words. Do NOT mention your internal state values directly.
- Then, continue the session by responding authentically to the therapist.

Your task is to generate a single JSON object containing three fields: "reply", "summary", and "resolution_status".

1.  **reply**: Create your next utterance. It should be a natural, brief response in your voice. Do not include narration.
2.  **summary**: Provide a compact summary of your current state, covering craving levels, trigger salience, confidence, and any recent lapses.
3.  **resolution_status**: Set to `true` if you are ready to end the session with a clear commitment to an action plan. Otherwise, set to `false`.

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

def policy_node(state: DialogueState) -> Dict[str, Any]:
    """
    Selects therapeutic strategies based on the current session and turn number.
    """
    session_number = state.get("session_number", 1)
    turn_index = state.get("turn_index", 0)

    # Get the agenda for the current session
    session_agenda = SESSION_AGENDAS.get(session_number, [])

    # Find the current phase based on the turn index
    for phase_info in session_agenda:
        min_turn, max_turn = phase_info["turn_range"]
        if min_turn <= turn_index <= max_turn:
            return {
                "current_agenda_phase": phase_info["phase"],
                "recommended_strategies": phase_info["strategies"],
            }

    # Default if no phase matches (e.g., if turn_index exceeds defined ranges)
    return {
        "current_agenda_phase": "Default",
        "recommended_strategies": ["mi_scales", "cbt_functional_analysis"],
    }


def therapist_node(state: DialogueState) -> Dict[str, Any]:
    """
    Generates the therapist's response using a summarized profile and strategy names to save tokens.
    """
    history_text = render_history_for_prompt(state["history"])

    # Step A: Select strategies using the Policy Node
    policy_decision = policy_node(state)
    current_agenda_phase = policy_decision["current_agenda_phase"]
    recommended_strategies = policy_decision["recommended_strategies"]

    # Track strategy usage
    strategy_counts = Counter(state["strategy_history"])
    strategy_usage_text = "\n".join(
        [f"- {strategy}: {count} times used." for strategy, count in strategy_counts.items()]
    )
    if not strategy_usage_text:
        strategy_usage_text = "No strategies used yet."

    def get_strategy_names(strategy_list: List[Dict[str, str]]) -> str:
        return ", ".join([f'"{item["name"]}"' for item in strategy_list])

    therapist_instructions = f"""
You are an expert therapist in a role-play simulation. Your goal is to conduct a therapeutic dialogue with a patient based on their profile summary.
You should be empathetic, non-judgmental, and collaborative.

PATIENT SUMMARY:
{state["patient_profile_summary"]}

LATEST PATIENT STATE SUMMARY:
{state["patient_state_summary"]}

PATIENT INTERNAL STATE:
- Craving: {state['patient_internal_state']['craving_level']:.2f}
- Stress: {state['patient_internal_state']['stress_level']:.2f}
- Self-Efficacy: {state['patient_internal_state']['self_efficacy']:.2f}
- Motivation: {state['patient_internal_state']['motivation_for_change']:.2f}
- Habit Strength: {state['patient_internal_state']['habit_strength']:.2f}
- Shame: {state['patient_internal_state']['shame']:.2f}

LAST ACTION OUTCOME: {state['last_action']}

CURRENT SESSION PHASE: {current_agenda_phase}

STRATEGY USAGE:
{strategy_usage_text}

RECOMMENDED STRATEGIES FOR THIS PHASE:
{recommended_strategies}

AVAILABLE STRATEGIES:
- MI Strategies: {get_strategy_names(MI_STRATEGIES)}
- CBT Strategies: {get_strategy_names(CBT_STRATEGIES)}
- Actionable Tools: {get_strategy_names(ACTIONABLE_TOOLS)}

INSTRUCTIONS:
- Your primary goal is to help the patient based on their state and the session phase. Use the patient's internal state to guide your choice of strategy. High stress might call for validation, while high motivation might be a good time for directive advice.
- At the start of the session, listen for and explore any stressors the patient reports.
- Your main task is to generate a single JSON object with two fields: "reply" and "strategy_classification".
  1. **reply**: Write your next utterance to the patient. Keep it concise, empathetic, and natural.
  2. **strategy_classification**: Classify your reply into ONE of the following categories: "MI-style reflection", "Validation", "Directive advice", "Confrontational style", or "Other". This classification will be used to update the patient's state.

The final output MUST be a valid JSON object.

CONVERSATION SO FAR:
{history_text}
"""

    # The user prompt is now just a trigger to generate the response based on the system prompt.
    therapist_prompt = "Generate your JSON response."

    response_str = call_llm(
        model=MODEL_THERAPIST,
        instructions=therapist_instructions,
        input_text=therapist_prompt,
        max_output_tokens=512,
    )

    try:
        if response_str.startswith("```json"):
            response_str = response_str[7:-4]
        response_data = json.loads(response_str)
        therapist_reply = response_data.get("reply", "[MISSING_REPLY]")
        strategy_classification = response_data.get("strategy_classification", "Other")
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"--- ERROR PARSING THERAPIST JSON RESPONSE ---")
        print(f"Failed to parse JSON: {e}")
        print(f"Raw response: {response_str}")
        therapist_reply = response_str
        strategy_classification = "Other"


    new_history = state["history"] + [{"role": "therapist", "content": therapist_reply}]
    new_turn_index = state["turn_index"] + 1

    return {
        "history": new_history,
        "turn_index": new_turn_index,
        "therapist_strategy_classification": strategy_classification,
        "current_agenda_phase": current_agenda_phase,
    }


def state_update_node(state: DialogueState) -> Dict[str, Any]:
    """
    Updates the patient's internal state based on stressors, decisions, and therapy.
    """
    s = state["patient_internal_state"].copy()

    # Define model parameters (these can be tuned)
    alpha_stress = 0.1  # Impact of each stressor
    lambda_decay = 0.05 # Decay of unprocessed stressors
    beta = [0.2, 0.3, 0.15, 0.15, 0.2] # Weights for craving calculation
    weights_decision = [0.4, 0.3, 0.2, 0.1, 0.2, 0.3, 0.1] # Weights for relapse probability
    eta_h_pos, eta_e_neg, eta_q_pos = 0.1, 0.05, 0.05 # Learning rates for urge-driven behavior
    eta_h_neg, eta_e_pos, eta_k_pos = 0.05, 0.05, 0.05 # Learning rates for coping behavior

    # 1. Process Stressors
    unprocessed_stressors = [s for s in state.get("stressor_ledger", []) if not s.get("processed")]
    stress_increase = len(unprocessed_stressors) * alpha_stress
    s["stress_level"] = min(1, s["stress_level"] + stress_increase + lambda_decay * len(unprocessed_stressors))
    for stressor in unprocessed_stressors:
        stressor["processed"] = True

    # 2. Update Craving Dynamics
    s['craving_level'] = np.clip(
        beta[0] * s['stress_level'] +
        beta[1] * s['habit_strength'] -
        beta[2] * s['cognitive_control'] -
        beta[3] * s['self_efficacy'] +
        beta[4] * s['emotional_pain'],
        0, 1
    )

    # 3. Decision Policy (Relapse Probability)
    logit = (
        weights_decision[0] * s['craving_level'] +
        weights_decision[1] * s['stress_level'] +
        weights_decision[2] * s['habit_strength'] -
        weights_decision[3] * s['cognitive_control'] -
        weights_decision[4] * s['self_efficacy'] -
        weights_decision[5] * s['motivation_for_change'] +
        weights_decision[6] * s['shame']
    )
    p_urge = 1 / (1 + np.exp(-logit))
    action = "urge-driven" if np.random.rand() < p_urge else "cope"

    # 4. Learning and Update Rules based on action
    if action == "urge-driven":
        s['habit_strength'] += eta_h_pos * (1 - s['habit_strength'])
        s['self_efficacy'] -= eta_e_neg
        s['shame'] += eta_q_pos
    else:  # cope
        s['habit_strength'] -= eta_h_neg
        s['self_efficacy'] += eta_e_pos
        s['cognitive_control'] += eta_k_pos

    # 5. Apply Therapy Impact
    if state.get("therapist_strategy_classification"):
        if state["therapist_strategy_classification"] == "MI-style reflection":
            s["motivation_for_change"] += 0.05
            s["self_efficacy"] += 0.05
        elif state["therapist_strategy_classification"] == "Validation":
            s["shame"] -= 0.1
            s["emotional_pain"] -= 0.05
        elif state["therapist_strategy_classification"] == "Directive advice":
            s["cognitive_control"] += 0.05
            # Could also increase shame if poorly timed, simplified for now
        elif state["therapist_strategy_classification"] == "Confrontational style":
            # Simplified, context would matter greatly here
            s["motivation_for_change"] += 0.02
            s["emotional_pain"] += 0.02

    # Clip all values to be within [0, 1]
    for key in s:
        s[key] = np.clip(s[key], 0, 1)

    return {"patient_internal_state": s, "last_action": action}


# Graph Routing and Construction

def environment_agent_node(state: DialogueState) -> DialogueState:
    """
    Simulates an environmental stressor between sessions.
    """
    # 1. Sample a stressor
    stressor = random.choice(ENVIRONMENT_STRESSORS)

    # 2. Log the stressor
    new_stressor_ledger = state.get("stressor_ledger", []) + [stressor]

    # Print the stressor if it's not the first session
    if state.get("session_number", 1) > 1:
        print(f"\n--- Stressor Injected Between Sessions ---\n{stressor['Description']}\n")

    # 3. Apply rule-based updates to patient state
    # (This is a simplified example)
    if "Work" in stressor["Category"]:
        # This would be replaced with more sophisticated logic
        print(f"Applying stressor: {stressor['Description']}")

    # 4. Reset for the next session
    return {
        **state,
        "turn_index": 0,
        "stressor_ledger": new_stressor_ledger,
        "patient_resolution_status": False,
    }

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

graph.add_node("environment", environment_agent_node)
graph.add_node("state_update", state_update_node)
graph.add_node("patient", patient_node)
graph.add_node("therapist", therapist_node)

graph.set_entry_point("environment")
graph.add_edge("environment", "state_update")
graph.add_edge("state_update", "patient")

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
        "patient": "state_update",  # Loop back to state_update after therapist
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

initial_state: DialogueState = {
    "history": [],  # empty: patient will start
    "patient_profile": example_patient_profile.strip(),
    "patient_profile_summary": patient_profile_summary,
    "difficulty": difficulty_setting,
    "difficulty_description": DIFFICULTY_DESCRIPTIONS[difficulty_setting],
    "max_turns": 60,
    "turn_index": 0,
    "strategy_history": [],
    "patient_resolution_status": False,
    "patient_state_summary": "",
    "stressor_ledger": [],
    "session_number": 1,
    "current_agenda_phase": "",
    "patient_internal_state": {
        "craving_level": 0.6,
        "stress_level": 0.7,
        "self_efficacy": 0.4,
        "emotional_pain": 0.6,
        "cognitive_control": 0.4,
        "motivation_for_change": 0.5,
        "habit_strength": 0.7,
        "shame": 0.5
    },
    "last_action": "none",
    "therapist_strategy_classification": ""
}

print("Starting simulation...")
NUM_SESSIONS = 6
current_state = initial_state
all_session_results = []

for session_num in range(NUM_SESSIONS):
    print(f"--- Starting Session {session_num + 1} ---")
    current_state["session_number"] = session_num + 1
    print(f"\n--- Patient State at Start of Session {session_num + 1} ---")
    print(current_state["patient_internal_state"])
    # The graph now handles the environment node internally
    result_state = app.invoke(current_state, config={"recursion_limit": 200})
    all_session_results.append(result_state)
    print(f"\n--- Patient State at End of Session {session_num + 1} ---")
    print(result_state["patient_internal_state"])
    # The state for the next session is the output of the previous one
    current_state = result_state


def print_dialogue(history: List[Dict[str, str]]):
    """Prints the dialogue history in a readable format."""
    print("\n--- Dialogue Transcript ---\n")
    for i, msg in enumerate(history):
        prefix = "Patient" if msg["role"] == "patient" else "Therapist"
        print(f"{i + 1:02d} {prefix}: {msg['content']}\n")


# Display results
for i, session_result in enumerate(all_session_results):
    print(f"\n--- Session {i + 1} Transcript ---")
    print_dialogue(session_result["history"])

    print(f"\n--- Session {i + 1} Strategies Used ---")
    if session_result["strategy_history"]:
        unique_strategies = sorted(list(set(session_result["strategy_history"])))
        for strategy in unique_strategies:
            print(f"- {strategy}")
    else:
        print("No strategies were recorded for this session.")


# Set output directory
output_dir = "C:/Users/vikto/Code/LangChain_code/SFT_Multi_turn_dialogue"
os.makedirs(output_dir, exist_ok=True)

# Create timestamped filename inside output directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_filename = f"simulated_dialogue_{timestamp}.json"
output_path = os.path.join(output_dir, output_filename)

# Prepare data for saving
output_data = [
    {
        "session_number": i + 1,
        "patient_profile": session_result["patient_profile"],
        "difficulty": session_result["difficulty"],
        "history": session_result["history"],
        "strategy_history": session_result["strategy_history"],
        "stressor_ledger": session_result.get("stressor_ledger", []),
    }
    for i, session_result in enumerate(all_session_results)
]

# Save JSON file
with open(output_path, "w", encoding="utf-8") as f:
    json.dump({"sessions": output_data}, f, indent=2)

print(f"Saved dialogue to {output_path}")

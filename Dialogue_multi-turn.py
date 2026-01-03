# -*- coding: utf-8 -*-
"""
Multiturn Therapeutic Dialogues Generation

Recreates a multi-session, multi-turn role-play setup:
 - Therapist Agent Node: Reads patient state + stressor ledger, Chooses intervention style, Generates dialogue
 - Patient Agent Node: Maintains internal state vector, Reports subjective experience, Makes decisions via State Update Node (relapse/no relapse)
 - State Update Node (Non-LLM): Applies equations, Updates state deterministically
 - Environment Node: Samples stressors, Updates stressor ledger

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

ENVIRONMENT_STRESSORS = [
  {
    "Category": "Social/Environmental",
    "Stressor": "Peer pressure",
    "Description": "Friends invited me to a bar after work",
    "Severity": "3",
    "Likely Duration": "Minutes"
  },
  {
    "Category": "Social/Environmental",
    "Stressor": "Exposure to drinking cues",
    "Description": "Walked past my usual liquor store after a stressful day",
    "Severity": "2",
    "Likely Duration": "Minutes"
  },

  {
    "Category": "Social/Environmental",
    "Stressor": "Drinking-centered events",
    "Description": "Attended an event where alcohol was central",
    "Severity": "2",
    "Likely Duration": "Hours-Days"
  },
  {
    "Category": "Social/Environmental",
    "Stressor": "Lack of social support",
    "Description": "No one to call when cravings hit",
    "Severity": "2",
    "Likely Duration": "Hours"
  },
  {
    "Category": "Social/Environmental",
    "Stressor": "Social isolation",
    "Description": "Spent the weekend alone with no plans",
    "Severity": "3",
    "Likely Duration": "Days–Weeks"
  },

  {
    "Category": "Interpersonal",
    "Stressor": "Relationship conflict",
    "Description": "Unresolved argument with partner",
    "Severity": "3",
    "Likely Duration": "Days–Weeks"
  },
  {
    "Category": "Interpersonal",
    "Stressor": "Breakup/rejection",
    "Description": "Partner ended relationship unexpectedly",
    "Severity": "3",
    "Likely Duration": "Weeks–Months"
  },
  {
    "Category": "Interpersonal",
    "Stressor": "Family conflict",
    "Description": "Ongoing tension or conflict with family members",
    "Severity": "3",
    "Likely Duration": "Weeks–Months"
  },

  {
    "Category": "Work/Academic",
    "Stressor": "Deadlines/pressure",
    "Description": "Multiple deadlines in same week",
    "Severity": "2",
    "Likely Duration": "Days–Weeks"
  },
  {
    "Category": "Work/Academic",
    "Stressor": "Job insecurity",
    "Description": "Fear of layoffs or reduced hours",
    "Severity": "2",
    "Likely Duration": "Weeks–Months"
  },

  {
    "Category": "Work/Academic",
    "Stressor": "Burnout/overwork",
    "Description": "Sustained overtime with little recovery",
    "Severity": "2",
    "Likely Duration": "Weeks–Months"
  },
  {
    "Category": "Work/Academic",
    "Stressor": "Academic failure",
    "Description": "Failed an important exam",
    "Severity": "2",
    "Likely Duration": "Days–Weeks"
  },
  {
    "Category": "Emotional/Cognitive",
    "Stressor": "Negative emotions",
    "Description": "Persistent anxiety, anger, or sadness",
    "Severity": "3",
    "Likely Duration": "Hours–Weeks"
  },
  {
    "Category": "Emotional/Cognitive",
    "Stressor": "Boredom/anhedonia",
    "Description": "Nothing feels engaging or rewarding",
    "Severity": "1",
    "Likely Duration": "Days–Weeks"
  },
  {
    "Category": "Emotional/Cognitive",
    "Stressor": "Overconfidence",
    "Description": "Feeling cured; belief one drink is safe",
    "Severity": "2",
    "Likely Duration": "Hours–Days"
  },
  {
    "Category": "Emotional/Cognitive",
    "Stressor": "Low self-efficacy",
    "Description": "Doubting ability to stay sober",
    "Severity": "3",
    "Likely Duration": "Weeks–Months"
  },
  {
    "Category": "Emotional/Cognitive",
    "Stressor": "Cognitive distortions",
    "Description": "I've already slipped so it doesn't matter",
    "Severity": "3",
    "Likely Duration": "Hours–Days"
  },
  {
    "Category": "Emotional/Cognitive",
    "Stressor": "Rumination",
    "Description": "Can't stop thinking about drinking",
    "Severity": "2",
    "Likely Duration": "Days"
  },
  {
    "Category": "Physical/Biological",
    "Stressor": "Pain or injury",
    "Description": "Chronic or acute pain",
    "Severity": "2",
    "Likely Duration": "Days–Weeks"
  },

  {
    "Category": "Physical/Biological",
    "Stressor": "Sleep deprivation",
    "Description": "Several nights of poor sleep due to racing thoughts or insomnia",
    "Severity": "2",
    "Likely Duration": "Days–Weeks"
  },
  {
    "Category": "Physical/Biological",
    "Stressor": "Withdrawal",
    "Description": "Ongoing irritability and anxiety post-quitting",
    "Severity": "2",
    "Likely Duration": "Weeks–Months"
  },
  {
    "Category": "Life Events",
    "Stressor": "Major transitions",
    "Description": "Moving to a new job / new city",
    "Severity": "2",
    "Likely Duration": "Weeks–Months"
  },
  {
    "Category": "Life Events",
    "Stressor": "Bereavement",
    "Description": "Death of a loved one",
    "Severity": "3",
    "Likely Duration": "Months"
  },

  {
    "Category": "Life Events",
    "Stressor": "Financial crisis",
    "Description": "Unexpected bills or debt",
    "Severity": "3",
    "Likely Duration": "Weeks–Months"
  },

  {
    "Category": "Treatment/Recovery",
    "Stressor": "Negative treatment experience",
    "Description": "Felt judged by clinician",
    "Severity": "2",
    "Likely Duration": "Weeks–Months"
  },
 {
    "Category": "Treatment/Recovery",
    "Stressor": "Successful refusal",
    "Description": "Refused a drink",
    "Severity": "3",
    "Likely Duration": "Hours–Days"
  },
 {
    "Category": "Treatment/Recovery",
    "Stressor": "Successful use of coping strategy",
    "Description": "Did not act on an urge to drink by using a coping strategy",
    "Severity": "3",
    "Likely Duration": "Hours–Days"
  },

  {
    "Category": "Emotional/Cognitive",
    "Stressor": "Cognitive impairment / mental fog",
    "Description": "Difficulty concentrating and remembering coping strategies",
    "Severity": "2",
    "Likely Duration": "Days–Weeks"
  },
  {
    "Category": "Emotional/Cognitive",
    "Stressor": "Trauma reminders",
    "Description": "Encountered a reminder of a past traumatic event",
    "Severity": "3",
    "Likely Duration": "Hours–Days"
  },

  {
    "Category": "Social/Environmental",
    "Stressor": "Living with active substance users",
    "Description": "Household member continues drinking or using substances",
    "Severity": "3",
    "Likely Duration": "Weeks–Months"
  },

]

import math

class PatientMemory:
    """A class to manage the patient's evolving state across sessions."""
    def __init__(self):
        self.craving_level = 5
        self.trigger_salience = 5
        self.motivation = 5
        self.self_efficacy = 5
        self.cognitive_control = 5
        self.habit_strength = 5
        self.lapse_flag = False
        self.stressor_ledger = [] # To track stressors between sessions

    def get_summary(self) -> str:
        """Returns a string summary of the patient's current memory state."""
        summary = (
            f"  - Craving Level: {self.craving_level}/10\n"
            f"  - Trigger Salience/Stress: {self.trigger_salience}/10\n"
            f"  - Motivation: {self.motivation}/10\n"
            f"  - Self-Efficacy/Confidence: {self.self_efficacy}/10\n"
            f"  - Cognitive Control/Adherence: {self.cognitive_control}/10\n"
            f"  - Habit Strength: {self.habit_strength}/10\n"
            f"  - Recent Lapse: {'Yes' if self.lapse_flag else 'No'}"
        )
        if self.stressor_ledger:
            summary += "\n  - Recent Stressors:\n"
            for stressor in self.stressor_ledger:
                summary += f"    - {stressor['Stressor']}: {stressor['Description']}\n"
        return summary

    def update_after_session(self):
        """Updates memory to reflect gains from a therapy session."""
        self.motivation = min(10, self.motivation + 1)
        self.self_efficacy = min(10, self.self_efficacy + 1)
        self.cognitive_control = min(10, self.cognitive_control + 1)
        self.trigger_salience = max(0, self.trigger_salience - 1)

        # Reset inter-session state
        self.stressor_ledger = []
        self.lapse_flag = False

    def _calculate_lapse_probability(self) -> float:
        """Calculates the probability of a lapse based on current memory state."""
        # Define weights for the logistic regression model
        w1, w2, w3, w4, w5, w6 = 0.3, 0.2, 0.2, 0.1, 0.1, 0.3 # Example weights

        # Calculate the weighted sum
        z = (w1 * self.craving_level +
             w2 * self.trigger_salience -
             w3 * self.self_efficacy -
             w4 * self.cognitive_control -
             w5 * self.motivation +
             w6 * self.habit_strength)

        # Apply the sigmoid function to get a probability
        probability = 1 / (1 + math.exp(-z))
        return probability

    def check_for_lapse(self):
        """Checks if a lapse occurs based on calculated probability."""
        lapse_probability = self._calculate_lapse_probability()
        if random.random() < lapse_probability:
            self.lapse_flag = True
            # A lapse temporarily reduces self-efficacy and motivation and reinforces habit strength
            self.self_efficacy = max(0, self.self_efficacy - 2)
            self.motivation = max(0, self.motivation - 2)
            self.habit_strength = min(10, self.habit_strength + 2)

    def apply_stressors(self, stressors: List[Dict[str, Any]]):
        """Applies a list of stressors to the patient's memory."""
        self.stressor_ledger.extend(stressors)
        for stressor in stressors:
            category = stressor.get("Category", "")
            if category == "Social/Environmental":
                self.trigger_salience = min(10, self.trigger_salience + 1)
                self.craving_level = min(10, self.craving_level + 1)
            elif category == "Interpersonal":
                self.trigger_salience = min(10, self.trigger_salience + 2)
            elif category == "Work/Academic":
                self.self_efficacy = max(0, self.self_efficacy - 1)
            elif category == "Emotional/Cognitive":
                self.craving_level = min(10, self.craving_level + 1)
                self.motivation = max(0, self.motivation - 1)
            elif category == "Physical/Biological":
                self.self_efficacy = max(0, self.self_efficacy - 1)
            elif category == "Life Events":
                self.trigger_salience = min(10, self.trigger_salience + 2)

        self.check_for_lapse()

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
        patient_memory: 'PatientMemory'
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
        "You are generally receptive to intervention and express willingness to follow coping plans and try alternative behaviors. "
        "You tend to respond positively to suggestions."
    ),
    "medium": (
        "You are ambivalent and show partial resistance. You may agree with "
        "some strategies, but push back on others."
    ),
    "hard": (
        "You have long-standing alcohol use, entrenched pessimism and low self-efficacy. You doubt your ability to recover "
        "and have substantial mistrust or skepticism about treatment. You often challenge or deflect "
        "suggestions, emphasize barriers, and may minimize the need for change."
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
        "behavioral patterns, barriers and motivations. This summary will be used by a therapist bot "
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
Your difficulty level description explains how resistant or ambivalent you are to therapist's suggestions. 
At the beginning of each session, report important events since last session.

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
    if "patient_memory" not in state:
        state["patient_memory"] = PatientMemory()

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
You are a licensed therapist in a role-play simulation conducting an ongoing course of therapy with a patient who has alcohol addiction. 
Your goal is to create a detailed, step-by-step conversation with a patient based on their profile summary that incorporates 
AVAILABLE STRATEGIES below.

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


def environment_agent_node(state: DialogueState) -> Dict[str, Any]:
    """
    Simulates environmental stressors affecting the patient between sessions.
    """
    # Randomly select 1 to 3 stressors to apply
    num_stressors = random.randint(1, 3)
    selected_stressors = random.sample(ENVIRONMENT_STRESSORS, num_stressors)

    patient_memory = state["patient_memory"]
    patient_memory.apply_stressors(selected_stressors)

    print(f"--- Environment Agent Applied Stressors ---")
    print(f"Patient memory state at the START of session {state['session_number']}:")
    print(patient_memory.get_summary())

    # This node only updates the patient_memory, so we return it
    return {
    **state,
    "patient_memory": patient_memory,
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

graph.set_entry_point("therapist")

graph.add_conditional_edges(
    "therapist",
    route_after_therapist,
    {
        "patient": "patient",
        END: END,
    },
)

graph.add_conditional_edges(
    "patient",
    route_after_patient,
    {
        "therapist": "therapist",
        END: END,
    },
)

app = graph.compile()

# Execution and Output
# Example Conversation Generation
# replace 'example_patient_profile' with synthesized profiles

# Example Patient Profile
example_patient_profile = f"""
The user describes themselves as very introverted and struggles with building new relationships. They display significant low self-esteem and
 self-loathing, evident in statements like 'I can't stand being myself,' 'I can't even look at myself some days,' and feeling 'pathetic, 
 embarrassing, and a disappointment.' There is a pattern of impulsivity, particularly in giving into 'just one drink' despite knowing they 
cannot moderate. They express deep shame, guilt, and disappointment, often wallowing in these emotions post-drinking. They perceive themselves 
as a 'fraud,' projecting an image of having 'their shit together' while struggling internally. They also demonstrate self-awareness of their 
self-destructive patterns and their inability to moderate.\nAlcohol Use History: The user, currently 25 years old, has been drinking excessively 
since age 15, a duration of 10 years, characterized by constant binge drinking. They were a daily drinker for almost 2 years, often drinking 
alone in their room 4-6 times a week, frequently blacking out. Prior to daily drinking, they binge drank every weekend. They have made multiple 
attempts at sobriety, with the longest recent streak being 29-30 days, but repeatedly relapse, often leading to binge drinking and blackouts 
They acknowledge an inability to moderate, stating 'it's either 0 drinks or 100 drinks'. They also mention a history of using alcohol to curb 
appetite, consistent with an eating disorder.\nSignificant Life Events: multiple instances of embarrassment at work-related events, such as 
getting blackout drunk at a Christmas party and later bawling their eyes out because they couldn't drink at another party, and a recent work lunch where they 
blacked out and needed coworkers to help them. A traumatic incident involved chipping a front tooth, puking, and being found passed out by 
their parents at a train station. They report sabotaging numerous platonic and romantic relationships due to their drinking habits and the 
desire to maintain a false image. They also mention the health impacts related to their eating disorder and alcohol use, stating 'alcohol and 
restricting almost took me there,' implying severe health risks.\nBehavioral Themes: The user exhibits patterns of extreme social isolation, 
particularly when drinking, often consuming alcohol alone in their room. They engage in high-risk behaviors while blacked out, such as leaving 
the house and ending up in unknown locations without memory. Secretive behavior includes sneaking alcohol into the house and hiding/disposing of 
empty bottles and cans. There's a clear pattern of self-neglect, including poor hygiene, neglecting personal care, and missing work due to 
hangovers/blackouts. The user struggles with boredom, which acts as a significant trigger for drinking, and relies on alcohol to cope with 
uncomfortable emotions and social anxiety. They also report severe sleep disturbances, waking up in a state of panic after drinking. There's a 
dangerous interaction between their drinking and an eating disorder, where alcohol is used to suppress appetite, leading to dangerous weight 
loss and putting themselves in perilous situations. They actively avoid places with triggers (bars, liquor stores). Relationships are sabotaged 
due to drinking and a desire to maintain a false image. They express self-punishment and struggle to break cycles of self-destructive behavior.
\nMotivations for Alcohol Use: The primary motivations for substance use appear to be coping with loneliness, emotional distress, and self-hatred 
('I drink because I can't tolerate the room and myself especially,' 'I drink alone because I can't stand being myself, I can't even look at 
myself some days'). Alcohol is also used as an escape mechanism and to numb emotions ('drown them all alcohol'). The user seeks immediate 
gratification, even knowing long-term negative consequences, and romanticizes the 'ritual' and initial excitement of drinking, chasing a past 
high. Social anxiety is a strong motivator, as they believe they are 'more likable' or 'fun' when drunk, using alcohol as 'liquid courage.' 
Boredom is identified as a significant trigger. They also mention seeking 'chaos' and a fantasy of 'living their best life' through drinking, 
despite repeated negative experiences. There is also a past motivation to use alcohol to curb appetite due to an eating disorder.",
\n"Barriers to treatment": "Emotional Reliance on Alcohol","Compulsive or Habitual Use","Disrupted Social Support","Fear of Judgment / Stigma".
"""

difficulty_setting = "hard"

# Generate a concise summary of the patient profile to save tokens
print("Summarizing patient profile...")
patient_profile_summary = summarize_patient_profile(example_patient_profile.strip())
print("Summary complete.")

# Store the data for all sessions
sessions_data = []

# Initialize Patient Memory
patient_memory = PatientMemory()

# Print initial memory state
print("--- Initial Patient Memory State (Before Session 1) ---")
print(patient_memory.get_summary())

# ✅ Apply environment stressors BETWEEN sessions
for session_number in range(1, 7):
    print(f"--- Running Session #{session_number} ---")

    initial_memory_summary = patient_memory.get_summary()
    
    if session_number > 1:
        state = {
            "session_number": session_number,
            "patient_memory": patient_memory,
        }

        state = environment_agent_node(state)

        assert initial_memory_summary != state["patient_memory"].get_summary(), (
            "Environment stressors were not applied correctly"
        )

        # Ensure we keep the mutated memory reference
        patient_memory = state["patient_memory"]

    # Invoke the graph for the current session
    result_state = app.invoke({
        "history": [],
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
        "patient_memory": patient_memory,
    }, config={"recursion_limit": 200})

    # Update patient memory with post-session gains
    patient_memory.update_after_session()

    print(f"\nPatient memory state at the END of session {session_number}:")
    print(patient_memory.get_summary())


    # Get the unique strategies used in this session
    strategies_this_session = sorted(list(set(result_state.get("strategy_history", []))))

    # Store the results for this session
    session_data = {
        "session_number": session_number,
        "session_goals": SESSION_GOALS.get(session_number, {}),
        "patient_memory_initial": initial_memory_summary,
        "dialogue": result_state["history"],
        "patient_memory_final": patient_memory.get_summary(),
        "strategies_used": strategies_this_session,
    }
    sessions_data.append(session_data)

    if strategies_this_session:
        print(f"**Strategies Used:** {', '.join(strategies_this_session)}")

    print(f"\n--- Session {session_number} Complete ---\n")

# Set output directory
output_dir = "C:\\Users\\vikto\\RecoveryBot Project"
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
    "stressors": patient_memory.get_summary()
}

# Save JSON file
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2)

print(f"Saved dialogue to {output_path}")

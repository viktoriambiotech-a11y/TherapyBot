import os
import csv
import json
import sys
from dotenv import load_dotenv
from openai import OpenAI, APIError

# ---------------------------------------------------------
# ENV + CLIENT SETUP
# ---------------------------------------------------------
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: The OPENAI_API_KEY environment variable is not set.")
    print("Please create a .env file in the script's directory and add the following line:")
    print("OPENAI_API_KEY='your_api_key_here'")
    sys.exit(1)

try:
    client = OpenAI(api_key=api_key)
except Exception as e:
    print(f"ERROR: Failed to initialize OpenAI client: {e}")
    sys.exit(1)

MODEL_PATIENT = "gpt-4o"

INPUT_FILE = r"C:\Users\vikto\RecoveryBot Project\Patient_Profiles_Nov9.csv"
OUTPUT_FILE = r"C:\Users\vikto\RecoveryBot Project\Patient_Profiles_Rated.json"

# ---------------------------------------------------------
# SYSTEM PROMPT (STRICT CLASSIFICATION LOGIC)
# ---------------------------------------------------------
SYSTEM_PROMPT = """
You are a clinical classification assistant.

Your task:
1. Identify barriers to alcohol addiction treatment using ONLY the rules provided.
2. Assign one or more barriers.
3. Determine difficulty rating using the decision rules.
4. Do NOT invent information.
5. If evidence is unclear, do NOT assign that barrier.

Use exact barrier names as provided.
"""

# ---------------------------------------------------------
# CLASSIFICATION PROMPT
# ---------------------------------------------------------
def build_prompt(profile_text):
    return f"""
PATIENT PROFILE:
{profile_text}

-----------------------------------
BARRIER CLASSIFICATION RULES
-----------------------------------

A. Individual Barriers

A1. Psychological Resistance (Denial / Ambivalence)
Assign if ANY:
- Minimizing harm
- Mixed motivation without plan
- Externalizing responsibility
Do NOT assign if problem is clearly acknowledged and actively addressed.

A2. Emotional Reliance on Alcohol
Assign if ANY:
- Alcohol used to cope with emotions
- Needed for mood regulation
- Emotional relief rather than social use

A3. Fear of Withdrawal / Fear of Sobriety
Assign if ANY:
- Fear of withdrawal
- Anxiety about life without alcohol
- Avoidance of abstinence

A4. Compulsive or Habitual Use
Assign if ANY:
- Routine-based drinking
- Automatic use
- "It just happens"

-----------------------------------
B. Social Barriers

B1. Disrupted Social Support
Assign if ANY:
- Isolation or conflict
- Unsupportive environment
- Partner/family undermines recovery

B2. Fear of Judgment / Stigma
Assign if ANY:
- Shame or fear of being judged
- Avoidance of help
- Small community privacy concerns

B3. Social Exposure to Drinking
Assign if ANY:
- Drinking peers
- Alcohol-centered social life
- Difficulty avoiding drinking settings

-----------------------------------
C. Systemic Barriers

C1. Distrust in Healthcare or Treatment
Assign if ANY:
- Negative treatment history
- Skepticism toward clinicians
- Belief treatment doesn't work

C2. Access Barriers
Assign if ANY:
- Financial barriers
- Long wait times
- Geographic/logistical issues

-----------------------------------
DIFFICULTY RATING RULES
-----------------------------------

Step 2.1: Count domains present:
Individual / Social / Systemic

Step 2.2: High-Resistance Indicators:
- Long-term heavy use
- Repeated relapse
- Strong distrust of providers
- Severe emotional dysregulation
- Alcohol as primary coping mechanism

Step 2.3: Rating Logic:

🟢 EASY:
- Barriers in 1 domain ONLY
- NO high-resistance indicators
- Clear motivation and cooperation

🟡 MEDIUM:
- Barriers in 2 domains OR
- Psychological ambivalence OR
- Habitual/emotional reliance without severe dysregulation

🔴 HARD (overrides all):
- ANY high-resistance indicator OR
- Barriers in 3 domains OR
- Severe emotional reliance + social or systemic barrier

-----------------------------------
OUTPUT FORMAT (STRICT)
-----------------------------------

Barrier List:
- [Barrier Name]
- [Barrier Name]

Difficulty Rating:
Easy / Medium / Hard
"""

# ---------------------------------------------------------
# PARSE LLM OUTPUT
# ---------------------------------------------------------
def parse_llm_output(text):
    """
    Parses the raw text output from the language model to extract
    the barrier list and difficulty rating.
    """
    barrier_list = []
    difficulty_rating = "Unknown"

    lines = text.strip().split('\n')

    in_barrier_section = False
    for line in lines:
        line = line.strip()
        if line.startswith("Barrier List:"):
            in_barrier_section = True
            continue
        elif line.startswith("Difficulty Rating:"):
            in_barrier_section = False
            difficulty_rating = line.replace("Difficulty Rating:", "").strip()
            continue

        if in_barrier_section and line.startswith("-"):
            barrier_list.append(line[1:].strip())

    return barrier_list, difficulty_rating

# ---------------------------------------------------------
# PROCESS FILES
# ---------------------------------------------------------
def get_patient_classification(profile_text):
    """
    Calls the OpenAI API to get the classification for a single patient profile.
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_PATIENT,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(profile_text)}
            ],
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    except APIError as e:
        print(f"  !! API Error during classification: {e}")
        return None
    except Exception as e:
        print(f"  !! An unexpected error occurred during API call: {e}")
        return None

def process_profiles():
    """
    Reads profiles from a CSV, gets ratings, and saves them to a single JSON file.
    """
    all_results = []
    try:
        print(f"INFO: Reading input file: {INPUT_FILE}")
        with open(INPUT_FILE, mode='r', encoding='latin-1') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                patient_id = row.get('user_id')
                profile_text = row.get('profile_text')

                if not patient_id or not profile_text:
                    print(f"WARNING: Skipping incomplete row: {row}")
                    continue

                print(f"Processing patient: {patient_id}...")

                classification_text = get_patient_classification(profile_text)

                if classification_text:
                    barrier_list, difficulty_rating = parse_llm_output(classification_text)

                    all_results.append({
                        "Patient ID": patient_id,
                        "Patient Profile Summary": profile_text,
                        "Barrier list": barrier_list,
                        "Difficulty Level": difficulty_rating
                    })
                else:
                    print(f"WARNING: Failed to classify patient {patient_id}. Skipping.")

        output_dir = os.path.dirname(OUTPUT_FILE)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
            json.dump(all_results, outfile, indent=2)

        print(f"\nSuccessfully processed {len(all_results)} profiles.")
        print(f"Output saved to {OUTPUT_FILE}")

    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{INPUT_FILE}'. Please check the path and try again.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    process_profiles()

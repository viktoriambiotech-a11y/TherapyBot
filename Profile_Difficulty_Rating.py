import os
import csv
import json
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------
# ENV + CLIENT SETUP
# ---------------------------------------------------------
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_PATIENT = "gpt-4o"

INPUT_FILE = r"C:\Users\vikto\RecoveryBot Project\Patient_Profiles_Nov9.csv"
OUTPUT_FILE = r"C:\Users\vikto\RecoveryBot Project\Patient_Profiles_Rated.json"

# Ensure the output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

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
# PROCESS FILES
# ---------------------------------------------------------
def process_profiles():
    """
    Reads profiles from a CSV, gets ratings, and saves them to a single JSON file.
    """
    all_results = []
    try:
        with open(INPUT_FILE, mode='r', encoding='latin-1') as infile:
            # Assuming the CSV has headers: 'user_id', 'profile_text'
            reader = csv.DictReader(infile)
            for row in reader:
                patient_id = row.get('user_id')
                profile_text = row.get('profile_text')

                if not patient_id or not profile_text:
                    print(f"Skipping incomplete row: {row}")
                    continue

                print(f"Processing patient: {patient_id}...")

                response = client.chat.completions.create(
                    model=MODEL_PATIENT,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_prompt(profile_text)}
                    ],
                    temperature=0.0
                )

                classification = response.choices[0].message.content.strip()

                all_results.append({
                    "patient_id": patient_id,
                    "classification": classification
                })

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
            json.dump(all_results, outfile, indent=2)

        print(f"\nSuccessfully processed {len(all_results)} profiles and saved to {OUTPUT_FILE}")

    except FileNotFoundError:
        print(f"ERROR: Input file not found at '{INPUT_FILE}'. Please check the path.")

if __name__ == "__main__":
    process_profiles()

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
1. Identify barriers to alcohol addiction treatment using the rules provided.
2. Assign one or more barriers.
3. Determine difficulty rating using the decision rules.
4. Do NOT invent information.
5. If evidence is unclear, do NOT assign that barrier.

Use the barrier names as provided.
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
    difficulty_rating = ""
    lines = text.strip().split('\n')

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("Barrier List:"):
            i += 1
            while i < len(lines) and lines[i].strip().startswith("-"):
                barrier_list.append(lines[i].strip()[1:].strip())
                i += 1
            continue # Continue to the next part of the main loop

        if line.startswith("Difficulty Level:"):
            # The rating might be on the same line or the next one
            rating = line.replace("Difficulty Level:", "").strip()
            if rating:
                difficulty_rating = rating
            elif i + 1 < len(lines) and lines[i+1].strip():
                difficulty_rating = lines[i+1].strip()
                i += 1 # Move past the rating value line
            i += 1
            continue

        i += 1

    return barrier_list, difficulty_rating or "Unknown"

# ---------------------------------------------------------
# PROCESS FILES
# ---------------------------------------------------------
def get_patient_classification(profile_text):
    """
    Calls the OpenAI API to get the classification for a single patient profile.
    """
    # The classification prompt is now constructed directly within this function
    # to avoid the NameError from the missing build_prompt function.
    user_prompt = f"""
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

🔴 HARD
Assign hard rating if ANY of the following are present:
-  High-resistance indicators (i.e. Long-term heavy use (multi-year pattern), Repeated relapse despite treatment, Strong distrust of providers, Severe emotional dysregulation (hopelessness, self-destructive coping), Alcohol used as primary coping mechanism),
OR
Barriers in 3 domains,
OR
Severe emotional reliance + social or systemic barrier
Typical profile:
“Alcohol is the only thing that helps and I don’t trust treatment.”

🟢 EASY
ALL must be true:
Barriers present in 1 domain only
AND
NO high-resistance indicators
AND
Clear motivation and willingness to cooperate
Typical profile:
“I want to stop, I know alcohol is a problem, I just need structure.”

🟡 MEDIUM
Barriers in 2 domains,
OR
Psychological ambivalence present,
OR
Habitual/emotional reliance without severe dysregulation
Typical profile:
“I know it’s a problem, but I keep slipping when stressed.”

-----------------------------------
OUTPUT FORMAT (STRICT)
-----------------------------------

Barrier List:
- [Barrier Name]
- [Barrier Name]

Difficulty Level:
Easy / Medium / Hard
"""
    try:
        response = client.chat.completions.create(
            model=MODEL_PATIENT,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
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
    This function is designed to handle CSVs where each patient's data spans multiple rows.
    """
    all_results = []
    current_patient_id = None
    current_profile_text = ""

    def process_single_patient(patient_id, profile_text):
        """Helper function to process one patient's complete profile."""
        if not patient_id or not profile_text.strip():
            print(f"WARNING: Skipping patient record with incomplete data. ID: {patient_id}")
            return

        print(f"Processing patient: {patient_id}...")
        classification_text = get_patient_classification(profile_text)

        if classification_text:
            barrier_list, difficulty_rating = parse_llm_output(classification_text)
            all_results.append({
                "Patient ID": patient_id,
                "Patient Profile Summary": profile_text.strip(),
                "Barrier list": barrier_list,
                "Difficulty Level": difficulty_rating
            })
        else:
            print(f"WARNING: Failed to classify patient {patient_id}. Skipping.")

    try:
        print(f"INFO: Reading input file: {INPUT_FILE}")
        with open(INPUT_FILE, mode='r', encoding='latin-1') as infile:
            reader = csv.reader(infile)

            for row in reader:
                if not row or not any(field.strip() for field in row):
                    continue  # Skip empty rows

                col1 = row[0].strip()
                col2 = row[1].strip() if len(row) > 1 else ""

                # A row with "User ID" and a UUID in the next column marks a new patient
                if "User ID" in col1 and len(col2) > 20:
                    # If we were already building a patient, process them first
                    if current_patient_id:
                        process_single_patient(current_patient_id, current_profile_text)

                    # Start the new patient record
                    current_patient_id = col2
                    current_profile_text = ""
                # A separator line marks the end of a patient record
                elif "---" in col1:
                    if current_patient_id:
                        process_single_patient(current_patient_id, current_profile_text)
                    # Reset for the next block
                    current_patient_id = None
                    current_profile_text = ""
                # Otherwise, it's content for the current patient
                elif current_patient_id:
                    current_profile_text += f"{col1}: {col2}\n"

            # After the loop, process the last patient in the file if they exist
            if current_patient_id:
                process_single_patient(current_patient_id, current_profile_text)

        # Write the final results to the JSON file
        output_dir = os.path.dirname(OUTPUT_FILE)
        if output_dir and not os.path.exists(output_dir):
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

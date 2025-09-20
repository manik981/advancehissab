import os
import uuid
import google.generativeai as genai
from gtts import gTTS
import json
from dotenv import load_dotenv

from ragnew import get_enhanced_prompt
from vectordbnew import (
    setup_vector_db, 
    add_user_prompt_to_db,
    setup_bad_prompts_db,
    add_to_bad_prompts_db,
    is_bad_prompts_db_empty,
    get_all_categories
)

load_dotenv()
setup_vector_db()
setup_bad_prompts_db()

# --- Prompts ---
PROMPT_PREPROCESS_CLASSIFY = """
You are a 2-step financial query processor. Your tasks are:
1.  Convert the user's pure Hindi query into natural, everyday Hinglish.
2.  Classify the query into one of the given categories.

You MUST return a single, valid JSON object with two keys: "hinglish_text" and "category".

Available Categories: {categories}

User's Hindi Query: "{hindi_text}"

JSON Output:
"""

PROMPT_SUMMARY = """
Analyze the final result from the following detailed text. Create a single, concise summary sentence in Hindi
that is perfect for a voice assistant.
Example:
Detailed Text: "Trip ka Hisaab: ... Ravi ko aapko ₹200 aur dene hain."
Your Summary: "Hisaab ke anusaar, Ravi ko aapko ₹200 aur dene hain."
"""

PROMPT_ERROR_ANALYSIS = """
You are a quality control analyst. A user has marked the following interaction as "Bad". 
Analyze the user's query and the assistant's response to identify the most likely mistake.
Provide a very short, simple, one-line explanation of the error in Hindi.
User's Query: "{user_story}"
Assistant's incorrect Response: "{model_response}"
Mistake Analysis (in Hindi):
"""

# --- OPTIMIZATION: Combined Pre-processing and Classification ---
def preprocess_and_classify(api_key: str, hindi_text: str) -> (str, str):
    """
    Performs Hindi-to-Hinglish conversion and category classification in a single LLM call.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        categories = get_all_categories()
        prompt = PROMPT_PREPROCESS_CLASSIFY.format(categories=categories, hindi_text=hindi_text)
        
        response = model.generate_content(prompt)
        
        # Clean up and parse the JSON response
        json_response_str = response.text.strip().replace("```json", "").replace("```", "")
        data = json.loads(json_response_str)
        
        hinglish_text = data.get("hinglish_text", hindi_text)
        category = data.get("category", "personal_expense_tracking")
        
        print(f"Pre-processing successful. Hinglish: '{hinglish_text}', Category: '{category}'")
        return hinglish_text, category
        
    except Exception as e:
        print(f"Pre-processing failed: {e}. Falling back.")
        return hindi_text, "personal_expense_tracking" # Fallback on error

# --- Core Processing Logic ---
def process_query_stream(api_key: str, hinglish_user_story: str, category: str):
    if not api_key: yield "❌ Error: Google API Key missing."; return
    if not hinglish_user_story: yield "⚠️ Kripya apni kahani likhein ya bolein."; return

    try:
        print("Step 2: Generating enhanced prompt...")
        enhanced_prompt = get_enhanced_prompt(hinglish_user_story, category)

        print("Step 3: Calling main LLM for calculation...")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response_stream = model.generate_content(enhanced_prompt, stream=True)

        for chunk in response_stream:
            if hasattr(chunk, "text") and chunk.text: yield chunk.text

    except Exception as e:
        yield f"⚠️ Hisaab lagate samay error aaya: {e}"

# --- Feedback Handling ---
def save_good_prompt(hinglish_story: str, category: str):
    print(f"Saving to good DB: {hinglish_story}")
    add_user_prompt_to_db(hinglish_story, category)

def save_bad_prompt(hinglish_story: str, model_response: str):
    print(f"Saving to bad DB: {hinglish_story}")
    add_to_bad_prompts_db(hinglish_story, model_response)

def analyze_bad_response(api_key: str, user_story: str, model_response: str) -> str:
    if is_bad_prompts_db_empty(): return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro') 
        prompt = PROMPT_ERROR_ANALYSIS.format(user_story=user_story, model_response=model_response)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error analysis failed: {e}")
        return None

# --- Audio Generation (Unchanged) ---
def generate_audio_summary(api_key: str, detailed_text: str, error_analysis: str = None, slow: bool = False, lang: str = "hi"):
    if not api_key: return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        summary_request = PROMPT_SUMMARY + f"\nDetailed Text: \"{detailed_text}\""
        response = model.generate_content(summary_request)
        summary_text = response.text.strip() if response and hasattr(response, "text") else "Hisaab taiyaar hai."
        
        final_audio_text = f"Galti ka vishleshan: {error_analysis}. {summary_text}" if error_analysis else summary_text
        
        audio_file_path = f"response_{uuid.uuid4().hex}.mp3"
        tts = gTTS(text=final_audio_text, lang=lang, slow=slow)
        tts.save(audio_file_path)
        cleanup_old_audio_files(keep=3)
        return audio_file_path
    except Exception as e:
        print(f"Audio summary error: {e}")
        return None

def cleanup_old_audio_files(keep=3):
    try:
        files = [f for f in os.listdir('.') if f.startswith("response_") and f.endswith(".mp3")]
        files.sort(key=os.path.getmtime, reverse=True)
        for f in files[keep:]: os.remove(f)
    except Exception as e:
        print(f"Purani audio files delete karte samay error aaya: {e}")



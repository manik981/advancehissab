import os
import uuid
import google.generativeai as genai
from gtts import gTTS
from dotenv import load_dotenv

# Import functions from RAG and VectorDB modules
# Note: In files ko bhi update karna hoga, khaas kar vectordb.py ko.
from ragnew import get_enhanced_prompt
from vectordbnew import (
    setup_vector_db, 
    add_user_prompt_to_db,
    setup_bad_prompts_db,
    add_to_bad_prompts_db,
    is_bad_prompts_db_empty
)

# Load environment variables
load_dotenv()

# Dono vector databases ko setup karein
setup_vector_db()
setup_bad_prompts_db()

# --- Prompts ---
PROMPT_HINDI_TO_HINGLISH = """
You are a language conversion expert. Convert the following pure Hindi sentence into natural, everyday Hinglish (Hindi written in the Roman script). The meaning and financial context must be perfectly preserved.

Hindi: "{hindi_text}"
Hinglish:
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

# --- Helper Functions for New Logic ---

def convert_hindi_to_hinglish(api_key: str, hindi_text: str) -> str:
    """
    Uses an LLM to convert pure Hindi text to Hinglish.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = PROMPT_HINDI_TO_HINGLISH.format(hindi_text=hindi_text)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Hinglish conversion error: {e}")
        return hindi_text # Fallback to original text on error

def analyze_bad_response(api_key: str, user_story: str, model_response: str) -> str:
    """
    Uses a second, more powerful LLM to analyze why a response was bad.
    """
    if is_bad_prompts_db_empty():
        return None # Agar bad prompts DB khali hai to analysis na karein

    try:
        # Aap yahan ek alag, powerful model (jaise Gemini Pro) istemal kar sakte hain
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro') 
        prompt = PROMPT_ERROR_ANALYSIS.format(user_story=user_story, model_response=model_response)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Error analysis failed: {e}")
        return None

# --- Core Processing Logic ---

def process_query_stream(api_key: str, hindi_user_story: str):
    """
    The main processing pipeline for a user query.
    """
    if not api_key:
        yield "❌ Error: Google API Key missing."
        return
    if not hindi_user_story:
        yield "⚠️ Kripya apni kahani likhein ya bolein."
        return

    try:
        # Step 1: Hindi user story ko Hinglish mein convert karein
        print("Step 1: Converting Hindi to Hinglish...")
        hinglish_user_story = convert_hindi_to_hinglish(api_key, hindi_user_story)
        print(f"Hinglish version: {hinglish_user_story}")

        # Step 2: Enhanced prompt generate karein (jisme category classification aur random example selection hoga)
        # Note: Iske liye rag.py aur vectordb.py mein changes zaroori hain
        print("Step 2: Generating enhanced prompt...")
        enhanced_prompt = get_enhanced_prompt(hinglish_user_story)

        # Step 3: Main LLM (Gemini Flash) ko call karein
        print("Step 3: Calling main LLM for calculation...")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response_stream = model.generate_content(enhanced_prompt, stream=True)

        for chunk in response_stream:
            if hasattr(chunk, "text") and chunk.text:
                yield chunk.text

    except Exception as e:
        yield f"⚠️ Hisaab lagate samay error aaya: {e}"

# --- Feedback Handling Functions (Called from app.py) ---

def save_good_prompt(hinglish_story: str):
    """Saves the prompt to the primary vector DB."""
    print(f"Saving to good DB: {hinglish_story}")
    add_user_prompt_to_db(hinglish_story)

def save_bad_prompt(hinglish_story: str, model_response: str):
    """Saves the prompt and response to the 'bad prompts' vector DB."""
    print(f"Saving to bad DB: {hinglish_story}")
    add_to_bad_prompts_db(hinglish_story, model_response)

# --- Audio Generation Logic ---

def generate_audio_summary(api_key: str, detailed_text: str, error_analysis: str = None, slow: bool = False, lang: str = "hi"):
    """
    Generates audio summary, now with optional error analysis.
    """
    if not api_key:
        return None
    try:
        # Step 1: Pehle normal summary generate karein
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        summary_request = PROMPT_SUMMARY + f"\nDetailed Text: \"{detailed_text}\""
        response = model.generate_content(summary_request)
        summary_text = response.text.strip() if response and hasattr(response, "text") else "Hisaab taiyaar hai."

        # Step 2: Agar error analysis hai, to use summary mein jodein
        final_audio_text = summary_text
        if error_analysis:
            final_audio_text = f"Galti ka vishleshan: {error_analysis}. {summary_text}"

        # Step 3: Audio file generate karein
        audio_file_path = f"response_{uuid.uuid4().hex}.mp3"
        tts = gTTS(text=final_audio_text, lang=lang, slow=slow)
        tts.save(audio_file_path)
        cleanup_old_audio_files(keep=3)
        return audio_file_path
    except Exception as e:
        print(f"Audio summary error: {e}")
        return None

def cleanup_old_audio_files(keep=3):
    """Deletes older audio files."""
    try:
        files = [f for f in os.listdir('.') if f.startswith("response_") and f.endswith(".mp3")]
        files.sort(key=os.path.getmtime, reverse=True)
        for f in files[keep:]:
            os.remove(f)
    except Exception as e:

        print(f"Purani audio files delete karte samay error aaya: {e}")

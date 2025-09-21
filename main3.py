import os
import uuid
import google.generativeai as genai
from gtts import gTTS
import json
from dotenv import load_dotenv

from ragnew import get_enhanced_prompt
from vectordbnew import (
    setup_vector_db, add_user_prompt_to_db,
    setup_bad_prompts_db, add_to_bad_prompts_db,
    get_all_categories, find_semantic_categories, find_random_examples_from_category
)

load_dotenv()
setup_vector_db()
setup_bad_prompts_db()

# --- Prompts ---
PROMPT_PREPROCESS_CLASSIFY = """
You MUST return a single, valid JSON object with "hinglish_text" and "category".
Available Categories: {categories}
User's Hindi Query: "{hindi_text}"
JSON Output:
"""
PROMPT_SUMMARY = "Create a single, concise summary sentence in Hindi for this text: \"{detailed_text}\""
PROMPT_ERROR_ANALYSIS = "User marked this as 'Bad'. Find the mistake in the response. Query: '{user_story}', Response: '{model_response}'. Analysis (in Hindi):"

# --- Main Processing Pipeline ---
def process_query_stream(api_key: str, hindi_user_story: str) -> (str, dict):
    """The main processing pipeline. Returns the response stream and a context dictionary."""
    genai.configure(api_key=api_key)
    context = {"user_hindi_query": hindi_user_story}

    try:
        # Step 1: Pre-process & Initial Classification
        model_flash = genai.GenerativeModel('gemini-1.5-flash')
        categories = get_all_categories()
        prompt = PROMPT_PREPROCESS_CLASSIFY.format(categories=categories, hindi_text=hindi_user_story)
        response = model_flash.generate_content(prompt)
        json_data = json.loads(response.text.strip().replace("```json", "").replace("```", ""))
        
        hinglish_story = json_data.get("hinglish_text", hindi_user_story)
        primary_category = json_data.get("category", "unknown")
        context.update({"hinglish_story": hinglish_story, "primary_category": primary_category})
        print(f"✅ Initial Classification Complete. Category: {primary_category}")

        # Step 2: Semantic Category Search
        semantic_categories = find_semantic_categories(hinglish_story, top_k=2)
        context["semantic_categories"] = semantic_categories
        print(f"✅ Semantic Search Complete. Top 2: {semantic_categories}")

        # Step 3: Random Example Retrieval (Updated to preserve category structure)
        examples_by_category = {}
        for cat in semantic_categories:
            retrieved_examples = find_random_examples_from_category(cat, max_examples=5, min_examples=1)
            if retrieved_examples:
                examples_by_category[cat] = retrieved_examples
        context["retrieved_examples"] = examples_by_category
        print(f"✅ Retrieved examples for categories: {list(examples_by_category.keys())}")


        # Step 4: Generate the Final Prompt
        enhanced_prompt = get_enhanced_prompt(hinglish_story, primary_category, semantic_categories, examples_by_category)
        
        # Step 5: Final Calculation
        response_stream = model_flash.generate_content(enhanced_prompt, stream=True)
        
        # 'yield from' response stream and return context at the end
        yield from response_stream
        return context

    except Exception as e:
        yield f"⚠️ Hisaab lagate samay error aaya: {e}"
        return context

# --- Feedback Handling ---
def save_good_prompt(context: dict, model_response: str):
    add_user_prompt_to_db(
        hinglish_prompt=context.get("hinglish_story"),
        model_response=model_response,
        primary_category=context.get("primary_category")
    )

def save_bad_prompt(context: dict, model_response: str):
    log_data = {**context, "model_response": model_response}
    add_to_bad_prompts_db(log_data=log_data)

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




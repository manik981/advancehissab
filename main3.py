import os
import uuid
import google.generativeai as genai
from groq import Groq # Groq ko import karein
from gtts import gTTS
import json
from dotenv import load_dotenv

from ragnew import get_enhanced_prompt
from vectordbnew import (
    setup_vector_db, add_user_prompt_to_db,
    setup_bad_prompts_db, add_to_bad_prompts_db,
    is_bad_prompts_db_empty, get_all_categories, find_semantic_categories, 
    find_random_examples_from_category
)

load_dotenv()
setup_vector_db()
setup_bad_prompts_db()

# --- NAYA CHANGE: Dono API Clients ko initialize karein ---
try:
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    genai.configure(api_key=gemini_api_key)
    groq_client = Groq(api_key=groq_api_key)
    print("✅ Gemini and Groq clients initialized successfully.")
except Exception as e:
    print(f"API Key initialization error: {e}")
    groq_client = None # Error hone par client ko None set karein

# --- Prompts (No changes here) ---
PROMPT_PREPROCESS_CLASSIFY = """
You MUST return a single, valid JSON object with "hinglish_text" and "category".
Available Categories: {categories}
User's Hindi Query: "{hindi_text}"
JSON Output:
"""
PROMPT_SUMMARY = "Create a single, concise summary sentence in Hindi for this text: \"{detailed_text}\""
PROMPT_ERROR_ANALYSIS = "User marked this as 'Bad'. Find the mistake in the response. Query: '{user_story}', Response: '{model_response}'. Analysis (in Hindi):"

# --- Main Processing Pipeline ---
def process_query_stream(hindi_user_story: str):
    """The main processing pipeline that orchestrates calls to different LLMs."""
    context = {"user_hindi_query": hindi_user_story}

    try:
        # --- NAYA CHANGE: Step 1 (Pre-processing) ab Groq ka istemal karega ---
        print("Step 1: Pre-processing with Groq...")
        categories = get_all_categories()
        prompt = PROMPT_PREPROCESS_CLASSIFY.format(categories=categories, hindi_text=hindi_user_story)
        
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192", # Groq ka fast model
        )
        json_data = json.loads(chat_completion.choices[0].message.content.strip().replace("```json", "").replace("```", ""))
        
        hinglish_story = json_data.get("hinglish_text", hindi_user_story)
        primary_category = json_data.get("category", "unknown")
        context.update({"hinglish_story": hinglish_story, "primary_category": primary_category})
        print(f"✅ Groq Pre-processing Complete. Category: {primary_category}")

        # Step 2: Semantic Category Search (No LLM call)
        semantic_categories = find_semantic_categories(hinglish_story, top_k=2)
        context["semantic_categories"] = semantic_categories
        print(f"✅ Semantic Search Complete. Top 2: {semantic_categories}")

        # Step 3: Random Example Retrieval (No LLM call)
        examples_by_category = {}
        for cat in semantic_categories:
            examples = find_random_examples_from_category(cat, max_examples=5, min_examples=1)
            if examples: examples_by_category[cat] = examples
        context["retrieved_examples"] = examples_by_category
        print(f"✅ Retrieved {sum(len(v) for v in examples_by_category.values())} examples.")

        # Step 4: Generate the Final Prompt (No LLM call)
        enhanced_prompt = get_enhanced_prompt(hinglish_story, primary_category, semantic_categories, examples_by_category)
        
        # --- NAYA CHANGE: Step 5 (Final Calculation) Gemini ka istemal karega ---
        print("Step 5: Final calculation with Gemini...")
        model_flash = genai.GenerativeModel('gemini-1.5-flash')
        response_stream = model_flash.generate_content(enhanced_prompt, stream=True)
        
        yield from response_stream
        return context

    except Exception as e:
        yield f"⚠️ Hisaab lagate samay error aaya: {e}"
        return context

# --- Feedback Handling (No changes in logic) ---
def save_good_prompt(context: dict, model_response: str):
    add_user_prompt_to_db(
        hinglish_prompt=context.get("hinglish_story"),
        model_response=model_response,
        primary_category=context.get("primary_category")
    )

def save_bad_prompt(context: dict, model_response: str):
    log_data = {**context, "model_response": model_response}
    add_to_bad_prompts_db(log_data=log_data)

# --- NAYA CHANGE: Error Analysis Gemini Pro ka istemal karega ---
def analyze_bad_response(context: dict, model_response: str) -> str:
    try:
        model_pro = genai.GenerativeModel('gemini-pro')
        prompt = PROMPT_ERROR_ANALYSIS.format(user_story=context.get("hinglish_story"), model_response=model_response)
        response = model_pro.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini Pro error analysis failed: {e}")
        return None

# --- NAYA CHANGE: Audio Summary Groq ka istemal karega ---
def generate_audio_summary(detailed_text: str, error_analysis: str = None):
    try:
        summary_request = PROMPT_SUMMARY.format(detailed_text=detailed_text)
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": summary_request}],
            model="llama3-8b-8192",
        )
        summary_text = chat_completion.choices[0].message.content.strip()

        final_audio_text = f"Galti ka vishleshan: {error_analysis}. {summary_text}" if error_analysis else summary_text
        
        audio_file_path = f"response_{uuid.uuid4().hex}.mp3"
        tts = gTTS(text=final_audio_text, lang="hi", slow=False)
        tts.save(audio_file_path)
        cleanup_old_audio_files(keep=3)
        return audio_file_path
    except Exception as e:
        print(f"Groq audio summary/gTTS error: {e}")
        return None

def cleanup_old_audio_files(keep=3):
    try:
        files = [f for f in os.listdir('.') if f.startswith("response_") and f.endswith(".mp3")]
        files.sort(key=os.path.getmtime, reverse=True)
        for f in files[keep:]: os.remove(f)
    except Exception as e:
        print(f"Purani audio files delete karte samay error aaya: {e}")





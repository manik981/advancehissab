import streamlit as st
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
import os
import io
from pydub import AudioSegment

# Naye advanced logic wali main file ko import karein
import main3 as main

# ---------------------------
# Streamlit App UI Setup
# ---------------------------
st.set_page_config(page_title="💰 Hissab Assistant", layout="centered")
st.title("💰 Hissab Assistant (Optimized & Fixed)")
st.caption("AI-powered financial calculator with a self-improving feedback loop.")

# --- Session State Initialization ---
if 'hindi_story' not in st.session_state:
    st.session_state.hindi_story = ""
if 'hinglish_story' not in st.session_state:
    st.session_state.hinglish_story = ""
if 'category' not in st.session_state:
    st.session_state.category = ""
if 'detailed_text' not in st.session_state:
    st.session_state.detailed_text = ""
if 'feedback_given' not in st.session_state:
    st.session_state.feedback_given = False
if 'error_analysis' not in st.session_state:
    st.session_state.error_analysis = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# --- Feedback Callback Functions ---
def handle_good_feedback():
    # 'context' object ki ab zaroorat nahi, seedhe state se values bhejenge
    main.save_good_prompt(st.session_state.hinglish_story, st.session_state.category, st.session_state.detailed_text)
    st.toast("✅ Shukriya! Isse system aur behtar hoga.")
    st.session_state.feedback_given = True

def handle_bad_feedback():
    api_key = os.getenv("GOOGLE_API_KEY")
    # Structured logging ke liye zaroori jaankari pass karein
    context = {
        "user_hindi_query": st.session_state.hindi_story,
        "hinglish_story": st.session_state.hinglish_story,
        "primary_category": st.session_state.category
    }
    main.save_bad_prompt(context, st.session_state.detailed_text)
    with st.spinner("Galti ka vishleshan kiya ja raha hai..."):
        analysis = main.analyze_bad_response(api_key, context, st.session_state.detailed_text)
        st.session_state.error_analysis = analysis
    st.toast(f"📝 Galti: {analysis}" if analysis else "📝 Feedback ke liye shukriya.")
    st.session_state.feedback_given = True

# --- Input Section (Audio code is unchanged) ---
mode = st.radio("Aap input kaise dena chahte hain:", ["🎤 Voice", "⌨️ Text"], horizontal=True)
user_story_input = None

if mode == "🎤 Voice":
    st.write("Niche diye gaye button par click karke apni aawaz record karein.")
    audio_info = mic_recorder(start_prompt="▶️ Record", stop_prompt="⏹️ Stop", key='recorder')
    if audio_info and audio_info['bytes']:
        st.info("Audio record ho gaya hai. Ab process kiya ja raha hai...")
        st.audio(audio_info['bytes'])
        recognizer = sr.Recognizer()
        converted_audio_path = "audio_converted.wav"
        try:
            sound = AudioSegment.from_file(io.BytesIO(audio_info['bytes']))
            sound.export(converted_audio_path, format="wav")
            with sr.AudioFile(converted_audio_path) as source:
                audio_data = recognizer.record(source)
            recognized_text = recognizer.recognize_google(audio_data, language='hi-IN')
            st.success(f"📝 Aapne kaha: {recognized_text}")
            user_story_input = recognized_text
        except Exception as e:
            st.error(f"Audio process karte samay error aaya: {e}")
        finally:
            if os.path.exists(converted_audio_path):
                os.remove(converted_audio_path)
else:
    user_story_input = st.text_area("Apni kahani yahan likhiye:", placeholder="Example: Mere paas 500 rupaye the...")

# --- Core Logic: Optimized Processing (FIXED) ---
if user_story_input and user_story_input != st.session_state.hindi_story:
    # Naye input aane par state reset karein
    st.session_state.hindi_story = user_story_input
    st.session_state.feedback_given = False
    st.session_state.detailed_text = ""
    st.session_state.error_analysis = None
    st.session_state.processing_complete = True # Processing shuru karein
    
    # Pre-processing yahin par karein
    with st.spinner("Input ko samajha ja raha hai... (Step 1/2)"):
        api_key = os.getenv("GOOGLE_API_KEY")
        # Groq (ya jo bhi model) se pehle pre-process karein
        hinglish_text, category = main.preprocess_and_classify(api_key, user_story_input)
        st.session_state.hinglish_story = hinglish_text
        st.session_state.category = category
    
    # Calculation shuru karein
    with st.spinner('Hisaab lagaya ja raha hai... (Step 2/2)'):
        # --- YEH HAI CORRECTED FUNCTION CALL ---
        response_generator = main.process_query_stream(api_key, st.session_state.hinglish_story, st.session_state.category)
        st.session_state.detailed_text = "".join(list(response_generator))

    st.rerun() # UI ko final result ke saath refresh karein

# --- Display Section ---
if st.session_state.processing_complete:
    st.divider()
    st.subheader("📊 Aapka Detailed Hisaab")
    
    with st.chat_message("user"):
        st.write(st.session_state.hindi_story)

    with st.chat_message("assistant"):
        st.write(st.session_state.detailed_text)
    
    if st.session_state.detailed_text and not st.session_state.feedback_given:
        st.write("Kya yeh jawab sahi tha?")
        col1, col2, _ = st.columns([1, 1, 3])
        col1.button("👍 Good", on_click=handle_good_feedback, use_container_width=True)
        col2.button("👎 Bad", on_click=handle_bad_feedback, use_container_width=True)

    if st.session_state.detailed_text:
        st.divider()
        st.subheader("🔊 Audio Summary")
        api_key = os.getenv("GOOGLE_API_KEY")
        with st.spinner('Audio summary banaya ja raha hai...'):
            audio_file = main.generate_audio_summary(
                api_key, 
                st.session_state.detailed_text,
                error_analysis=st.session_state.error_analysis
            )
            if audio_file and os.path.exists(audio_file):
                st.audio(audio_file, format="audio/mp3")
            else:
                st.warning("Audio summary generate nahi ho paya.")


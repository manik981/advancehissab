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
st.title("💰 Hissab Assistant (Hybrid RAG Version)")
st.caption("AI calculator with semantic search and self-improving feedback.")

# --- Session State Initialization ---
# 'context' dictionary ab saari zaroori jaankari store karegi
if 'context' not in st.session_state:
    st.session_state.context = {}
if 'detailed_text' not in st.session_state:
    st.session_state.detailed_text = ""
if 'feedback_given' not in st.session_state:
    st.session_state.feedback_given = False
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# --- Feedback Callback Functions (Updated to use context) ---
def handle_good_feedback():
    # 'main3.py' ko poora context dictionary pass karein
    main.save_good_prompt(st.session_state.context, st.session_state.detailed_text)
    st.toast("✅ Shukriya! Is example se system aur behtar hoga.")
    st.session_state.feedback_given = True

def handle_bad_feedback():
    api_key = os.getenv("GOOGLE_API_KEY")
    # 'main3.py' ko poora context dictionary pass karein for structured logging
    main.save_bad_prompt(st.session_state.context, st.session_state.detailed_text)
    
    with st.spinner("Galti ka vishleshan kiya ja raha hai..."):
        analysis = main.analyze_bad_response(api_key, st.session_state.context, st.session_state.detailed_text)
        # Context mein error analysis store karein
        st.session_state.context['error_analysis'] = analysis
    
    st.toast(f"📝 Galti: {analysis}" if analysis else "📝 Feedback ke liye shukriya.")
    st.session_state.feedback_given = True

# --- Input Section (Audio code unchanged) ---
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

# --- Core Logic: Processing and State Management ---
if user_story_input and user_story_input != st.session_state.context.get("user_hindi_query"):
    st.session_state.feedback_given = False
    st.session_state.detailed_text = ""
    st.session_state.processing_complete = False
    
    with st.spinner("Hisaab lagaya ja raha hai... (Multi-step process)"):
        api_key = os.getenv("GOOGLE_API_KEY")
        
        # Generator se response stream karein
        response_generator = main.process_query_stream(api_key, user_story_input)
        
        # Pehle response ko stream karke detailed_text mein store karein
        st.session_state.detailed_text = st.write_stream(response_generator)
        
        # Aakhir mein, generator se context object ko capture karein
        # Iske liye main3.py mein 'yield from' ki jagah 'return' ka istemal kiya gaya hai
        st.session_state.context = response_generator.gi_return
    
    st.session_state.processing_complete = True

# --- Display Section ---
if st.session_state.processing_complete:
    st.divider()
    st.subheader("📊 Aapka Detailed Hisaab")
    
    with st.chat_message("user"):
        st.write(st.session_state.context.get("user_hindi_query", ""))

    with st.chat_message("assistant"):
        st.write(st.session_state.detailed_text)
    
    # --- Feedback Buttons ---
    if st.session_state.detailed_text and not st.session_state.feedback_given:
        st.write("Kya yeh jawab sahi tha?")
        col1, col2, _ = st.columns([1, 1, 3])
        col1.button("👍 Good", on_click=handle_good_feedback, use_container_width=True)
        col2.button("👎 Bad", on_click=handle_bad_feedback, use_container_width=True)

    # --- Audio Summary ---
    if st.session_state.detailed_text:
        st.divider()
        st.subheader("🔊 Audio Summary")
        api_key = os.getenv("GOOGLE_API_KEY")
        with st.spinner('Audio summary banaya ja raha hai...'):
            audio_file = main.generate_audio_summary(
                api_key, 
                st.session_state.detailed_text,
                error_analysis=st.session_state.context.get("error_analysis")
            )
            if audio_file and os.path.exists(audio_file):
                st.audio(audio_file, format="audio/mp3")
            else:
                st.warning("Audio summary generate nahi ho paya.")


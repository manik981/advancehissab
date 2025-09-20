import streamlit as st
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder
import os
import io
from pydub import AudioSegment

# Naye advanced logic wali main file ko import karein
import main2 as main

# ---------------------------
# Streamlit App UI Setup
# ---------------------------
st.set_page_config(page_title="💰 Hissab Assistant", layout="centered")

st.title("💰 Hissab Assistant (Advanced Version)")
st.caption("AI-powered financial calculator with a self-improving feedback loop.")

# --- Session State Initialization ---
# Isse app user ke input aur model ke response ko yaad rakhega
if 'response_generated' not in st.session_state:
    st.session_state.response_generated = False
if 'hindi_story' not in st.session_state:
    st.session_state.hindi_story = ""
if 'hinglish_story' not in st.session_state: # Hinglish version ko store karne ke liye
    st.session_state.hinglish_story = ""
if 'detailed_text' not in st.session_state:
    st.session_state.detailed_text = ""
if 'feedback_given' not in st.session_state:
    st.session_state.feedback_given = False
if 'error_analysis' not in st.session_state:
    st.session_state.error_analysis = None

# --- Feedback Callback Functions ---
def handle_good_feedback():
    """'Good' button click karne par yeh function chalega."""
    main.save_good_prompt(st.session_state.hinglish_story)
    st.toast("✅ Shukriya! Isse system aur behtar hoga.")
    st.session_state.feedback_given = True

def handle_bad_feedback():
    """'Bad' button click karne par yeh function chalega."""
    api_key = os.getenv("GOOGLE_API_KEY")
    main.save_bad_prompt(st.session_state.hinglish_story, st.session_state.detailed_text)
    
    # Galti ka vishleshan karne ke liye LLM ko call karein
    with st.spinner("Galti ka vishleshan kiya ja raha hai..."):
        analysis = main.analyze_bad_response(api_key, st.session_state.hinglish_story, st.session_state.detailed_text)
        st.session_state.error_analysis = analysis
    
    if analysis:
        st.toast(f"📝 Galti: {analysis}")
    else:
        st.toast("📝 Feedback ke liye shukriya.")
    st.session_state.feedback_given = True

# --- Input Section ---
mode = st.radio("Aap input kaise dena chahte hain:", ["🎤 Voice", "⌨️ Text"], horizontal=True)
user_story_input = None

if mode == "🎤 Voice":
    st.write("Niche diye gaye button par click karke apni aawaz record karein.")
    audio_info = mic_recorder(start_prompt="▶️ Record", stop_prompt="⏹️ Stop", key='recorder')
    
    if audio_info and audio_info['bytes']:
        st.info("Audio record ho gaya hai. Ab process kiya ja raha hai...")
        st.audio(audio_info['bytes']) # Original audio play karein
        
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

# ---------------------------
# Core Logic: Process Query and Display Results
# ---------------------------
if user_story_input:
    # Naye input aane par state reset karein
    st.session_state.hindi_story = user_story_input
    st.session_state.response_generated = True
    st.session_state.feedback_given = False
    st.session_state.detailed_text = ""
    st.session_state.error_analysis = None
    st.rerun() # Rerun to start processing immediately

if st.session_state.response_generated:
    st.divider()
    st.subheader("📊 Aapka Detailed Hisaab")
    
    # User ka sawaal dikhayein
    with st.chat_message("user"):
        st.write(st.session_state.hindi_story)

    # Model ka jawab dikhayein
    with st.chat_message("assistant"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            st.error("❌ GOOGLE_API_KEY set nahi hai.")
        else:
            # Response ko sirf ek baar generate karein
            if not st.session_state.detailed_text:
                with st.spinner('Smart RAG system hisaab laga raha hai...'):
                    # Puraane 'main' se 'main2' (naye) logic ko call karein
                    response_generator = main.process_query_stream(api_key, st.session_state.hindi_story)
                    st.session_state.detailed_text = st.write_stream(response_generator)
                    # Hinglish version ko save karein (main2.py se milega)
                    st.session_state.hinglish_story = main.convert_hindi_to_hinglish(api_key, st.session_state.hindi_story)
            else:
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
        with st.spinner('Audio summary banaya ja raha hai...'):
            audio_file = main.generate_audio_summary(
                api_key, 
                st.session_state.detailed_text,
                error_analysis=st.session_state.error_analysis # Error analysis pass karein
            )
            if audio_file and os.path.exists(audio_file):
                st.audio(audio_file, format="audio/mp3")
            else:
                st.warning("Audio summary generate nahi ho paya.")


import streamlit as st
import psutil

# -------------------------------
# Manage Station state variables
# -------------------------------

def initialize_session_state():
    defaults = {
        "audio_path": None,
        "audio_ready": False,
        "transcription": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# 🔍 Show memory info after 
def display_memory_once():
    if 'memory_logged' not in st.session_state:
        mem = psutil.virtual_memory()
        st.markdown(f"**Memory Used:** {mem.percent}%")
        st.session_state.memory_logged = True

# Reset the app
def reset_session_state_except_model():
    keys_to_keep = {"classifier", "whisper"}
    for key in list(st.session_state.keys()):
        if key not in keys_to_keep:
            del st.session_state[key]

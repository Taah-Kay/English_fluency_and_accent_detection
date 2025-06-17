import os
import streamlit as st
from speechbrain.pretrained.interfaces import foreign_class
from faster_whisper import WhisperModel


# -------------------------------
# Load Model (Cached)
# -------------------------------
@st.cache_resource(show_spinner="Loading model...") # making sure we only load the model once per every app instance
def load_accent_model():
    """Loads custom accent classification model."""
    if not os.getenv("HF_TOKEN"):
        st.error("Hugging Face token not found.")
        st.stop()
    try:
        return foreign_class(
            source="Jzuluaga/accent-id-commonaccent_xlsr-en-english",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier"
        )
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

@st.cache_resource(show_spinner="Loading Whisper...")
def load_whisper():
    return WhisperModel("tiny", device="cpu", compute_type="int8_float32")

import streamlit as st
import tempfile
import shutil
import psutil
import torch
import torchaudio

from utils.audio_processing import trim_audio, download_audio_as_wav
from utils.video_processing import trim_video
from utils.model_loader import load_accent_model, load_whisper
from utils.accent_analysis import analyze_accent
from utils.session_utils import initialize_session_state, display_memory_once, reset_session_state_except_model

st.title("🎙️ English Accent Audio Detector")

# Initialize session state
initialize_session_state()

# Load models once
if 'classifier' not in st.session_state:
    st.session_state.classifier = load_accent_model()
if 'whisper' not in st.session_state:
    st.session_state.whisper = load_whisper()

# Memory info
display_memory_once()

# Reset state for a new analysis
if st.button("🔄 Analyze new video"):
    reset_session_state_except_model()
    st.rerun()

# Check for ffmpeg
if not shutil.which("ffmpeg"):
    raise EnvironmentError("FFmpeg not found. Please install or add it to PATH.")

# Input options
option = st.radio("Choose input method:", ["Upload video file", "Enter Video Url"])

if option == "Upload video file":
    uploaded_video = st.file_uploader("Upload your video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_video is not None:
        temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        with open(temp_video_path.name, "wb") as f:
            f.write(uploaded_video.read())
        audio_path = trim_video(temp_video_path.name)
        st.success("✅ Video uploaded successfully.")
        st.session_state.audio_path = audio_path

elif option == "Enter Video Url":
    yt_url = st.text_input("Paste YouTube URL")
    if st.button("Download Video"):
        with st.spinner("Downloading video..."):
            audio_path = download_audio_as_wav(yt_url)
            audio_path = trim_audio(audio_path)
        if audio_path:
            st.success("✅ Video downloaded successfully.")
            st.session_state.audio_path = audio_path

# Transcription and Accent Analysis
if st.session_state.audio_path and not st.session_state.transcription:
    if st.button("🎧 Extract Audio"):
        st.session_state.audio_ready = True
        st.audio(st.session_state.audio_path, format='audio/wav')

        mem = psutil.virtual_memory()
        st.write(f"🔍 Memory used: {mem.percent}%")
        with st.spinner("🔍 Transcribing with Whisper..."):
            segments, _ = st.session_state.whisper.transcribe(st.session_state.audio_path, beam_size=5)
            transcription = " ".join([seg.text for seg in segments])
            st.session_state.transcription = transcription

        st.success("📝 Transcription complete.")
        st.markdown(f"**Transcription:**\n\n{st.session_state.transcription}")

if st.session_state.transcription:
    if st.button("🗣️ Analyze Accent"):
        with st.spinner("🔍 Analyzing accent..."):
            try:
                mem = psutil.virtual_memory()
                st.write(f"🔍 Memory used: {mem.percent}%")
                waveform, sample_rate = torchaudio.load(st.session_state.audio_path)
                readable_accent, confidence = analyze_accent(waveform, sample_rate, st.session_state.classifier)

                if readable_accent:
                    st.success(f"✅ Accent Detected: **{readable_accent}**")
                    st.info(f"📊 Confidence: {confidence}%")
                else:
                    st.warning("Could not determine accent.")

            except Exception as e:
                st.error("❌ Failed to analyze accent.")
                st.code(str(e))

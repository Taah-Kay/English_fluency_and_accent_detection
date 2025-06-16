import streamlit as st
import tempfile
import shutil
import psutil
import torch
import torchaudio

from utils.audio_processing import trim_audio, download_audio_as_wav
from utils.video_processing import trim_video
from models.model_loader import load_accent_model, load_whisper
from utils.accent_analysis import analyze_accent
from utils.session_utils import initialize_session_state, display_memory_once, reset_session_state_except_model
from models.custom_interface import CustomEncoderWav2vec2Classifier

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
        shutil.rmtree(temp_dir, ignore_errors=True) # clear temporary files

elif option == "Enter Video Url":
    yt_url = st.text_input("Paste YouTube URL")
    if st.button("Download Video"):
        with st.spinner("Downloading video..."):
            audio_path = download_audio_as_wav(yt_url)
            audio_path = trim_audio(audio_path)
        if audio_path:
            st.success("✅ Video downloaded successfully.")
            st.session_state.audio_path = audio_path
            shutil.rmtree(temp_dir, ignore_errors=True)  # clear temporary files

# Transcription and Accent Analysis
if st.session_state.audio_path and not st.session_state.transcription:
    if st.button("🎧 Extract Audio"):
        st.session_state.audio_ready = True
        st.audio(st.session_state.audio_path, format='audio/wav')
        
        mem = psutil.virtual_memory()
        st.write(f"🔍 Memory used: {mem.percent}%")    
        #Detect Language AND FILTER OUT NON-ENGLISH AUDIOS FOR ANALYSIS
        segments, info = st.session_state.whisper.transcribe(st.session_state.audio_path, beam_size=1)
            
        # Convert segments (generator) to full transcription string
        st.session_state.transcription = " ".join([segment.text for segment in segments])
                  
        if info.language != "en":
                    
            st.error("❌ This video does not appear to be in English. Please provide a clear English video.")
        else:    
            # Show transcription for audio
            with st.spinner("Transcribing audio..."):
                st.markdown(" Transcript Preview")
                st.markdown(st.session_state.transcription)
                st.success("🎵 Audio extracted and ready for analysis!")
                mem = psutil.virtual_memory()
                st.write(f"🔍 Memory used: {mem.percent}%")

       

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
                    shutil.rmtree(temp_dir, ignore_errors=True)  # clear temporary files
                else:
                    st.warning("Could not determine accent.")
                    shutil.rmtree(temp_dir, ignore_errors=True)  # clear temporary files
            except Exception as e:
                st.error("❌ Failed to analyze accent.")
                st.code(str(e))

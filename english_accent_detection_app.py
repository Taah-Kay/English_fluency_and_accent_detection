
import os
from custom_interface import CustomEncoderWav2vec2Classifier
import streamlit as st
from moviepy.editor import VideoFileClip  
import requests
import tempfile
import subprocess
import torchaudio
import torch
from speechbrain.pretrained.interfaces import foreign_class
from faster_whisper import WhisperModel
from huggingface_hub import login
import psutil
import traceback
import shutil
from pydub import AudioSegment
from moviepy.config import change_settings

change_settings({"FFMPEG_BINARY": "ffmpeg"})  # Ensures MoviePy uses system ffmpeg
AudioSegment.converter = shutil.which("ffmpeg")


# -------------------------------
# Utility Function: Download Video
# -------------------------------

def download_audio_as_wav(url, max_filesize_mb=70):
    """
    Downloads audio from a URL using yt-dlp, extracts to mp3,
    then converts it to WAV using ffmpeg. Ensures file size is within the limit.
    Returns path to the .wav file or None on failure.
    """
    try:
        temp_dir = tempfile.mkdtemp()
        max_bytes = max_filesize_mb * 1024 * 1024
        output_template = os.path.join(temp_dir, "audio.%(ext)s")

        # Download using yt-dlp
        download_cmd = [
            "yt-dlp",
            "-f", f"bestaudio[filesize<={max_bytes}]",
            "--extract-audio",
            "--audio-format", "mp3",
            "-o", output_template,
            url
        ]

        result = subprocess.run(download_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            st.error("❌ yt-dlp failed to download the audio.")
            st.code(result.stderr.decode())
            return None

        # Locate downloaded .mp3 file
        mp3_files = [f for f in os.listdir(temp_dir) if f.endswith(".mp3")]
        if not mp3_files:
            st.error("❌ No MP3 file found after download.")
            return None
        mp3_path = os.path.join(temp_dir, mp3_files[0])

        # Convert to .wav using ffmpeg
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        convert_cmd = [
            "ffmpeg", "-y",
            "-i", mp3_path,
            "-ar", "16000",  # Resample to 16kHz
            "-ac", "1",      # Mono
            "-acodec", "pcm_s16le",  # WAV codec
            "-y",       
            temp_wav.name
        ]
        subprocess.run(convert_cmd, check=True)

        return temp_wav.name

    except subprocess.CalledProcessError as e:
        st.error("❌ ffmpeg failed during conversion.")
        st.code(str(e))
        return None

    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return None
# --------------------------
# Utility: Trim videos to 2 minutes
# --------------------------

def trim_video(video_path, max_duration=120):
    """
    Trims the video to the specified duration (in seconds) and extracts audio using ffmpeg.
    Returns the path to the trimmed audio (.wav).
    """
    try:
        # Use MoviePy only to check the video duration
        video = VideoFileClip(video_path, audio=True, verbose=True)
        duration = video.duration
        video.close()

        # Prepare output audio path
        audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name

        # Use ffmpeg to extract & trim audio directly
        command = [
            "ffmpeg",
            "-i", video_path,
            "-t", str(min(duration, max_duration)),  # Trim if needed
            "-ar", "16000",  # Resample to 16kHz
            "-ac", "1",      # Mono
            "-acodec", "pcm_s16le",  # WAV codec
            "-y",            # Overwrite if exists
            audio_path
        ]

        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            st.error("❌ ffmpeg failed to extract audio.")
            st.code(result.stderr.decode())
            return None

        return audio_path

    except Exception as e:
        st.error(f"❌ Error trimming video: {e}")
        st.code(traceback.format_exc())
        return None

# --------------------------
# Utility: Trim audios to 2 minutes
# --------------------------

def trim_audio(input_wav_path, max_duration_sec=120):
    """
    Trims the input .wav file to the first `max_duration_sec` seconds.
    Returns the path to the trimmed .wav file.
    """
    try:
        # Load audio using pydub
        audio = AudioSegment.from_wav(input_wav_path)

        # Trim to max_duration_sec
        trimmed_audio = audio[:max_duration_sec * 1000]  # pydub uses milliseconds

        # Save to a new temporary .wav file
        trimmed_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        trimmed_audio.export(trimmed_file.name, format="wav")

        return trimmed_file.name

    except Exception as e:
        st.error(f"❌ Error trimming audio: {e}")
        return None

# --------------------------
# Utility: Show memory once
# --------------------------
def display_memory_once():
    if 'memory_logged' not in st.session_state:
        mem = psutil.virtual_memory()
        st.markdown(f"🧠 **Memory Used:** {mem.percent}%")
        st.session_state.memory_logged = True

# --------------------------
# Utility: Initialize session vars
# --------------------------
def initialize_session_state():
    defaults = {
        "audio_path": None,
        "audio_ready": False,
        "transcription": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# -------------------------------
# Load Model (Cached)
# -------------------------------
@st.cache_resource(show_spinner="Loading model...")   # making sure we only load the model once per every app instance
def load_accent_model():
    """
    Loads the pre-trained accent classification model from HuggingFace.
    """

    st.write("🔧 Initializing PyTorch and model...")
    
    if not os.getenv("HF_TOKEN") and not os.getenv("hf_token"):
        
        st.error("❌ Hugging Face token not found. Please set HF_TOKEN in environment variables.")
        st.stop()
        
    try:
        classifier = foreign_class(
            source="Jzuluaga/accent-id-commonaccent_xlsr-en-english",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier"
        )

        st.success("✅ Model loaded successfully.")
        return classifier
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

# Load Whisper model (tiny version for speed)
@st.cache_resource
def load_whisper():
    return WhisperModel("tiny", device="cpu")

# -------------------------------
# Accent Prediction
# -------------------------------

# Accent label map
ACCENT_LABELS = {
    "us": "American Accent",
    "england": "British Accent",
    "australia": "Australian Accent",
    "indian": "Indian Accent",
    "canada": "Canadian Accent",
    "bermuda": "Bermudian Accent",
    "scotland": "Scottish Accent",
    "african": "African Accent",
    "ireland": "Irish Accent",
    "newzealand": "New Zealand Accent",
    "wales": "Welsh Accent",
    "malaysia": "Malaysian Accent",
    "philippines": "Philippine Accent",
    "singapore": "Singaporean Accent",
    "hongkong": "Hong Kong Accent",
    "southatlandtic": "South Atlantic Accent"
}
def analyze_accent(audio_tensor, sample_rate, model):
    """
    Uses the loaded model to classify the accent from the audio file.
    Returns the accent label and confidence score.
    """
    classifier = model
    audio_tensor = audio_tensor
    sample_rate = sample_rate
    
     
    try:

        # Convert stereo to mono (if needed)
        if audio_tensor.shape[0] > 1:
            audio_tensor = audio_tensor.mean(dim=0, keepdim=True)  # [1, time]

        # Remove channel dimension to get [time]
        audio_tensor = audio_tensor.squeeze(0)

        # Add batch dimension to get [1, time]
        audio_tensor = audio_tensor.unsqueeze(0).to(torch.float32) 
        
        device = torch.device("cpu")  # or "cuda" if using GPU
        audio_tensor = audio_tensor.to(device)
        
        mem = psutil.virtual_memory()
        st.write(f"🔍 Memory used: {mem.percent}%")
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio_tensor = resampler(audio_tensor)
           
        with torch.no_grad():  
            out_prob, score, index, text_lab = classifier.classify_batch(audio_tensor)
        
            accent_label = text_lab[0]
            readable_accent = ACCENT_LABELS.get(accent_label, accent_label.title() + " accent")
        
        return readable_accent, round(score[0].item() * 100, 2)
    
    except Exception as e:
        st.error(f"❌ Error during accent classification:\n{traceback.format_exc()}")
        return None, None

# -------------------------------
# Main Streamlit App
# -------------------------------
def main():
    
        
  
    def reset_session_state_except_model():
        
        keys_to_keep = {"classifier", "whisper" }  # Keep only the models
        keys_to_delete = [key for key in st.session_state.keys() if key not in keys_to_keep]
    
        for key in keys_to_delete:
            del st.session_state[key]
            
    # Initialize session vars
    initialize_session_state()

    #  Load models only once
    if 'classifier' not in st.session_state: 
        st.session_state.classifier = load_accent_model()
    if 'whisper' not in st.session_state: 
        st.session_state.whisper = load_whisper()
 

    # 🔍 Show memory info after model load
    display_memory_once()

    if st.button("🔄 Analyze new video"):
        reset_session_state_except_model()
        st.rerun()


  
    # check if ffmpeg installed     
    if not shutil.which("ffmpeg"):
        raise EnvironmentError("FFmpeg not found. Please install ffmpeg or add it to PATH.")

        
    st.title("🎙️ English Accent Audio Detector")

    
    

    # Input selection
    option = st.radio("Choose input method:", ["Upload video file","Enter Video Url"])
    

    # File uploader option
    if option == "Upload video file":
        uploaded_video = st.file_uploader("Upload your video", type=["mp4", "mov", "avi", "mkv"])
        if uploaded_video is not None:
            temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            with open(temp_video_path.name, "wb") as f:
                f.write(uploaded_video.read())
            audio_path = trim_video(temp_video_path.name)    # returns an audio file not more than 2 minutes long
            st.success("✅ Video uploaded successfully.")
            st.session_state.audio_path = audio_path 

  
      #YouTube video downloads
    elif option == "Enter Video Url":
        yt_url = st.text_input("Paste YouTube")
        if st.button("Download Video"):
            with st.spinner("Downloading video..."):
                audio_path = download_audio_as_wav(yt_url)
                audio_path = trim_audio(audio_path)      
            if audio_path:
                st.success("✅ Video downloaded successfully.")
                st.session_state.audio_path = audio_path 
                
    mem = psutil.virtual_memory()
    st.write(f"🔍 Memory used: {mem.percent}%")
    # Process and analyze video
    if st.session_state.audio_path and not st.session_state.transcription:   
        if st.button("Extract Audio"):
            
            st.session_state.audio_ready = True      
            st.audio(st.session_state.audio_path , format='audio/wav')
            
                
            try:
                st.success("Now filtering language")
            # Detect Language AND FILTER OUT NON-ENGLISH AUDIOS FOR ANALYSIS
                segments, info = st.session_state.whisper.transcribe(st.session_state.audio_path, beam_size=5)
                st.success("Now joining segments ")
                # Convert segments (generator) to full transcription string
                st.session_state.transcription = " ".join([segment.text for segment in segments])
                st.success("sucessfully created segments")    
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
            except Exception as e:
                        st.error(f"❌ Error filtering audio: {e}")
                        st.stop()

                  
                
                
                # Perform accent analysis
                 
    if st.session_state.audio_ready:   
        if st.button("Analyze accent"):
            try:         
                with st.spinner("Analyzing accent..."):
                         
                    waveform, sample_rate = torchaudio.load(st.session_state.audio_path) # Process the audio for model inference
                    accent, confidence = analyze_accent(waveform, sample_rate, st.session_state.classifier) #Parse the processed audio to the model

                            # Display results
                st.subheader("🎧 Accent Detection Result")
                st.write(f"The speaker in the video has a ", accent)
                st.write(f"🧠 Confidence Score: **{confidence}%**")


                    
            except Exception as e:
                st.error(f"❌ Error during accent analysis: {e}")
                st.stop()

                    


# Run the app
if __name__ == "__main__":
    main()

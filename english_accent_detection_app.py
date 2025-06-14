
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


# -------------------------------
# Utility Function: Download Video
# -------------------------------
def download_video_from_url(url):
    """
    Downloads a video from the given URL and saves it to a temporary file.
    Returns the path to the saved file.
    """
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            with open(temp_file.name, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return temp_file.name
        else:
            st.error("❌ Failed to download video from the URL.")
            return None
    except Exception as e:
        st.error(f"❌ Error downloading video: {e}")
        return None

def download_social_video(url):
    """Download 240p video from YouTube yt-dlp.""" # we are more interested in the audio not picture quality ***Memory management
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        command = [
            "yt-dlp",
            "-f", "bestvideo[height<=240]+bestaudio/best[height<=240]",
            "-o", temp_file.name,
            url
        ]
        subprocess.run(command, check=True)
        return temp_file.name
    except subprocess.CalledProcessError as e:
        st.error("Download failed. The URL may be invalid or unsupported.")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


# -------------------------------
# Utility Function: Extract Audio
# -------------------------------
def extract_audio(video_path):
    """
    Extracts audio from the video and saves it as a WAV file.
    Returns the path to the audio file.
    """
    try:
        video = VideoFileClip(video_path)
        if video.audio is None:
            st.error("❌ No audio found in the video.")
            return None

        audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        video.audio.write_audiofile(audio_path, fps=16000, codec='pcm_s16le')
        return audio_path
    except Exception as e:
        st.error(f"❌ Error extracting audio: {e}")
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
        "video_path": None,
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
    import shutil
    if not shutil.which("ffmpeg"):
        raise EnvironmentError("FFmpeg not found. Please install ffmpeg or add it to PATH.")

        
    st.title("🎙️ English Accent Audio Detector")

    
    

    # Input selection
    option = st.radio("Choose input method:", ["Upload video file", "Enter direct MP4 URL","Enter YouTube link"])
    video_path = None

    # File uploader option
    if option == "Upload video file":
        uploaded_video = st.file_uploader("Upload your video", type=["mp4", "mov", "avi", "mkv"])
        if uploaded_video is not None:
            temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            with open(temp_video_path.name, "wb") as f:
                f.write(uploaded_video.read())
            video_path = temp_video_path.name
            st.success("✅ Video uploaded successfully.")
            st.session_state.video_path = video_path 

    # Direct URL input option
    elif option == "Enter direct MP4 URL":
        video_url = st.text_input("Enter direct video URL (e.g., MP4 link)")
        if st.button("Download Video"):
            video_path = download_video_from_url(video_url)
            if video_path:
                st.success("✅ Video downloaded successfully.")
                st.session_state.video_path = video_path 

     
      #YouTube and TikTok video downloads
    elif option == "Enter YouTube link":
        yt_url = st.text_input("Paste YouTube")
        if st.button("Download from Social Media"):
            video_path = download_social_video(yt_url)
            if video_path:
                st.success("✅ Video downloaded successfully.")
                st.session_state.video_path = video_path 

    # Process and analyze video
    if st.session_state.video_path and not st.session_state.transcription:   
        if st.button("Extract Audio"):
            
    
            audio_path = extract_audio(st.session_state.video_path)   
            st.session_state.audio_path = audio_path
            st.session_state.audio_ready = True
            
            if audio_path:
                st.audio(st.session_state.audio_path , format='audio/wav')

                
                try:
                # Step 1: Detect Language AND FILTER OUT NON-ENGLISH AUDIOS FOR ANALYSIS
                    st.session_state.transcription, info = st.session_state.whisper.transcribe(st.session_state.audio_path, beam_size=5)

                except Exception as e:
                            st.error(f"❌ Error filtering audio: {e}")
                            st.stop()

                  
                if info.language != "en":
                        os.remove(video_path)
                        os.remove(audio_path)
                        st.error("❌ This video does not appear to be in English. Please provide a clear English video.")
                else:    
                    # Step 3: Show transcription for audio
                    
                    st.markdown(" Transcript Preview: ", st.session_state.transcription)
                    st.success("🎵 Audio extracted and ready for analysis!")
                
                # Perform accent analysis
                 
    if st.session_state.audio_ready and st.session_state.audio_path:   
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

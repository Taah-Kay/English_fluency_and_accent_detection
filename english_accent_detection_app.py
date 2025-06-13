import streamlit as st
from moviepy.editor import VideoFileClip  
import requests
import tempfile
import subprocess
from huggingface_hub import login
import os

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
    """Download 240p video from YouTube or TikTok using yt-dlp.""" # we are more interested in the audio not picture quality ***Memory management
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


# -------------------------------
# Load Model (Cached)
# -------------------------------
@st.cache_resource(show_spinner="Loading model...")   # making sure we only load the model once per every app instance
def load_accent_model():
    # Authenticate with Hugging Face to avoid 429 errors
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
    """
    Loads the pre-trained accent classification model from HuggingFace.
    """
    login(token=os.getenv("hf_token"))
    
    st.write("🔧 Initializing PyTorch and model...")
    from speechbrain.pretrained.interfaces import foreign_class
    import torchaudio

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
def analyze_accent(audio_tensor, sample_rate):
    """
    Uses the loaded model to classify the accent from the audio file.
    Returns the accent label and confidence score.
    """
    try:
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio_tensor = resampler(audio_tensor)
            
        out_prob, score, index, text_lab = classifier.classify_batch(audio_tensor)
        accent_label = text_lab[0]
        readable_accent = ACCENT_LABELS.get(accent_label, accent_label.title() + " Accent")
        
        return readable_accent, round(score[0].item() * 100, 2)
    
    except Exception as e:
        st.error(f"❌ Error during accent classification: {e}")
        st.stop()


# -------------------------------
# Main Streamlit App
# -------------------------------
def main():
    st.title("🎙️ English Accent Audio Detector")

    # Load model only once
    classifier = load_accent_model()

    # Input selection
    option = st.radio("Choose input method:", ["Upload video file", "Enter direct MP4 URL","Enter YouTube or Tiktok link"])
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

    # Direct URL input option
    elif option == "Enter direct MP4 URL":
        video_url = st.text_input("Enter direct video URL (e.g., MP4 link)")
        if st.button("Download Video"):
            video_path = download_video_from_url(video_url)
            if video_path:
                st.success("✅ Video downloaded successfully.")

     
      #YouTube and TikTok video downloads
    elif option == "Enter YouTube or Tiktok link":
        yt_url = st.text_input("Paste YouTube/TikTok link")
        if st.button("Download from Social Media"):
            video_path = download_social_video(yt_url)
            if video_path:
                st.success("Video downloaded from social media.")

    # Process and analyze video
    if video_path:
        if st.button("Extract Audio"):
            audio_path = extract_audio(video_path)
            if audio_path:
                st.audio(audio_path, format='audio/wav')
                st.success("🎵 Audio extracted and ready for analysis!")

                # Perform accent analysis
                if st.button("Analyze accent"):
                    try:
                        waveform, sample_rate = torchaudio.load(audio_path) # Process the audio for model inference
                        st.success("Sucessfully created a waveform!")
                        accent, confidence = classify_accent(waveform, sample_rate) #Parse the processed audio to the model
                    except Exception as e:
                        st.error(f"❌ Error during accent analysis: {e}")
                        st.stop()

                # Display results
                st.subheader("🎧 Accent Detection Result")
                st.write(f"The speaker in the video has a **{accent}** accent.")
                st.write(f"🧠 Confidence Score: **{confidence}%**")

                # Provide interpretation of confidence score
                if confidence > 85:
                    st.success("✅ High confidence in prediction.")
                elif confidence > 60:
                    st.info("ℹ️ Moderate confidence. Might be some accent overlap.")
                else:
                    st.warning("⚠️ Low confidence. Try using a clearer audio sample.")


# Run the app
if __name__ == "__main__":
    main()

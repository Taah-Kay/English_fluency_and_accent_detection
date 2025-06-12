import streamlit as st
from moviepy.editor import VideoFileClip  
import requests
import tempfile


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
@st.cache_resource
def load_accent_model():
    """
    Loads the pre-trained accent classification model from HuggingFace.
    """
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
def analyze_accent(audio_path, classifier):
    """
    Uses the loaded model to classify the accent from the audio file.
    Returns the accent label and confidence score.
    """
    try:
        out_prob, score, index, label = classifier.classify_file(audio_path)
        score = round(score[0].item() * 100, 2)
        return label, score
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
    option = st.radio("Choose input method:", ["Upload video file", "Enter direct MP4 URL"])
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

    # URL input option
    elif option == "Enter direct MP4 URL":
        video_url = st.text_input("Enter direct video URL (e.g., MP4 link)")
        if st.button("Download Video"):
            video_path = download_video_from_url(video_url)
            if video_path:
                st.success("✅ Video downloaded successfully.")

    # Process and analyze video
    if video_path:
        if st.button("Extract Audio"):
            audio_path = extract_audio(video_path)
            if audio_path:
                st.audio(audio_path, format='audio/wav')
                st.success("🎵 Audio extracted and ready for analysis!")

                # Perform accent analysis
                st.info("Analyzing accent...")
                try:
                    accent, confidence = analyze_accent(audio_path, classifier)
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

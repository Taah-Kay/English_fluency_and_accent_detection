import os
import tempfile
import subprocess
import streamlit as st
from pydub import AudioSegment
import shutil

AudioSegment.converter = shutil.which("ffmpeg")

# -------------------------------
# Utility Function: Download audio from a Video url
# -------------------------------
def download_audio_as_wav(url, max_filesize_mb=70):
    """
    Downloads audio from a URL using yt-dlp, then converts it to WAV using ffmpeg.
    Ensures file size is within the limit. Returns path to .wav file or None on failure.
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
            "--audio-format", "mp3",  # You may consider omitting this for better quality
            "--no-playlist",
            "-o", output_template,
            url
        ]

        subprocess.run(download_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        # Find downloaded audio
        mp3_files = [f for f in os.listdir(temp_dir) if f.endswith(".mp3")]
        if not mp3_files:
            st.error("❌ No MP3 file found after download.")
            return None
        mp3_path = os.path.join(temp_dir, mp3_files[0])

        # Convert to WAV
        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        convert_cmd = ["ffmpeg", "-y", "-i", mp3_path, temp_wav.name]
        subprocess.run(convert_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        
        return temp_wav.name

    except subprocess.CalledProcessError as e:
        st.error("❌ Audio download or conversion failed.")
        st.code(e.stderr.decode() if hasattr(e, 'stderr') else str(e))
        if mp3_path and os.path.exists(mp3_path):
            os.remove(mp3_path)
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
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
        if trimmed_file and os.path.exists(trimmed_file.name):
            os.remove(trimmed_file.name)
        return None

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
    Downloads audio from a URL using yt-dlp, extracts to mp3,
    then converts it to WAV using ffmpeg.
    """
    try:
        temp_dir = tempfile.mkdtemp()
        max_bytes = max_filesize_mb * 1024 * 1024
        output_template = os.path.join(temp_dir, "audio.%(ext)s")

        download_cmd = [
            "yt-dlp", "-f", f"bestaudio[filesize<={max_bytes}]",
            "--extract-audio", "--audio-format", "mp3",
            "-o", output_template, url
        ]
        result = subprocess.run(download_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            st.error("❌ yt-dlp failed.")
            st.code(result.stderr.decode())
            return None

        mp3_path = next((os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".mp3")), None)
        if not mp3_path:
            st.error("❌ No MP3 file found.")
            return None

        temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        convert_cmd = [
            "ffmpeg", "-y", "-i", mp3_path, "-ar", "16000", "-ac", "1",
            "-acodec", "pcm_s16le", temp_wav.name
        ]
        subprocess.run(convert_cmd, check=True)
        return temp_wav.name

    except subprocess.CalledProcessError as e:
        st.error("❌ ffmpeg conversion failed.")
        st.code(str(e))
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return None

# --------------------------
# Utility: Trim audios to 2 minutes
# --------------------------
def trim_audio(input_wav_path, max_duration_sec=120):
    """Trims WAV file to `max_duration_sec` seconds."""
    try:
        audio = AudioSegment.from_wav(input_wav_path)
        trimmed_audio = audio[:max_duration_sec * 1000]
        trimmed_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        trimmed_audio.export(trimmed_file.name, format="wav")
        return trimmed_file.name
    except Exception as e:
        st.error(f"❌ Error trimming audio: {e}")
        return None

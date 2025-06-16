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
    Supports fallback formats (.m4a, .webm, .opus) if .mp3 not found.
    Cleans up temporary files after use.
    Returns path to .wav file or None on failure.
    """
    audio_path = None
    temp_wav = None

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            max_bytes = max_filesize_mb * 1024 * 1024
            output_template = os.path.join(temp_dir, "audio.%(ext)s")

            # yt-dlp download command
            download_cmd = [
                "yt-dlp",
                "-f", f"bestaudio[filesize<={max_bytes}]",
                "--extract-audio",
                "--audio-format", "mp3",
                "--no-playlist",
                "--no-cache-dir",
                "--restrict-filenames",
                "-o", output_template,
                url
            ]

            subprocess.run(download_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

            # Try to locate audio file (mp3 or fallback)
            common_exts = [".mp3", ".m4a", ".webm", ".opus"]
            for ext in common_exts:
                matches = [f for f in os.listdir(temp_dir) if f.endswith(ext)]
                if matches:
                    audio_path = os.path.join(temp_dir, matches[0])
                    break

            if not audio_path or not os.path.exists(audio_path):
                st.error("❌ No supported audio file found after download.")
                return None

            # Convert to WAV (outside temp_dir so it persists)
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            convert_cmd = ["ffmpeg", "-y", "-i", audio_path, temp_wav.name]
            subprocess.run(convert_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

            # Return WAV file path; temp_dir and downloaded audio cleaned automatically
            return temp_wav.name

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode() if hasattr(e, "stderr") else str(e)
        if "st" in globals():
            st.error("❌ Audio download or conversion failed.")
            st.code(error_msg)
        else:
            print("Error during processing:", error_msg)
        # Cleanup wav if created
        if temp_wav is not None and os.path.exists(temp_wav.name):
            os.remove(temp_wav.name)
        return None

    except Exception as e:
        if "st" in globals():
            st.error("❌ Unexpected error occurred.")
            st.code(str(e))
        else:
            print("Unexpected error:", e)
        if temp_wav is not None and os.path.exists(temp_wav.name):
            os.remove(temp_wav.name)
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

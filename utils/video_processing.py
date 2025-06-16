import tempfile
import subprocess
import os
from moviepy.editor import VideoFileClip
import streamlit as st
import traceback
import shutil


# --------------------------
# Utility: Trim videos to 2 minutes
# --------------------------
def trim_video(video_path, max_duration=120):
    """Trims video to max_duration (in seconds) and extracts audio."""
    try:
        video = VideoFileClip(video_path)
        duration = video.duration
        video.close()

        audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        command = [
            "ffmpeg", "-i", video_path,
            "-t", str(min(duration, max_duration)),
            "-ar", "16000", "-ac", "1",
            "-acodec", "pcm_s16le", "-y", audio_path
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            st.error("❌ ffmpeg audio extraction failed.")
            os.remove(audio_path)  # Clean up failed temp file
            st.code(result.stderr.decode())
            return None

        return audio_path
    except Exception as e:
        st.error(f"❌ Error trimming video: {e}")
        os.remove(audio_path)
        st.code(traceback.format_exc())
        return None
        
    finally:
        # Clean up input video if it was a temp file
        if "tmp" in video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass  # Avoid crashing on cleanup

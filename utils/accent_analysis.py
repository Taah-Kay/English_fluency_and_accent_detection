import torch
import torchaudio
import streamlit as st
import traceback
import psutil


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
    """Classifies audio to identify English accent."""
    try:
        # Convert stereo to mono (if needed)
        if audio_tensor.shape[0] > 1:
            audio_tensor = audio_tensor.mean(dim=0, keepdim=True)
        audio_tensor = audio_tensor.squeeze(0).unsqueeze(0).to(torch.float32)

        # Convert to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            audio_tensor = resampler(audio_tensor)

        audio_tensor = audio_tensor.to("cpu")
        with torch.no_grad():
            # Perform Classification
            out_prob, score, index, text_lab = model.classify_batch(audio_tensor)
            accent_label = text_lab[0]
            readable = ACCENT_LABELS.get(accent_label, accent_label.title() + " accent")
            return readable, round(score[0].item() * 100, 2)
    except Exception:
        st.error("❌ Error during classification.")
        st.code(traceback.format_exc())
        return None, None

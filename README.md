# 🎙️ English Accent Audio Detector

This Streamlit web application detects the English accent of a speaker in a video or audio file. It leverages pre-trained machine learning models and allows users to upload a video or input a YouTube URL. The app extracts and trims the audio, transcribes the speech, filters for English language, and then predicts the accent from a set of common English accents.

🚀 Features
* ✅ Upload videos or provide YouTube/video url links

* 🎧 Automatic audio extraction and trimming (max 2 minutes)

* 🗣️ Transcription and English language filtering

* 🧠 Accent classification using a Wav2Vec2-based model

* 💬 Clear display of the speaker's accent and confidence score

### 🧪 Model Information
This app uses the Accent-ID model developed by Juan Pablo Zuluaga on the CommonAccent dataset.

The model is implemented using the SpeechBrain toolkit, a powerful open-source speech processing framework built on PyTorch.


## ⚙️ Setup Instructions

This application requires **Python 3.10**. Follow these steps to install and run the app locally:

---

* ✅ 1. Clone the Repository


git clone https://github.com/Taah-Kay/English_fluency_and_accent_detection.git
cd English_fluency_and_accent_detection

* 🐍 2. Set Up a Virtual Environment (Recommended)
python3.10 -m venv venv
source venv/bin/activate         # On Windows: venv\Scripts\activate

* 📦 3. Install Python Requirements
Make sure you're in the root folder, then run:
pip install -r requirements.txt

* 🛠️ 4. Install System Dependencies
This app uses ffmpeg for audio/video processing. Install it based on your OS:
Ubuntu/Debian:
sudo apt update
sudo apt install ffmpeg

macOS (with Homebrew):
brew install ffmpeg

Windows:
Download from https://ffmpeg.org/download.html and add it to your system PATH.

* 🔐 5. Set Your Hugging Face Token
  * The app uses models from Hugging Face. You need an access token:

  * Create a token at: https://huggingface.co/settings/tokens

  Then set it as an environment variable:
   * export HF_TOKEN=your_token_here        # Linux/macOS
   * set HF_TOKEN=your_token_here           # Windows CMD
   * $env:HF_TOKEN="your_token_here"        # Windows PowerShell

* ▶️ 6. Run the App
 Launch the Streamlit app:
 streamlit run app.py
 Then open your browser at: http://localhost:8501

### 📁 Project Structure
<pre lang="text"> English_Accent_Audio_Detector/ ├── app.py # Main Streamlit application ├── README.md # Project documentation with usage instructions ├── requirements.txt # Python dependencies ├── packages.txt # System-level packages (e.g., ffmpeg) ├── utils/ # Utility modules │ ├── __init__.py # Makes utils a package │ ├── accent_analysis.py # Logic for analyzing and classifying accents │ ├── audio_processing.py# Audio trimming and processing │ ├── session_utils.py # Session state utilities │ └── video_processing.py# YouTube/video handling ├── models/ # Model management ├── __init__.py # Makes models a package ├── custom_interface.py# HuggingFace interface for the model └── model_loader.py # Loads Wav2Vec2 and Whisper models </pre>


### 🚀 How to Use
* Run the app:

* Choose Input Method:

  * Upload a video file (.mp4, .mov, .avi, .mkv)

  * Or paste a valid YouTube URL.

* Click "Download Video" (for YouTube) or upload file directly.

* Click "Extract Audio"
  The app:

    * Extracts and trims audio to 2 minutes

    * Transcribes the content

    * Checks for English language

* Click "Analyze Accent"

  Displays the detected accent and confidence score.

### 🧠 Supported English Accents
* American
* British
* Australian
* Indian
* Canadian
* Bermudian
* Scottish
* African
* Irish
* New Zealand
* Welsh
* Malaysian
* Philippine
* Singaporean
* Hong Kong
* South Atlantic

### 🛠️ Troubleshooting
FFmpeg Not Found: Ensure it’s installed and accessible from the terminal/command prompt.

Model Load Errors: Ensure your Hugging Face token is valid.

Audio Errors: Use clear, spoken English in the video/audio source.

### 📬 Contact
Created by Ryan Kembo
📧 kemboryan@gmail.com
🔗https://github.com/Taah-Kay

### 🧾 Citation & Acknowledgements
This app uses the Accent-ID model from Hugging Face:

Juan Pablo Zuluaga – Jzuluaga/accent-id-commonaccent_xlsr-en-english 
https://github.com/JuanPZuluaga/accent-recog-slt2022

Built on SpeechBrain, a general-purpose speech toolkit:

<pre lang="markdown"> ```bibtex @misc{speechbrain, title={{SpeechBrain}: A General-Purpose Speech Toolkit}, author={Mirco Ravanelli and Titouan Parcollet and Peter Plantinga and Aku Rouhe and Samuele Cornell and Loren Lugosch and Cem Subakan and Nauman Dawalatabad and Abdelwahab Heba and Jianyuan Zhong and Ju-Chieh Chou and Sung-Lin Yeh and Szu-Wei Fu and Chien-Feng Liao and Elena Rastorgueva and François Grondin and William Aris and Hwidong Na and Yan Gao and Renato De Mori and Yoshua Bengio}, year={2021}, eprint={2106.04624}, archivePrefix={arXiv}, primaryClass={eess.AS}, note={arXiv:2106.04624} } ``` </pre>



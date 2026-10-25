import io
import os
import tempfile
import hashlib
import json
from datetime import datetime

import requests
import numpy as np
import pandas as pd
import librosa
import librosa.display
import streamlit as st
import matplotlib.pyplot as plt
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


TITLE = "VoiceGuard AI — Advanced"
DETECTOR_ID = "garystafford/wav2vec2-deepfake-voice-detector"
ASR_ID = "openai/whisper-tiny"


def tinyfish_search(query):
    """Search the web using the configured TinyFish API key."""
    try:
        api_key = st.secrets.get("TINYFISH_API_KEY")
    except Exception:
        api_key = None

    api_key = api_key or os.getenv("TINYFISH_API_KEY")

    if not api_key:
        st.error(
            "TinyFish API key nahi mili. Project folder mein "
            ".streamlit/secrets.toml check karo."
        )
        return

    try:
        response = requests.get(
            "https://api.search.tinyfish.ai",
            params={"query": query},
            headers={"X-API-Key": api_key},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        results = data.get("results", []) if isinstance(data, dict) else []
        if not results and isinstance(data, dict):
            results = data.get("data", [])

        if not results:
            st.info("Koi search result nahi mila, ya API response ka format alag hai.")
            return

        for item in results:
            if not isinstance(item, dict):
                continue
            title = item.get("title") or item.get("name") or "Untitled result"
            snippet = item.get("snippet") or item.get("description") or item.get("content") or ""
            url = item.get("url") or item.get("link") or ""

            st.markdown(f"### {title}")
            if snippet:
                st.write(snippet)
            if isinstance(url, str) and url.startswith(("https://", "http://")):
                st.markdown(f"[Open source]({url})")
            st.divider()

    except Exception as exc:
        st.error(f"Search failed: {exc}")


st.set_page_config(page_title=TITLE, page_icon="🛡️", layout="wide")

st.title("🛡️ VoiceGuard AI (Advanced Edition)")
st.caption("AI Voice Cloning & Deepfake Audio Forensic Screening Assistant")

FEATURES = [
    "Live Audio Recording & File Upload",
    "Batch Audio Upload (Multi-file Support)",
    "AI Deepfake Voice Detection",
    "Real vs Fake Score Breakdown",
    "Waveform & Mel-Spectrogram Analysis",
    "Acoustic Metrics & Characteristics",
    "Speech-to-Text Transcript (Whisper Tiny)",
    "Multi-format Export (PDF, JSON, CSV)"
]
with st.expander("✨ Advanced Project Features", expanded=False):
    feature_cols = st.columns(2)
    for index, feature in enumerate(FEATURES):
        feature_cols[index % 2].markdown(f"**{index + 1}.** {feature}")

if "history" not in st.session_state:
    st.session_state.history = []
if "last" not in st.session_state:
    st.session_state.last = None


with st.sidebar:
    st.header("Privacy & Limitations")
    st.write(
        "Audio is processed locally/temporarily. Temporary files are deleted "
        "after processing. Model scores are not definitive proof."
    )
    if st.button("Clear session history"):
        st.session_state.history = []
        st.session_state.last = None
        st.success("History cleared.")


@st.cache_resource(show_spinner=False)
def load_detector():
    extractor = AutoFeatureExtractor.from_pretrained(DETECTOR_ID)
    model = AutoModelForAudioClassification.from_pretrained(DETECTOR_ID)
    model.eval()
    return extractor, model


@st.cache_resource(show_spinner=False)
def load_asr():
    from transformers import pipeline
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("automatic-speech-recognition", model=ASR_ID, device=device)


def metrics(y, sr):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return {
        "Duration (seconds)": round(len(y) / sr, 2),
        "Sample rate (Hz)": int(sr),
        "RMS energy": round(float(np.sqrt(np.mean(y**2))), 6),
        "Zero-crossing rate": round(float(np.mean(librosa.feature.zero_crossing_rate(y=y))), 6),
        "Spectral centroid (Hz)": round(float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))), 2),
        "Spectral rolloff (Hz)": round(float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))), 2),
        "Mean MFCC variance": round(float(np.mean(np.var(mfcc, axis=1))), 4),
    }


def make_pdf(result):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()

    story = [
        Paragraph("VoiceGuard AI — Advanced Forensic Report", styles["Title"]),
        Paragraph("Created: " + result["created_at"], styles["Normal"]),
        Paragraph("File: " + result["filename"], styles["Normal"]),
        Spacer(1, 12),
        Paragraph("Model Output Summary", styles["Heading2"]),
    ]

    rows = [["Model label", "Score (%)"]]
    rows += [[item["label"], f'{item["score"] * 100:.2f}%'] for item in result.get("scores", [])]
    score_table = Table(rows, colWidths=[330, 120])
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))

    story += [score_table, Spacer(1, 12), Paragraph("Audio Characteristics", styles["Heading2"])]
    rows2 = [["Metric", "Value"]] + [[key, str(value)] for key, value in result.get("metrics", {}).items()]
    metrics_table = Table(rows2, colWidths=[280, 170])
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgreen),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))

    transcript = (result.get("transcript") or "Not generated.").replace("&", "&amp;")
    story += [
        metrics_table, Spacer(1, 12),
        Paragraph("Transcript", styles["Heading2"]),
        Paragraph(transcript, styles["BodyText"])
    ]

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


# Main Input Tabs: File Upload OR Live Recording
tab_upload, tab_record = st.tabs(["📁 Upload File(s)", "🎙️ Record Live Audio"])

audio_items_to_process = []

with tab_upload:
    uploaded_files = st.file_uploader(
        "Upload audio file(s) for analysis",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        accept_multiple_files=True
    )
    if uploaded_files:
        for f in uploaded_files:
            audio_items_to_process.append({"name": f.name, "bytes": f.getvalue()})

with tab_record:
    st.write("Click the microphone button to start recording your voice:")
    recorded_audio = st.audio_input("Record audio from microphone")
    if recorded_audio is not None:
        rec_bytes = recorded_audio.getvalue()
        rec_name = f"live_recording_{datetime.now().strftime('%H%M%S')}.wav"
        audio_items_to_process.append({"name": rec_name, "bytes": rec_bytes})

# Processing Section
if audio_items_to_process:
    for item_data in audio_items_to_process:
        raw = item_data["bytes"]
        file_name = item_data["name"]
        file_id = hashlib.sha256(raw).hexdigest()[:12]
        
        with st.container():
            st.subheader(f"🔊 Source: {file_name}")
            st.audio(raw)
            st.caption(f"Size: {len(raw) / 1024:.1f} KB · ID: {file_id}")

            col1, col2 = st.columns(2)
            detect = col1.button(f"Analyze {file_name}", key=f"det_{file_id}", type="primary")
            transcribe = col2.button(f"Transcribe {file_name}", key=f"tr_{file_id}")

            if detect:
                temp_path = None
                try:
                    with st.spinner("Analyzing audio with deepfake detector..."):
                        suffix = os.path.splitext(file_name)[1] or ".wav"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                            temp_file.write(raw)
                            temp_path = temp_file.name

                        y, sr = librosa.load(temp_path, sr=16000, mono=True, duration=20)

                        if len(y) < int(sr * 0.5):
                            st.error("Audio is too short (min 0.5s required).")
                        else:
                            extractor, model = load_detector()
                            inputs = extractor(y, sampling_rate=sr, return_tensors="pt")

                            with torch.no_grad():
                                logits = model(**inputs).logits
                                probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()

                            labels = getattr(model.config, "id2label", {}) or {}
                            scores = [
                                {
                                    "label": str(labels.get(i, labels.get(str(i), f"Class {i}"))),
                                    "score": float(probability),
                                }
                                for i, probability in enumerate(probabilities)
                            ]
                            scores.sort(key=lambda item: item["score"], reverse=True)

                            result = {
                                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "filename": file_name,
                                "file_id": file_id,
                                "model_id": DETECTOR_ID,
                                "scores": scores,
                                "metrics": metrics(y, sr),
                                "transcript": "",
                                "raw_bytes": raw
                            }
                            st.session_state.last = result
                            st.session_state.history.insert(0, result.copy())
                            st.success("Analysis complete!")
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)

            if transcribe:
                temp_path = None
                try:
                    with st.spinner("Generating transcript via Whisper Tiny..."):
                        suffix = os.path.splitext(file_name)[1] or ".wav"
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                            temp_file.write(raw)
                            temp_path = temp_file.name

                        transcript = load_asr()(temp_path).get("text", "").strip()

                        if st.session_state.last and st.session_state.last.get("file_id") == file_id:
                            st.session_state.last["transcript"] = transcript
                        else:
                            result = {
                                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "filename": file_name,
                                "file_id": file_id,
                                "model_id": "Whisper Tiny",
                                "scores": [],
                                "metrics": {},
                                "transcript": transcript,
                                "raw_bytes": raw
                            }
                            st.session_state.last = result
                            st.session_state.history.insert(0, result.copy())
                        st.success("Transcript generated!")
                except Exception as exc:
                    st.error(f"Transcription failed: {exc}")
                finally:
                    if temp_path and os.path.exists(temp_path):
                        os.remove(temp_path)
            st.divider()

# Results display for the last evaluated audio
if st.session_state.last:
    result = st.session_state.last
    st.header("📊 Results & Visualization Dashboard")

    if result.get("scores"):
        st.subheader("Model Score Breakdown")
        for score_item in result["scores"]:
            st.write(f'**{score_item["label"]}** — {score_item["score"] * 100:.2f}%')
            st.progress(max(0.0, min(1.0, score_item["score"])))

        top_score = result["scores"][0]
        if any(w in top_score["label"].lower() for w in ("fake", "spoof", "synthetic", "generated", "deepfake", "ai")):
            st.warning("⚠️ Top label indicates potential synthetic/fake speech.")
        else:
            st.info("ℹ️ Top label does not clearly indicate synthetic speech.")

    st.subheader("Acoustic Characteristics")
    if result.get("metrics"):
        cols = st.columns(3)
        for idx, (k, v) in enumerate(result["metrics"].items()):
            cols[idx % 3].metric(k, v)

    # Waveform & Mel-Spectrogram Visualization
    if "raw_bytes" in result:
        try:
            waveform, sr_w = librosa.load(io.BytesIO(result["raw_bytes"]), sr=16000, mono=True, duration=20)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
            # Waveform
            ax1.plot(np.arange(len(waveform)) / sr_w, waveform, color='blue', linewidth=0.6)
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Amplitude")
            ax1.set_title("Waveform")

            # Mel-Spectrogram
            S = librosa.feature.melspectrogram(y=waveform, sr=sr_w, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_dB, sr=sr_w, x_axis='time', y_axis='mel', ax=ax2)
            fig.colorbar(img, ax=ax2, format='%+2.0f dB')
            ax2.set_title("Mel-Spectrogram")

            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception:
            st.caption("Visualizations could not be generated.")

    if result.get("transcript"):
        st.subheader("Transcript")
        st.text_area("Recognized Speech", result["transcript"], height=100)

    # Export options
    st.subheader("📥 Export Reports")
    col_pdf, col_json, col_csv = st.columns(3)
    
    with col_pdf:
        st.download_button("Download PDF Report", data=make_pdf(result), file_name="voiceguard_report.pdf", mime="application/pdf", use_container_width=True)
    
    with col_json:
        json_data = json.dumps({k: v for k, v in result.items() if k != "raw_bytes"}, indent=2)
        st.download_button("Download JSON", data=json_data, file_name="voiceguard_data.json", mime="application/json", use_container_width=True)
    
    with col_csv:
        if result.get("metrics"):
            df_csv = pd.DataFrame(list(result["metrics"].items()), columns=["Metric", "Value"])
            st.download_button("Download CSV Metrics", data=df_csv.to_csv(index=False), file_name="voiceguard_metrics.csv", mime="text/csv", use_container_width=True)

st.divider()
st.header("🌐 TinyFish Web Search")
query = st.text_input("Search the web for AI voice cloning info:", placeholder="AI voice cloning detection news")
if st.button("Search with TinyFish"):
    if query.strip():
        with st.spinner("Searching..."):
            tinyfish_search(query.strip())
    else:
        st.warning("Please enter a query.")
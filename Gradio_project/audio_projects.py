import gradio as gr
import numpy as np
import traceback
from transformers import pipeline

# Optional resampling helper (if librosa installed)
try:
    import librosa
    HAVE_LIBROSA = True
except Exception:
    HAVE_LIBROSA = False

print("Loading models... (this may download them the first run)")
asr_model = pipeline("automatic-speech-recognition", model="openai/whisper-small", device="cpu")
emotion_model = pipeline("audio-classification", model="superb/hubert-large-superb-er", device="cpu")
print("Models loaded.")

def _to_mono_float32(sr, data):
    """Return (sr, mono_float32_array). Handles int16/stereo etc."""
    arr = np.asarray(data)
    # If shape is (n, channels) or (channels, n) handle common cases
    if arr.ndim == 2:
        # if shape (n_samples, channels) average channels
        if arr.shape[1] <= 2:
            arr = arr.mean(axis=1)
        else:
            arr = arr.mean(axis=1)
    # if it's (channels, n), transpose
    if arr.ndim == 2 and arr.shape[0] <= 2:
        arr = arr.mean(axis=0)
    # convert integer PCM to float in [-1, 1]
    if np.issubdtype(arr.dtype, np.integer):
        # assume int16
        maxv = np.iinfo(arr.dtype).max
        arr = arr.astype(np.float32) / maxv
    else:
        arr = arr.astype(np.float32)
    # ensure 1D
    arr = arr.reshape(-1)
    return sr, arr

def _maybe_resample(sr, arr, target_sr=16000):
    if sr == target_sr:
        return arr, sr
    if HAVE_LIBROSA:
        arr_res = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
        return arr_res.astype(np.float32), target_sr
    else:
        # fallback: warn but still try with original sr
        return arr, sr

def transcribe(audio):
    if audio is None:
        return "No audio detected"
    try:
        sr, data = audio  # Gradio gives (sr, numpy_array)
    except Exception as e:
        return f"Bad audio format from Gradio: {e}"

    # debug info
    try:
        print(f"[transcribe] incoming sr={sr}, dtype={getattr(data,'dtype',None)}, shape={np.asarray(data).shape}")
    except Exception:
        print("[transcribe] couldn't print incoming audio metadata")

    sr, arr = _to_mono_float32(sr, data)
    arr, sr = _maybe_resample(sr, arr, target_sr=16000)  # whisper commonly uses 16k
    print(f"[transcribe] after -> sr={sr}, dtype={arr.dtype}, shape={arr.shape}")

    try:
        result = asr_model({"array": arr, "sampling_rate": sr})
        # some pipeline may return dict with 'text' or 'chunks'; handle gracefully
        if isinstance(result, dict) and "text" in result:
            return result["text"]
        if isinstance(result, str):
            return result
        # else try first item
        if isinstance(result, (list, tuple)) and len(result):
            first = result[0]
            if isinstance(first, dict) and "text" in first:
                return first["text"]
        return str(result)
    except Exception as e:
        tb = traceback.format_exc()
        print("[transcribe] Exception:", tb)
        return f"Transcription error: {e}\n\nTraceback:\n{tb}"

def detect_emotion(audio):
    if audio is None:
        return "Please record audio first."
    try:
        sr, data = audio
    except Exception as e:
        return f"Bad audio format from Gradio: {e}"

    print(f"[emotion] incoming sr={sr}, dtype={getattr(data,'dtype',None)}, shape={np.asarray(data).shape}")
    sr, arr = _to_mono_float32(sr, data)
    arr, sr = _maybe_resample(sr, arr, target_sr=16000)
    print(f"[emotion] after -> sr={sr}, dtype={arr.dtype}, shape={arr.shape}")

    try:
        result = emotion_model({"array": arr, "sampling_rate": sr})
        # result usually list of {label, score}
        if isinstance(result, list) and len(result):
            top = result[0]
            return f"Emotion: {top.get('label')} (confidence: {round(top.get('score',0),3)})"
        return str(result)
    except Exception as e:
        tb = traceback.format_exc()
        print("[emotion] Exception:", tb)
        return f"Emotion detection error: {e}\n\nTraceback:\n{tb}"

def process_audio(audio):
    t = transcribe(audio)
    e = detect_emotion(audio)
    return t, e

with gr.Blocks() as demo:
    gr.Markdown("## 🎤 Speech Recognition + Emotion Detection (debug mode)")
    audio_input = gr.Audio(sources=["microphone"], type="numpy", label="Record your voice")
    text_output = gr.Textbox(label="Transcribed Text")
    emotion_output = gr.Textbox(label="Detected Emotion")
    btn = gr.Button("Process")
    btn.click(fn=process_audio, inputs=audio_input, outputs=[text_output, emotion_output])

demo.launch()

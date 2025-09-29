"""Realtime system-audio translator using faster-whisper and DeepSeek.

Install dependencies:
  pip install pyaudiowpatch faster-whisper==1.0.1 ctranslate2 numpy librosa \
              python-dotenv openai
  
Create a .env file with:
  DEEPSEEK_API_KEY=sk-...
"""

import os
import sys
import time
import queue
import threading
import numpy as np
import pyaudiowpatch as pyaudio
import librosa
from dotenv import load_dotenv
from faster_whisper import WhisperModel
import ctranslate2
import openai

# ----------------------------------------------------------------------
# 0) Environment & DeepSeek
# ----------------------------------------------------------------------

def configure_openai() -> None:
    """Load API key from .env and configure OpenAI/DeepSeek."""
    load_dotenv()
    openai.api_base = "https://api.deepseek.com/v1"
    openai.api_key = os.getenv("DEEPSEEK_API_KEY", "")
    openai.verify_ssl_certs = False
    if not openai.api_key:
        sys.exit("\u274c  .env file missing DEEPSEEK_API_KEY")

configure_openai()

# ----------------------------------------------------------------------
# 1) pyaudiowpatch Loopback stream
# ----------------------------------------------------------------------
SR = 16_000
CHUNK_SEC = 0.8
q_in: queue.Queue[bytes] = queue.Queue()

DEVICE_INDEX_ENV = os.getenv("LOOPBACK_DEVICE_INDEX")
DEVICE_INDEX = int(DEVICE_INDEX_ENV) if DEVICE_INDEX_ENV and DEVICE_INDEX_ENV.isdigit() else None


def audio_callback(in_data, frame_count, time_info, status):
    q_in.put(in_data)
    return (None, pyaudio.paContinue)


def list_audio_devices(pa: pyaudio.PyAudio) -> None:
    """Print available devices for debugging."""
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        host_api = pa.get_host_api_info_by_index(info["hostApi"])["name"]
        print(f"[{i}] {info.get('name')} - {host_api}")


pa = pyaudio.PyAudio()


def open_loopback_stream():
    try:
        if DEVICE_INDEX is not None:
            info = pa.get_device_info_by_index(DEVICE_INDEX)
        else:
            wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
            info = pa.get_device_info_by_host_api_device_index(
                wasapi["index"], wasapi["defaultOutputDevice"]
            )

        if not info.get("isLoopbackDevice", False):
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=min(info.get("maxInputChannels", 2), 2),
                rate=int(info.get("defaultSampleRate", SR)),
                input=True,
                frames_per_buffer=int(SR * CHUNK_SEC),
                input_device_index=info["index"],
                loopback=True,
                stream_callback=audio_callback,
            )
        else:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=min(info.get("maxInputChannels", 2), 2),
                rate=int(info.get("defaultSampleRate", SR)),
                input=True,
                frames_per_buffer=int(SR * CHUNK_SEC),
                input_device_index=info["index"],
                stream_callback=audio_callback,
            )
    except Exception as exc:
        print("Failed opening loopback stream:", exc)
        list_audio_devices(pa)
        raise

    return stream, info.get("defaultSampleRate", SR), info.get("maxInputChannels", 1), info


try:
    stream, SRC_SR, CH_IN, dev_info = open_loopback_stream()
    stream.start_stream()
except Exception as e:
    sys.exit(f"\u274c  Loopback stream failed: {e}")
print(f"\U0001f399  Loopback capture started ({dev_info.get('name')})")

# ----------------------------------------------------------------------
# 2) Load faster-whisper Large-v3 INT8
# ----------------------------------------------------------------------

device = "cuda" if ctranslate2.get_device_count("cuda") else "cpu"
ctype = "int8_float16" if device == "cuda" else "int8"
print(f"\U0001f50d  Loading Whisper model... ({device}, {ctype})")
model = WhisperModel("large-v3", device=device, compute_type=ctype)
print("\u2705  Whisper model ready")

# ----------------------------------------------------------------------
# 3) Utilities
# ----------------------------------------------------------------------

def to_mono_resample(raw_bytes: bytes, src_sr: int) -> np.ndarray:
    audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if CH_IN == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)
    if src_sr != SR:
        audio = librosa.resample(audio, src_sr, SR)
    return audio


def translate_stream(text: str) -> str:
    rsp = openai.ChatCompletion.create(
        model="deepseek-chat",
        stream=True,
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a translator. Translate to natural Korean."},
            {"role": "user", "content": text},
        ],
    )
    acc = []
    for ch in rsp:
        tok = ch.choices[0].delta.get("content")
        if tok:
            print(tok, end="", flush=True)
            acc.append(tok)
    print()
    return "".join(acc).strip()


# ----------------------------------------------------------------------
# 4) Worker - convert 0.8s chunks with Whisper and stream translation
# ----------------------------------------------------------------------

def worker() -> None:
    state = None
    while True:
        raw = q_in.get()
        wav = to_mono_resample(raw, SRC_SR)
        segs, state = model.transcribe_chunk(
            wav, state=state, vad_filter=True, language="en", beam_size=1
        )
        for s in segs:
            eng = s.text.strip()
            if not eng:
                continue
            print("\n[EN]", eng)
            try:
                translate_stream(eng)
            except Exception as exc:
                print("\u26d4  Translation error:", exc)


threading.Thread(target=worker, daemon=True).start()

print("\U0001f7e2  Streaming translation... (Ctrl-C to stop)")
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nStopping...")
finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()

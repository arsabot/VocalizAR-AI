import os
import sys
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import librosa
import soundfile as sf

# Forzar rutas de CosyVoice
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COSYVOICE_DIR = os.path.join(BASE_DIR, "CosyVoice")
MATCHA_DIR = os.path.join(COSYVOICE_DIR, "third_party", "Matcha-TTS")

if COSYVOICE_DIR not in sys.path:
    sys.path.insert(0, COSYVOICE_DIR)
if MATCHA_DIR not in sys.path:
    sys.path.insert(0, MATCHA_DIR)

# Modo offline estricto
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

print(f"CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Dispositivo GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

from cosyvoice.cli.cosyvoice import AutoModel

MODEL_DIR = os.path.join(BASE_DIR, "modelos", "Fun-CosyVoice3-0.5B-2512")
PROMPT_WAV_ORIGINAL = os.path.join(BASE_DIR, "voces", "Voz_Referencia_Original.wav")

TEMP_PROMPT = os.path.join(BASE_DIR, "temp_audios", "ref_curada_test.wav")
os.makedirs(os.path.dirname(TEMP_PROMPT), exist_ok=True)

y, sr = librosa.load(PROMPT_WAV_ORIGINAL, sr=24000, mono=True)
yt, _ = librosa.effects.trim(y, top_db=25)
max_len = int(18.0 * sr)
if len(yt) > max_len:
    yt = yt[:max_len]
max_v = max(abs(yt.max()), abs(yt.min()))
if max_v > 0:
    yt = yt / max_v * 0.95
sf.write(TEMP_PROMPT, yt, sr, subtype="PCM_16")
print(f"Muestra de voz de referencia curada: {len(yt)/sr:.2f}s guardada en {TEMP_PROMPT}")

print("\n--- Cargando CosyVoice 3 ---")
t0 = time.time()
cosyvoice = AutoModel(model_dir=MODEL_DIR, fp16=False)
t_load = time.time() - t0
print(f"Modelo cargado con éxito en {t_load:.2f}s")
if torch.cuda.is_available():
    print(f"VRAM en uso tras carga: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
    print(f"VRAM reservada: {torch.cuda.memory_reserved() / (1024**3):.2f} GB")

# Test 1: Inferencia cross-lingual
texto_prueba = "Hola, esta es una prueba técnica de CosyVoice 3 funcionando de manera completamente local y offline en la RTX 3090."
prompt_cross = f"You are a helpful assistant.<|endofprompt|>{texto_prueba}"

print("\n--- Generando con inference_cross_lingual ---")
t_gen0 = time.time()
outputs = []
for output in cosyvoice.inference_cross_lingual(prompt_cross, TEMP_PROMPT, stream=False, text_frontend=False):
    outputs.append(output['tts_speech'])

t_gen = time.time() - t_gen0
audio = torch.concat(outputs, dim=1)
dur_audio = audio.shape[1] / cosyvoice.sample_rate
rtf = t_gen / dur_audio
print(f"Audio generado en {t_gen:.2f}s ({dur_audio:.2f}s de voz | RTF: {rtf:.2f})")

out_wav = os.path.join(BASE_DIR, "temp_audios", "test_cosy3_cross_lingual.wav")
sf.write(out_wav, audio.squeeze(0).cpu().numpy(), cosyvoice.sample_rate)
print(f"Archivo cross-lingual guardado en: {out_wav}")

# Test 2: Inferencia zero-shot con prompt_text obtenido de la referencia
print("\n--- Transcribiendo referencia para zero-shot con Whisper local ---")
from faster_whisper import WhisperModel
whisper_model = WhisperModel("large-v3", device="cuda", compute_type="float16")
segments, _ = whisper_model.transcribe(TEMP_PROMPT, language="es")
ref_transcript = " ".join([s.text.strip() for s in segments]).strip()
print(f"Prompt transcript: \"{ref_transcript}\"")

full_prompt_text = f"You are a helpful assistant.<|endofprompt|>{ref_transcript}"

print("\n--- Generando con inference_zero_shot ---")
t_zs0 = time.time()
zs_outputs = []
for output in cosyvoice.inference_zero_shot(texto_prueba, full_prompt_text, TEMP_PROMPT, stream=False, text_frontend=False):
    zs_outputs.append(output['tts_speech'])

t_zs = time.time() - t_zs0
zs_audio = torch.concat(zs_outputs, dim=1)
dur_zs = zs_audio.shape[1] / cosyvoice.sample_rate
rtf_zs = t_zs / dur_zs
print(f"Audio zero-shot generado en {t_zs:.2f}s ({dur_zs:.2f}s de voz | RTF: {rtf_zs:.2f})")

out_zs_wav = os.path.join(BASE_DIR, "temp_audios", "test_cosy3_zero_shot.wav")
sf.write(out_zs_wav, zs_audio.squeeze(0).cpu().numpy(), cosyvoice.sample_rate)
print(f"Archivo zero-shot guardado en: {out_zs_wav}")

if torch.cuda.is_available():
    print(f"\nVRAM en uso tras generación: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")
    print(f"VRAM pico máxima: {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB")

print("\n¡Prueba completa finalizada con ÉXITO!")

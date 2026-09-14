import os
import sys
import time
import gc
import torch
import soundfile as sf
import librosa
from TTS.api import TTS
from TTS.utils.synthesizer import Synthesizer

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

TEXTO_BENCHMARK = (
    "Hola, ¿cómo estás? Hoy quiero contarte algo importante. "
    "A veces las cosas no salen como esperamos, pero eso no significa "
    "que tengamos que rendirnos. Lo importante es seguir avanzando, "
    "aprender de cada experiencia y encontrar una nueva manera de hacerlo."
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CARPETA_MODELO_AR = os.path.join(BASE_DIR, "modelos", "xtts_argentino")
MODELO_ESTANDAR = "tts_models/multilingual/multi-dataset/xtts_v2"
CARPETA_BENCHMARK = os.path.join(BASE_DIR, "benchmark_resultados")
os.makedirs(CARPETA_BENCHMARK, exist_ok=True)

VOZ_ORIGINAL = os.path.join(BASE_DIR, "voces", "Voz_Referencia_Original.wav")
if not os.path.exists(VOZ_ORIGINAL):
    VOZ_ORIGINAL = os.path.join(BASE_DIR, "voz_referencia.wav")

print("=========================================================")
print("🚀 INICIANDO BENCHMARK DE MOTORES TTS LOCALES / OFFLINE")
print("=========================================================")
print(f"Texto: '{TEXTO_BENCHMARK[:60]}...'")
print(f"Dispositivo: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Preprocesar audio de referencia para máxima fidelidad:
# Tomar un segmento de 8-12 segundos limpio y sin silencios largos
def preparar_muestra_optima(ruta_origen):
    y, sr = librosa.load(ruta_origen, sr=24000, mono=True)
    # Recortar silencios extremos
    yt, _ = librosa.effects.trim(y, top_db=25)
    # Tomar los mejores 10 segundos
    limite = int(10 * sr)
    if len(yt) > limite:
        # Tomar de 1s a 11s para evitar chasquido de inicio
        inicio = min(int(1.0 * sr), len(yt) - limite)
        muestra = yt[inicio : inicio + limite]
    else:
        muestra = yt
    # Normalizar
    max_val = max(abs(muestra.max()), abs(muestra.min()))
    if max_val > 0:
        muestra = muestra / max_val * 0.95
    ruta_optima = os.path.join(CARPETA_BENCHMARK, "ref_optimizada.wav")
    sf.write(ruta_optima, muestra, sr)
    return ruta_optima

ruta_ref_optima = preparar_muestra_optima(VOZ_ORIGINAL)
print(f"Muestra de voz optimizada generada en: {ruta_ref_optima}")

resultados = []

def probar_configuracion(nombre, motor, speaker_wav, rep_penalty=10.0, temp=0.75, top_p=0.85):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    salida_wav = os.path.join(CARPETA_BENCHMARK, f"{nombre}.wav")
    
    t0 = time.perf_counter()
    
    # Inferencia con parámetros específicos
    # Usar synthesizer directo o tts_to_file
    if getattr(motor, "synthesizer", None) is not None:
        # Synthesizer directo
        motor.tts_to_file(
            text=TEXTO_BENCHMARK,
            file_path=salida_wav,
            speaker_wav=speaker_wav,
            language="es",
            split_sentences=True,
            temperature=temp,
            repetition_penalty=rep_penalty,
            top_p=top_p,
        )
    else:
        motor.tts_to_file(
            text=TEXTO_BENCHMARK,
            file_path=salida_wav,
            speaker_wav=speaker_wav,
            language="es",
            split_sentences=True,
        )
    
    t_total = time.perf_counter() - t0
    
    # Medir duración de audio generado
    info = sf.info(salida_wav)
    duracion_audio = info.duration
    rtf = t_total / duracion_audio
    vram_pico_gb = torch.cuda.max_memory_allocated() / 1e9
    
    resultado = {
        "nombre": nombre,
        "tiempo_inf_s": round(t_total, 2),
        "duracion_audio_s": round(duracion_audio, 2),
        "rtf": round(rtf, 3),
        "vram_gb": round(vram_pico_gb, 2),
        "salida": salida_wav,
    }
    resultados.append(resultado)
    print(f"\n--- {nombre} ---")
    print(f"Tiempo Inferencia: {t_total:.2f} s | Duración Audio: {duracion_audio:.2f} s | RTF: {rtf:.3f}")
    print(f"VRAM Pico: {vram_pico_gb:.2f} GB | Archivo: {salida_wav}")
    return resultado

# 1. XTTS Estándar con parámetros por defecto (original)
print("\n[1/4] Probando XTTS-v2 Estándar (Parámetros por Defecto)...")
motor_estandar = TTS(MODELO_ESTANDAR, gpu=True)
probar_configuracion("1_xtts_estandar_default", motor_estandar, VOZ_ORIGINAL, rep_penalty=10.0, temp=0.75)

# 2. XTTS Estándar con Parámetros Optimizados para Prosodia y Naturalidad
print("\n[2/4] Probando XTTS-v2 Estándar (Prosodia Optimizada + Voz Curada)...")
probar_configuracion("2_xtts_estandar_optimizado", motor_estandar, ruta_ref_optima, rep_penalty=2.8, temp=0.68, top_p=0.85)

# Descargar motor estándar
del motor_estandar
gc.collect()
torch.cuda.empty_cache()

# 3. XTTS Argentino (MarianBasti) con Parámetros por Defecto
print("\n[3/4] Probando XTTS-v2 Argentino (Parámetros por Defecto)...")
motor_ar = TTS(progress_bar=False, gpu=True)
motor_ar.synthesizer = Synthesizer(model_dir=CARPETA_MODELO_AR, use_cuda=True)
motor_ar.model_name = "xtts_argentino"
probar_configuracion("3_xtts_argentino_default", motor_ar, VOZ_ORIGINAL, rep_penalty=10.0, temp=0.75)

# 4. XTTS Argentino con Parámetros Optimizados para Prosodia y Naturalidad
print("\n[4/4] Probando XTTS-v2 Argentino (Prosodia Optimizada + Voz Curada)...")
probar_configuracion("4_xtts_argentino_optimizado", motor_ar, ruta_ref_optima, rep_penalty=2.8, temp=0.68, top_p=0.85)

del motor_ar
gc.collect()
torch.cuda.empty_cache()

print("\n=========================================================")
print("📊 RESUMEN COMPARATIVO DEL BENCHMARK")
print("=========================================================")
print(f"{'Configuración':<32} | {'Tiempo (s)':<10} | {'Audio (s)':<10} | {'RTF':<8} | {'VRAM (GB)':<9}")
print("-" * 75)
for r in resultados:
    print(f"{r['nombre']:<32} | {r['tiempo_inf_s']:<10} | {r['duracion_audio_s']:<10} | {r['rtf']:<8} | {r['vram_gb']:<9}")

print("\n🎉 Benchmark finalizado exitosamente.")

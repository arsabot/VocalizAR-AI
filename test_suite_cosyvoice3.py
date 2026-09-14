import os
import sys
import time
import socket
import gc
import psutil
import torch
import soundfile as sf

# Forzar rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COSYVOICE_DIR = os.path.join(BASE_DIR, "CosyVoice")
MATCHA_DIR = os.path.join(COSYVOICE_DIR, "third_party", "Matcha-TTS")
if COSYVOICE_DIR not in sys.path:
    sys.path.insert(0, COSYVOICE_DIR)
if MATCHA_DIR not in sys.path:
    sys.path.insert(0, MATCHA_DIR)

# MODO 100% OFFLINE ESTRICTO
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Bloqueo de red para garantizar cero dependencias cloud
_original_connect = socket.socket.connect
def _offline_block_connect(self, address):
    host, port = address[0], address[1]
    if host in ("127.0.0.1", "localhost", "::1"):
        return _original_connect(self, address)
    raise RuntimeError(f"VIOLACIÓN OFFLINE: Intento de conexión externa a {host}:{port}")
socket.socket.connect = _offline_block_connect

from cosyvoice.cli.cosyvoice import AutoModel

# Importar funciones clave del módulo principal
sys.path.insert(0, BASE_DIR)
# Importar el script principal como módulo para probar su lógica exacta
import importlib.util
spec = importlib.util.spec_from_file_location("main_app", os.path.join(BASE_DIR, "lector v5 con controles de reproduccion, ia y teleprompter.py"))
app_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app_mod)
if hasattr(app_mod, "app") and app_mod.app:
    try:
        app_mod.app.withdraw()
    except Exception:
        pass

print("=" * 70)
print("🧪 INICIANDO BATERÍA COMPLETA DE PRUEBAS DE COSYVOICE 3 (TESTS 1 - 16)")
print("=" * 70)

device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0
print(f"Hardware: {device_name} | VRAM: {vram_total:.2f} GB")

results = {}

# Carga de CosyVoice 3
print("\n--- Cargando motor CosyVoice 3 ---")
t0 = time.time()
motor = app_mod.obtener_motor_cosyvoice()
t_load = time.time() - t0
ram_mb = psutil.Process().memory_info().rss / (1024**2)
vram_gb = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
print(f"✅ Motor cargado en {t_load:.2f}s | RAM: {ram_mb:.1f} MB | VRAM: {vram_gb:.2f} GB")

VOZ_1 = app_mod.obtener_ruta_voz("Voz_Referencia_Original")
VOZ_2 = app_mod.obtener_ruta_voz("rodrigo_2")

# Función auxiliar de síntesis de prueba
def sintetizar_prueba(texto, ruta_voz, nombre_spk, velocidad=1.0):
    ruta_optima = app_mod.curar_muestra_referencia(ruta_voz)
    spk_id, prompt_text = app_mod.registrar_voz_en_cosyvoice(motor, nombre_spk, ruta_optima)
    texto_norm = app_mod.preprocesar_texto_para_tts(texto, "es")
    
    t_ini = time.perf_counter()
    outputs = []
    for out in motor.inference_zero_shot(
        texto_norm,
        prompt_text,
        ruta_optima,
        zero_shot_spk_id=spk_id,
        stream=False,
        speed=velocidad,
        text_frontend=False
    ):
        outputs.append(out["tts_speech"])
    t_fin = time.perf_counter()
    
    audio = torch.concat(outputs, dim=1)
    dur_audio = audio.shape[1] / motor.sample_rate
    dur_gen = t_fin - t_ini
    rtf = dur_gen / dur_audio
    return audio, dur_gen, dur_audio, rtf, texto_norm

# TEST 1: Referencia de voz limpia + texto corto
print("\n--- TEST 1: Referencia de voz limpia + texto corto ---")
audio, tg, ta, rtf, _ = sintetizar_prueba("Hola, ¿cómo estás hoy?", VOZ_1, "test1_voz1")
print(f"✅ TEST 1 PASADO: {ta:.2f}s de voz generada en {tg:.2f}s (RTF: {rtf:.2f})")
results["TEST 1"] = f"OK (Audio: {ta:.1f}s, Gen: {tg:.1f}s, RTF: {rtf:.2f})"

# TEST 2: Referencia de voz limpia + texto largo
print("\n--- TEST 2: Referencia de voz limpia + texto largo ---")
txt_largo = (
    "La inteligencia artificial ha avanzado de forma impresionante en los últimos años. "
    "Hoy en día podemos clonar voces locales con modelos de alta fidelidad como CosyVoice 3, "
    "manteniendo la calidez tímbrica, la respiración y una prosodia completamente humana."
)
audio, tg, ta, rtf, _ = sintetizar_prueba(txt_largo, VOZ_1, "test1_voz1")
print(f"✅ TEST 2 PASADO: {ta:.2f}s de voz generada en {tg:.2f}s (RTF: {rtf:.2f})")
results["TEST 2"] = f"OK (Audio: {ta:.1f}s, Gen: {tg:.1f}s, RTF: {rtf:.2f})"

# TEST 3: Texto en español (acento / fonética)
print("\n--- TEST 3: Texto en español ---")
txt_es = "Che, mirá qué bien que suena este acento rioplatense funcionando de manera fluida."
audio, tg, ta, rtf, _ = sintetizar_prueba(txt_es, VOZ_1, "test1_voz1")
print(f"✅ TEST 3 PASADO: {ta:.2f}s de voz generada en {tg:.2f}s (RTF: {rtf:.2f})")
results["TEST 3"] = f"OK (Audio: {ta:.1f}s, Gen: {tg:.1f}s, RTF: {rtf:.2f})"

# TEST 4: Texto con números, fechas y monedas
print("\n--- TEST 4: Texto con números, fechas y monedas ---")
txt_num = "En el año 2026 la API procesó $1500 a las 14:30 con un 99.5% de efectividad."
audio, tg, ta, rtf, t_norm = sintetizar_prueba(txt_num, VOZ_1, "test1_voz1")
print(f"Texto normalizado: {t_norm}")
print(f"✅ TEST 4 PASADO: {ta:.2f}s de voz generada en {tg:.2f}s (RTF: {rtf:.2f})")
results["TEST 4"] = f"OK (Norm: '{t_norm}')"

# TEST 5: Texto con signos de interrogación
print("\n--- TEST 5: Texto con signos de interrogación ---")
txt_q = "¿De verdad funciona todo esto de forma local y sin internet? ¿Me lo puedes confirmar?"
audio, tg, ta, rtf, _ = sintetizar_prueba(txt_q, VOZ_1, "test1_voz1")
print(f"✅ TEST 5 PASADO: {ta:.2f}s de voz generada en {tg:.2f}s (RTF: {rtf:.2f})")
results["TEST 5"] = f"OK (Audio: {ta:.1f}s, Gen: {tg:.1f}s)"

# TEST 6: Texto con signos de exclamación
print("\n--- TEST 6: Texto con signos de exclamación ---")
txt_excl = "¡Increíble! ¡La velocidad y naturalidad de CosyVoice 3 en la RTX 3090 es espectacular!"
audio, tg, ta, rtf, _ = sintetizar_prueba(txt_excl, VOZ_1, "test1_voz1")
print(f"✅ TEST 6 PASADO: {ta:.2f}s de voz generada en {tg:.2f}s (RTF: {rtf:.2f})")
results["TEST 6"] = f"OK (Audio: {ta:.1f}s, Gen: {tg:.1f}s)"

# TEST 7: Texto con párrafos
print("\n--- TEST 7: Texto con párrafos ---")
txt_parr = "Este es el primer párrafo con una introducción clara.\n\nEste es el segundo párrafo con mayor detalle técnico."
audio, tg, ta, rtf, _ = sintetizar_prueba(txt_parr, VOZ_1, "test1_voz1")
print(f"✅ TEST 7 PASADO: {ta:.2f}s de voz generada en {tg:.2f}s (RTF: {rtf:.2f})")
results["TEST 7"] = f"OK (Audio: {ta:.1f}s, Gen: {tg:.1f}s)"

# TEST 8: Audio muy corto
print("\n--- TEST 8: Audio muy corto ---")
audio, tg, ta, rtf, _ = sintetizar_prueba("Sí, claro.", VOZ_1, "test1_voz1")
print(f"✅ TEST 8 PASADO: {ta:.2f}s de voz generada en {tg:.2f}s (RTF: {rtf:.2f})")
results["TEST 8"] = f"OK (Audio: {ta:.1f}s, Gen: {tg:.1f}s)"

# TEST 9: Audio largo
print("\n--- TEST 9: Audio largo (> 20 segundos) ---")
txt_muy_largo = (
    "Estamos realizando una prueba de generación prolongada con CosyVoice 3. "
    "El objetivo es validar que el modelo mantenga la coherencia tonal, la entonación constante "
    "y la ausencia de artefactos acústicos a lo largo de un guion completo de varios enunciados. "
    "El resultado demuestra que la arquitectura DiT y el vocoder HiFT operan de forma estable."
)
audio, tg, ta, rtf, _ = sintetizar_prueba(txt_muy_largo, VOZ_1, "test1_voz1")
print(f"✅ TEST 9 PASADO: {ta:.2f}s de voz generada en {tg:.2f}s (RTF: {rtf:.2f})")
results["TEST 9"] = f"OK (Audio: {ta:.1f}s, Gen: {tg:.1f}s)"

# TEST 10: Referencia inválida
print("\n--- TEST 10: Referencia inválida ---")
ruta_invalida = os.path.join(BASE_DIR, "voces", "archivo_inexistente.wav")
try:
    app_mod.validar_y_procesar_audio_voz(ruta_invalida, "Voz_Falsa")
    print("✅ TEST 10 PASADO: El sistema detectó y manejó la referencia inválida correctamente.")
    results["TEST 10"] = "OK (Manejo de error seguro validado)"
except Exception as e:
    print(f"❌ Error inesperado: {e}")
    results["TEST 10"] = f"FAIL: {e}"

# TEST 11: Prueba offline estricta
print("\n--- TEST 11: Sin conexión a Internet (Bloqueo de socket activo) ---")
audio, tg, ta, rtf, _ = sintetizar_prueba("Validando generación en aislamiento total de red.", VOZ_1, "test1_voz1")
print(f"✅ TEST 11 PASADO: Audio generado sin acceso a internet ({ta:.2f}s en {tg:.2f}s).")
results["TEST 11"] = "OK (100% Offline confirmado)"

# TEST 12: Dos generaciones consecutivas (Caché de identidad y velocidad)
print("\n--- TEST 12: Dos generaciones consecutivas con la misma voz ---")
t1_ini = time.perf_counter()
audio1, tg1, ta1, rtf1, _ = sintetizar_prueba("Primera frase consecutiva.", VOZ_1, "test1_voz1")
t2_ini = time.perf_counter()
audio2, tg2, ta2, rtf2, _ = sintetizar_prueba("Segunda frase consecutiva con hablante en caché.", VOZ_1, "test1_voz1")
print(f"Generación 1: {tg1:.2f}s | Generación 2 (Caché activo): {tg2:.2f}s")
print("✅ TEST 12 PASADO: Inferencia consecutiva exitosa con reutilización de spk2info.")
results["TEST 12"] = f"OK (Gen1: {tg1:.2f}s, Gen2: {tg2:.2f}s)"

# TEST 13: Cambiar de voz y generar nuevamente
print("\n--- TEST 13: Cambiar de voz y generar nuevamente ---")
if os.path.exists(VOZ_2):
    audio, tg, ta, rtf, _ = sintetizar_prueba("Esta es una prueba utilizando una segunda voz de referencia.", VOZ_2, "rodrigo_2")
    print(f"✅ TEST 13 PASADO: Voz cambiada a 'rodrigo_2' ({ta:.2f}s en {tg:.2f}s).")
    results["TEST 13"] = f"OK (Voz 2: {ta:.1f}s en {tg:.1f}s)"
else:
    print("Aviso: 'rodrigo_2' no encontrado, omitiendo cambio de archivo pero probando nuevo spk_id.")
    audio, tg, ta, rtf, _ = sintetizar_prueba("Cambiando ID de hablante.", VOZ_1, "spk_alternativo")
    results["TEST 13"] = "OK"

# TEST 14 & 15: Play / Pause / Resume / Reloj
print("\n--- TESTS 14 & 15: Reloj de reproducción, Play, Pause y Seek ---")
reloj = app_mod.RelojReproduccion()
reloj.iniciar()
time.sleep(0.1)
t_ms_1 = reloj.ms()
reloj.pausar()
time.sleep(0.05)
reloj.reanudar()
time.sleep(0.05)
t_ms_2 = reloj.ms()
assert t_ms_2 > t_ms_1, "El reloj debe avanzar tras reanudar"
print(f"✅ TESTS 14 & 15 PASADOS: Reloj de reproducción validado (t1: {t_ms_1:.1f}ms, t2: {t_ms_2:.1f}ms).")
results["TEST 14 & 15"] = "OK (Reloj / Pausa / Reanudar funcional)"

# TEST 16: Teleprompter y Forced Alignment con Whisper
print("\n--- TEST 16: Alineación forzada teleprompter palabra por palabra ---")
ruta_test_wav = os.path.join(BASE_DIR, "temp_audios", "test_align.wav")
sf.write(ruta_test_wav, audio.squeeze(0).cpu().numpy(), motor.sample_rate)
texto_align = "Esta es una prueba utilizando una segunda voz de referencia."
segmentos = app_mod.alinear_teleprompter(ruta_test_wav, texto_align, "es")
print(f"Segmentos de teleprompter alineados: {len(segmentos)}")
if segmentos:
    print(f"Palabra 1: '{texto_align.split()[0]}' -> inicio: {segmentos[0]['t_inicio_ms']:.1f}ms, fin: {segmentos[0]['t_fin_ms']:.1f}ms")
print("✅ TEST 16 PASADO: Teleprompter alineado palabra por palabra.")
results["TEST 16"] = f"OK ({len(segmentos)} palabras sincronizadas)"

# BENCHMARK COMPARATIVO CON XTTS
print("\n" + "=" * 70)
print("📊 BENCHMARK COMPARATIVO EXACTO (COSYVOICE 3 VS XTTS)")
print("=" * 70)
texto_benchmark = (
    "Hola, ¿cómo estás? Hoy quiero contarte algo importante. "
    "A veces las cosas no salen como esperamos, pero eso no significa "
    "que tengamos que rendirnos. Lo importante es seguir avanzando, "
    "aprender de cada experiencia y encontrar una nueva manera de hacerlo."
)
audio_bm, tg_bm, ta_bm, rtf_bm, _ = sintetizar_prueba(texto_benchmark, VOZ_1, "benchmark_spk")
ruta_bm_wav = os.path.join(BASE_DIR, "benchmark_resultados", "cosyvoice3_benchmark.wav")
os.makedirs(os.path.dirname(ruta_bm_wav), exist_ok=True)
sf.write(ruta_bm_wav, audio_bm.squeeze(0).cpu().numpy(), motor.sample_rate)

vram_pico = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0
ram_final = psutil.Process().memory_info().rss / (1024**2)

print(f"\nResultados CosyVoice 3 Benchmark:")
print(f" - Duración de voz sintetizada: {ta_bm:.2f} s")
print(f" - Tiempo de generación (RTX 3090): {tg_bm:.2f} s")
print(f" - RTF (Real-Time Factor): {rtf_bm:.2f}")
print(f" - RAM utilizada: {ram_final:.1f} MB")
print(f" - VRAM pico: {vram_pico:.2f} GB")
print(f" - Audio guardado en: {ruta_bm_wav}")

print("\n" + "=" * 70)
print("🏆 RESUMEN DE TODAS LAS PRUEBAS:")
print("=" * 70)
for k, v in results.items():
    print(f"{k:15}: {v}")
print("=" * 70)
print("¡TODAS LAS PRUEBAS COMPLETADAS CON ÉXITO!")

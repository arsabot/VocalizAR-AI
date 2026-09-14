import os
import sys

# Silenciar prompt de bienvenida de Pygame
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

# Modo 100% Offline: Desactivar comprobaciones de red en HuggingFace y Transformers
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Asegurar codificación UTF-8 en consola de Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Silenciar advertencias futuras, avisos de proveedores de ONNX y SDPA
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import pygame
import time
import shutil
import gc
import re
import bisect
from difflib import SequenceMatcher
from datetime import datetime

import numpy as np
import scipy.signal as signal
import torch
import torchaudio
import soundfile as sf
import librosa
from num2words import num2words

# =====================================================================
# 1. CONFIGURACIÓN INICIAL Y DIRECTORIOS LOCALES
# =====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Rutas de CosyVoice 3 y Matcha-TTS
COSYVOICE_DIR = os.path.join(BASE_DIR, "CosyVoice")
MATCHA_DIR = os.path.join(COSYVOICE_DIR, "third_party", "Matcha-TTS")
if COSYVOICE_DIR not in sys.path:
    sys.path.insert(0, COSYVOICE_DIR)
if MATCHA_DIR not in sys.path:
    sys.path.insert(0, MATCHA_DIR)

from cosyvoice.cli.cosyvoice import AutoModel

CARPETA_MODELO_COSY3 = os.path.join(BASE_DIR, "modelos", "Fun-CosyVoice3-0.5B-2512")
CARPETA_VOCES = os.path.join(BASE_DIR, "voces")
CARPETA_VOCES_OPT = os.path.join(CARPETA_VOCES, ".optimizadas")
CARPETA_TRANSCRIPCIONES = os.path.join(CARPETA_VOCES, ".transcripciones")
CARPETA_TEMP = os.path.join(BASE_DIR, "temp_audios")
CARPETA_GUARDADOS = os.path.join(BASE_DIR, "Audios_Guardados")

os.makedirs(CARPETA_VOCES, exist_ok=True)
os.makedirs(CARPETA_VOCES_OPT, exist_ok=True)
os.makedirs(CARPETA_TRANSCRIPCIONES, exist_ok=True)
os.makedirs(CARPETA_TEMP, exist_ok=True)
os.makedirs(CARPETA_GUARDADOS, exist_ok=True)

# Migrar voz de referencia inicial si existe
VOZ_REF_ORIGINAL = os.path.join(BASE_DIR, "voz_referencia.wav")
VOZ_DEFAULT_NOMBRE = "Voz_Referencia_Original"
VOZ_DEFAULT_PATH = os.path.join(CARPETA_VOCES, f"{VOZ_DEFAULT_NOMBRE}.wav")

if os.path.exists(VOZ_REF_ORIGINAL) and not os.path.exists(VOZ_DEFAULT_PATH):
    try:
        shutil.copy(VOZ_REF_ORIGINAL, VOZ_DEFAULT_PATH)
    except Exception as e:
        print(f"Aviso al migrar voz: {e}")

# Inicialización de Pygame Mixer con frecuencia adecuada para CosyVoice 3 (24 kHz)
try:
    pygame.mixer.quit()
except Exception:
    pass
pygame.mixer.init(frequency=24000, size=-16, channels=2, buffer=1024)
VOLUMEN_NORMAL = 1.0
volumen_actual = 1.0
audio_muteado = False
canal_audio = pygame.mixer.Channel(0)
audio_numpy_actual = None
sr_audio_actual = 24000
cambio_velocidad_solicitado = False
nueva_velocidad_solicitada = 1.0

def limpiar_archivos_temporales():
    """Elimina archivos temporales de audio de sesiones previas en raíz y temp_audios."""
    if os.path.exists(CARPETA_TEMP):
        for nombre in os.listdir(CARPETA_TEMP):
            if nombre.endswith(".wav"):
                try:
                    os.remove(os.path.join(CARPETA_TEMP, nombre))
                except Exception:
                    pass
    for nombre in os.listdir(BASE_DIR):
        if nombre.startswith("temp_audio_") and nombre.endswith(".wav"):
            try:
                os.remove(os.path.join(BASE_DIR, nombre))
            except Exception:
                pass

limpiar_archivos_temporales()

# =====================================================================
# 2. ESTADO GLOBAL Y CONFIGURACIONES DE PROSODIA NATURAL
# =====================================================================
estado_audio = "detenido"  # "detenido", "generando", "reproduciendo", "pausado"
reiniciar_solicitado = False
sesion_reproduccion_id = 0

archivo_actual = None
texto_actual = ""
idioma_actual = ""
modelo_actual = "estandar"
voz_actual = VOZ_DEFAULT_NOMBRE
estilo_actual = "conversacional"
velocidad_generacion_actual = 1.0
velocidad_reproduccion_actual = 1.0

segmentos_teleprompter = []
reloj_reproduccion = None
duracion_audio_total_ms = 0.0
tamano_fuente_teleprompter = 22

# Configuración de parámetros de prosodia humana y naturalidad (calibrados en benchmark)
ESTILOS_PROSODIA = {
    "conversacional": {
        "etiqueta": "🎭 Conversacional (Natural)",
        "temperature": 0.68,
        "repetition_penalty": 2.8,
        "top_p": 0.85,
        "length_penalty": 1.0,
        "descripcion": "Entonación cálida, ritmo fluido y pausas humanas de respiración.",
    },
    "locutor": {
        "etiqueta": "🎙️ Locutor (Dinámico)",
        "temperature": 0.72,
        "repetition_penalty": 2.4,
        "top_p": 0.88,
        "length_penalty": 1.0,
        "descripcion": "Mayor rango tonal, énfasis expresivo y ritmo enérgico.",
    },
    "audiolibro": {
        "etiqueta": "📖 Audiolibro (Calmo)",
        "temperature": 0.60,
        "repetition_penalty": 3.2,
        "top_p": 0.80,
        "length_penalty": 1.05,
        "descripcion": "Cadencia pausada, lectura clara y solemne.",
    },
}

# Locks de concurrencia
bloqueo_motor = threading.Lock()
bloqueo_sintesis = threading.Lock()
bloqueo_whisper = threading.Lock()
bloqueo_voces = threading.Lock()

motor_cosyvoice = None
modelo_whisper = None

# Caché en memoria de muestras optimizadas
cache_muestras_optimas = {}

# Traducción en vivo (opcional)
job_traduccion_id = None
traduccion_seq = 0

WHISPER_POR_IDIOMA = {
    "es": "es",
    "es-ar": "es",
    "en": "en",
    "fr": "fr",
    "it": "it",
    "de": "de",
    "pt": "pt",
    "ru": "ru",
    "ja": "ja",
    "ko": "ko",
    "zh": "zh",
    "zh-cn": "zh",
}

IDIOMAS_VOZ_OPCIONES = [
    "AUTO - Auto-detectar idioma",
    "ES - Español (Estándar / Neutro)",
    "ES-AR - Español (Argentina / Rioplatense)",
    "EN - Inglés",
    "PT - Portugués",
    "FR - Francés",
    "IT - Italiano",
    "DE - Alemán",
    "RU - Ruso",
    "JA - Japonés",
    "KO - Coreano",
    "ZH - Chino (Mandarín)",
]

MAPA_IDIOMAS = {
    "es": ("es", "cosyvoice3"),
    "es-ar": ("es", "cosyvoice3"),
    "en": ("en", "cosyvoice3"),
    "pt": ("pt", "cosyvoice3"),
    "fr": ("fr", "cosyvoice3"),
    "it": ("it", "cosyvoice3"),
    "de": ("de", "cosyvoice3"),
    "ru": ("ru", "cosyvoice3"),
    "ja": ("ja", "cosyvoice3"),
    "ko": ("ko", "cosyvoice3"),
    "zh": ("zh", "cosyvoice3"),
    "zh-cn": ("zh", "cosyvoice3"),
}

CODIGO_A_ETIQUETA = {
    "es": "ES - Español (Estándar / Neutro)",
    "es-ar": "ES-AR - Español (Argentina / Rioplatense)",
    "en": "EN - Inglés",
    "pt": "PT - Portugués",
    "fr": "FR - Francés",
    "it": "IT - Italiano",
    "de": "DE - Alemán",
    "ru": "RU - Ruso",
    "ja": "JA - Japonés",
    "ko": "KO - Coreano",
    "zh": "ZH - Chino (Mandarín)",
    "zh-cn": "ZH - Chino (Mandarín)",
}

DETECTADO_A_CODIGO = {
    "es": "es",
    "en": "en",
    "pt": "pt",
    "fr": "fr",
    "it": "it",
    "de": "de",
    "ru": "ru",
    "ja": "ja",
    "ko": "ko",
    "zh": "zh",
    "zh-cn": "zh",
    "zh-tw": "zh",
}

IDIOMAS_TRADUCCION = [
    ("ES - Español", "es"),
    ("EN - Inglés", "en"),
    ("PT - Portugués", "pt"),
    ("FR - Francés", "fr"),
    ("IT - Italiano", "it"),
    ("DE - Alemán", "de"),
    ("RU - Ruso", "ru"),
    ("JA - Japonés", "ja"),
    ("KO - Coreano", "ko"),
    ("ZH - Chino (simplificado)", "zh-CN"),
]


# =====================================================================
# 3. MÓDULO NLP: PREPROCESAMIENTO DE TEXTO Y PROSODIA EN ESPAÑOL
# =====================================================================
ABREVIATURAS_ES = {
    r"\bDr\.(?=\s|$)": "Doctor",
    r"\bDra\.(?=\s|$)": "Doctora",
    r"\bSr\.(?=\s|$)": "Señor",
    r"\bSra\.(?=\s|$)": "Señora",
    r"\bSrta\.(?=\s|$)": "Señorita",
    r"\bej\.(?=\s|$)": "por ejemplo",
    r"\betc\.(?=\s|$)": "etcétera",
    r"\bpág\.(?=\s|$)": "página",
    r"\bpágs\.(?=\s|$)": "páginas",
    r"\bEE\.UU\.(?=\s|$)": "Estados Unidos",
    r"\bvs\.(?=\s|$)": "versus",
    r"\bart\.(?=\s|$)": "artículo",
    r"\bProf\.(?=\s|$)": "Profesor",
    r"\bProfa\.(?=\s|$)": "Profesora",
    r"\bkm\b": "kilómetros",
    r"\bkg\b": "kilogramos",
    r"\bmin\b": "minutos",
    r"\bh\b": "horas",
}

SIGLAS_ES = {
    r"\bAPI\b": "a-pe-i",
    r"\bAPIs\b": "a-pe-is",
    r"\bIA\b": "i-a",
    r"\bTTS\b": "te-te-ese",
    r"\bPDF\b": "pe-de-efe",
    r"\bGPS\b": "ge-pe-ese",
    r"\bCPU\b": "ce-pe-u",
    r"\bGPU\b": "ge-pe-u",
    r"\bHTML\b": "hache-te-eme-ele",
    r"\bURL\b": "u-erre-ele",
    r"\bUI\b": "u-i",
    r"\bUX\b": "u-quis",
    r"\bUSB\b": "u-ese-be",
    r"\bRAM\b": "ram",
    r"\bVRAM\b": "ve-ram",
    r"\bRTX\b": "erre-te-quis",
    r"\bWiFi\b": "uai-fai",
    r"\bWi-Fi\b": "uai-fai",
}

MESES_ES = {
    1: "enero", 2: "febrero", 3: "marzo", 4: "abril",
    5: "mayo", 6: "junio", 7: "julio", 8: "agosto",
    9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre"
}

def preprocesar_texto_para_tts(texto, idioma="es"):
    """
    Normaliza el texto para que la voz generada por CosyVoice 3 suene con máxima naturalidad:
    - Expande abreviaturas en palabras completas.
    - Convierte siglas técnicas a pronunciación fonética clara.
    - Interpreta años (2026 -> dos mil veintiséis) y fechas completas.
    - Convierte monedas ($1500 -> mil quinientos pesos, USD -> dólares, etc.).
    - Convierte números y porcentajes con decimales.
    - Regula puntuación para generar pausas y respiraciones prosódicas.
    """
    if not texto or not texto.strip():
        return ""

    t = texto.strip()

    if idioma in ("es", "es-ar"):
        # 1. Expandir abreviaturas
        for patron, expansion in ABREVIATURAS_ES.items():
            t = re.sub(patron, expansion, t, flags=re.IGNORECASE)

        # 2. Siglas y términos tecnológicos comunes
        for patron, expansion in SIGLAS_ES.items():
            t = re.sub(patron, expansion, t)

        # 3. Fechas en formato DD/MM/AAAA o DD-MM-AAAA
        def reemplazar_fecha(m):
            dia = int(m.group(1))
            mes = int(m.group(2))
            ano = int(m.group(3))
            if 1 <= dia <= 31 and 1 <= mes <= 12:
                txt_dia = num2words(dia, lang="es")
                txt_mes = MESES_ES.get(mes, str(mes))
                txt_ano = num2words(ano, lang="es")
                return f"{txt_dia} de {txt_mes} de {txt_ano}"
            return m.group(0)

        t = re.sub(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b", reemplazar_fecha, t)

        # 4. Monedas: $1500, USD 100, US$ 50, €20
        def reemplazar_pesos(m):
            val = m.group(1).replace(".", "").replace(",", ".")
            try:
                num = float(val) if "." in val else int(val)
                txt = num2words(num, lang="es")
                return f"{txt} pesos"
            except Exception:
                return m.group(0)

        t = re.sub(r"\$\s*(\d+(?:[.,]\d+)?)", reemplazar_pesos, t)

        def reemplazar_dolares(m):
            val = m.group(1).replace(".", "").replace(",", ".")
            try:
                num = float(val) if "." in val else int(val)
                txt = num2words(num, lang="es")
                return f"{txt} dólares"
            except Exception:
                return m.group(0)

        t = re.sub(r"(?:USD|US\$)\s*(\d+(?:[.,]\d+)?)", reemplazar_dolares, t, flags=re.IGNORECASE)

        def reemplazar_euros(m):
            val = m.group(1).replace(".", "").replace(",", ".")
            try:
                num = float(val) if "." in val else int(val)
                txt = num2words(num, lang="es")
                return f"{txt} euros"
            except Exception:
                return m.group(0)

        t = re.sub(r"€\s*(\d+(?:[.,]\d+)?)", reemplazar_euros, t)

        # 5. Porcentajes (ej: 25% o 99.5%)
        def reemplazar_porcentaje(m):
            val = m.group(1).replace("%", "").strip()
            if "." in val or "," in val:
                partes = re.split(r"[.,]", val)
                try:
                    txt_entero = num2words(int(partes[0]), lang="es")
                    txt_dec = num2words(int(partes[1]), lang="es")
                    return f"{txt_entero} coma {txt_dec} por ciento"
                except Exception:
                    return f"{val} por ciento"
            try:
                num = int(val)
                return f"{num2words(num, lang='es')} por ciento"
            except Exception:
                return f"{val} por ciento"

        t = re.sub(r"(\d+(?:[.,]\d+)?)\s*%", reemplazar_porcentaje, t)

        # 6. Horas (ej: 14:30 -> catorce y treinta)
        def reemplazar_hora(m):
            h = int(m.group(1))
            minutos = int(m.group(2))
            try:
                txt_h = num2words(h, lang="es")
                txt_min = num2words(minutos, lang="es") if minutos > 0 else "en punto"
                return f"{txt_h} y {txt_min}" if minutos > 0 else f"{txt_h} {txt_min}"
            except Exception:
                return m.group(0)

        t = re.sub(r"\b(\d{1,2}):(\d{2})\b", reemplazar_hora, t)

        # 7. Decimales con coma (ej: 15,5 -> quince coma cinco)
        def reemplazar_decimal(m):
            entero = m.group(1).replace(".", "")
            decimal = m.group(2)
            try:
                txt_entero = num2words(int(entero), lang="es")
                txt_decimal = num2words(int(decimal), lang="es")
                return f"{txt_entero} coma {txt_decimal}"
            except Exception:
                return m.group(0)

        t = re.sub(r"\b(\d{1,3}(?:\.\d{3})*),(\d+)\b", reemplazar_decimal, t)

        # 8. Años y números enteros aislados (ej: 2026 -> dos mil veintiséis)
        def reemplazar_entero(m):
            val = m.group(1).replace(".", "")
            try:
                num = int(val)
                if num < 1000000000:
                    return num2words(num, lang="es")
                return val
            except Exception:
                return val

        t = re.sub(r"\b(\d{1,3}(?:\.\d{3})*|\d+)\b", reemplazar_entero, t)

        # 9. Normalizar puntuación prosódica para pausas naturales
        t = re.sub(r",([^\s])", r", \1", t)
        t = re.sub(r"\.([^\s\.])", r". \1", t)
        t = re.sub(r"!([^\s!])", r"! \1", t)
        t = re.sub(r"\?([^\s\?])", r"? \1", t)
        t = re.sub(r"[-—]\s*", " — ", t)
        t = re.sub(r"[ \t]+", " ", t)

    return t


def dividir_texto_para_sintesis(texto, max_palabras=40):
    """
    Divide el texto en fragmentos semánticamente coherentes (por párrafos y oraciones)
    para que la inferencia autorregresiva de CosyVoice 3 procese textos de cualquier longitud
    sin sufrir truncamiento prematuro de la ventana de atención ni cortes acústicos.
    """
    if not texto or not texto.strip():
        return []

    # 1. Separar por párrafos (saltos de línea)
    lineas = [p.strip() for p in re.split(r'\n+', texto) if p.strip()]

    fragmentos = []
    for linea in lineas:
        palabras_linea = linea.split()
        if len(palabras_linea) <= max_palabras:
            fragmentos.append(linea)
        else:
            # Si el párrafo excede el tamaño óptimo, dividir por oraciones (. ? ! ; :)
            oraciones = re.split(r'(?<=[.?!;:])\s+', linea)
            buffer = ""
            for o in oraciones:
                o = o.strip()
                if not o:
                    continue
                if not buffer:
                    buffer = o
                elif len((buffer + " " + o).split()) <= max_palabras:
                    buffer += " " + o
                else:
                    fragmentos.append(buffer)
                    buffer = o
            if buffer:
                # Si una sola oración todavía supera max_palabras, dividir por comas
                if len(buffer.split()) > max_palabras:
                    sub_partes = re.split(r'(?<=[,])\s+', buffer)
                    sub_buf = ""
                    for sp in sub_partes:
                        sp = sp.strip()
                        if not sp:
                            continue
                        if not sub_buf:
                            sub_buf = sp
                        elif len((sub_buf + " " + sp).split()) <= max_palabras:
                            sub_buf += " " + sp
                        else:
                            fragmentos.append(sub_buf)
                            sub_buf = sp
                    if sub_buf:
                        fragmentos.append(sub_buf)
                else:
                    fragmentos.append(buffer)

    return fragmentos


# =====================================================================
# 4. CURACIÓN ACÚSTICA DE MUESTRAS DE VOZ (CLONACIÓN DE ALTA FIDELIDAD)
# =====================================================================
def curar_muestra_referencia(ruta_origen):
    """
    Toma un archivo de voz y genera una muestra de referencia acústicamente óptima para CosyVoice 3:
    - Recorta silencios iniciales y finales.
    - Limita la ventana a un máximo de 22 segundos para cumplir con el límite estricto
      de extracción de tokens de voz de CosyVoice (<= 30s) sin mutilar la referencia.
    - Normaliza la amplitud a 0.95 de pico para evitar saturación en el vocoder HiFT.
    - Almacena en carpeta .optimizadas para reutilización rápida.
    """
    if not os.path.exists(ruta_origen):
        return ruta_origen

    nombre_base = os.path.splitext(os.path.basename(ruta_origen))[0]
    ruta_optima = os.path.join(CARPETA_VOCES_OPT, f"{nombre_base}_opt.wav")

    # Si ya existe y es más reciente que el origen, retornar desde caché
    if os.path.exists(ruta_optima) and os.path.getmtime(ruta_optima) >= os.path.getmtime(ruta_origen):
        return os.path.abspath(ruta_optima)

    try:
        y, sr = librosa.load(ruta_origen, sr=24000, mono=True)

        # Recortar silencios extremos
        yt, _ = librosa.effects.trim(y, top_db=25)

        # Limitar a máximo 22 segundos (para dar margen seguro bajo el límite de 30s de CosyVoice)
        limite_dur = int(22.0 * sr)
        if len(yt) > limite_dur:
            inicio = min(int(0.5 * sr), len(yt) - limite_dur)
            muestra = yt[inicio : inicio + limite_dur]
        else:
            muestra = yt

        # Normalizar amplitud
        max_val = max(abs(muestra.max()), abs(muestra.min()))
        if max_val > 0:
            muestra = muestra / max_val * 0.95

        sf.write(ruta_optima, muestra, sr, subtype="PCM_16")
        return os.path.abspath(ruta_optima)
    except Exception as e:
        print(f"Aviso al curar muestra de voz: {e}")
        return os.path.abspath(ruta_origen)


def obtener_transcripcion_voz(ruta_audio_optima, avisar=None):
    """
    Obtiene o transcribe con Whisper local el texto hablado en la muestra de referencia.
    Lo almacena en .transcripciones/ para evitar re-transcripción en futuras ejecuciones.
    """
    nombre_base = os.path.splitext(os.path.basename(ruta_audio_optima))[0]
    ruta_txt = os.path.join(CARPETA_TRANSCRIPCIONES, f"{nombre_base}.txt")
    if os.path.exists(ruta_txt):
        try:
            with open(ruta_txt, "r", encoding="utf-8") as f:
                contenido = f.read().strip()
                if contenido:
                    return contenido
        except Exception:
            pass

    try:
        if avisar:
            avisar("🎙️ Extrayendo transcripción de referencia con Whisper local...")
        modelo_w = obtener_modelo_whisper()
        segmentos, _ = modelo_w.transcribe(
            os.path.abspath(ruta_audio_optima),
            language="es",
            vad_filter=False,
            beam_size=5,
            temperature=0.0,
        )
        texto = " ".join(seg.text.strip() for seg in segmentos).strip()
        if not texto:
            texto = "Hola, esta es mi voz de referencia para clonación."
        with open(ruta_txt, "w", encoding="utf-8") as f:
            f.write(texto)
        return texto
    except Exception as e:
        print(f"Aviso al transcribir referencia: {e}")
        return "Hola, esta es mi voz de referencia para clonación."


def registrar_voz_en_cosyvoice(motor, nombre_voz, ruta_audio_optima, avisar=None):
    """
    Registra la voz de referencia en la memoria de CosyVoice 3 (add_zero_shot_spk).
    Extrae tokens de habla y embedding CamP++ una única vez y los mantiene en VRAM.
    """
    spk_id = f"spk_{nombre_voz}"
    ruta_abs = os.path.abspath(ruta_audio_optima)
    if spk_id not in motor.frontend.spk2info:
        if avisar:
            avisar(f"🎙️ Registrando identidad acústica de '{nombre_voz}' en VRAM...")
        transcripcion = obtener_transcripcion_voz(ruta_abs, avisar=avisar)
        prompt_text = f"You are a helpful assistant.<|endofprompt|>{transcripcion}"
        motor.add_zero_shot_spk(prompt_text, ruta_abs, spk_id)
    else:
        transcripcion = obtener_transcripcion_voz(ruta_abs, avisar=None)
        prompt_text = f"You are a helpful assistant.<|endofprompt|>{transcripcion}"
    return spk_id, prompt_text


def listar_voces_disponibles():
    with bloqueo_voces:
        voces = []
        if os.path.exists(CARPETA_VOCES):
            for f in os.listdir(CARPETA_VOCES):
                if f.lower().endswith(".wav"):
                    voces.append(os.path.splitext(f)[0])
        if not voces:
            voces = [VOZ_DEFAULT_NOMBRE]
        return sorted(voces)


def obtener_ruta_voz(nombre_voz):
    ruta = os.path.join(CARPETA_VOCES, f"{nombre_voz}.wav")
    if os.path.exists(ruta):
        return os.path.abspath(ruta)
    if os.path.exists(VOZ_DEFAULT_PATH):
        return os.path.abspath(VOZ_DEFAULT_PATH)
    if os.path.exists(VOZ_REF_ORIGINAL):
        return os.path.abspath(VOZ_REF_ORIGINAL)
    return None


def validar_y_procesar_audio_voz(ruta_origen, nombre_voz):
    nombre_limpio = "".join(c for c in nombre_voz if c.isalnum() or c in " _-").strip().replace(" ", "_")
    if not nombre_limpio:
        return False, "El nombre de la voz no es válido.", 0.0

    ruta_destino = os.path.join(CARPETA_VOCES, f"{nombre_limpio}.wav")

    try:
        y, sr = librosa.load(ruta_origen, sr=24000, mono=True)
        duracion = len(y) / sr

        if duracion < 2.0:
            return False, f"El audio es demasiado corto ({duracion:.1f}s). Se recomiendan al menos 5 segundos.", duracion

        if duracion > 60.0:
            y = y[: int(60.0 * sr)]
            duracion = 60.0

        max_val = max(abs(y.max()), abs(y.min()))
        if max_val > 0:
            y = y / max_val * 0.95

        sf.write(ruta_destino, y, sr, subtype="PCM_16")
        curar_muestra_referencia(ruta_destino)
        return True, f"Voz '{nombre_limpio}' guardada y optimizada ({duracion:.1f}s).", duracion

    except Exception as e:
        return False, f"Error al procesar archivo de audio: {e}", 0.0


# =====================================================================
# 5. CARGA DE MODELOS COSYVOICE 3 Y WHISPER LOCALES EN VRAM
# =====================================================================
def liberar_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def obtener_motor_cosyvoice(avisar=None):
    global motor_cosyvoice
    with bloqueo_motor:
        if motor_cosyvoice is None:
            if avisar:
                avisar("⏳ Cargando CosyVoice 3 (0.5B Multilingual) en NVIDIA RTX 3090...")
            print("⏳ Inicializando CosyVoice 3 en GPU...")
            use_fp16 = torch.cuda.is_available()
            motor_cosyvoice = AutoModel(model_dir=CARPETA_MODELO_COSY3, fp16=use_fp16)
            print("✅ CosyVoice 3 cargado con éxito en VRAM.")
            if avisar:
                avisar("✅ CosyVoice 3 listo en GPU.")
        return motor_cosyvoice


def obtener_modelo_whisper():
    global modelo_whisper
    with bloqueo_whisper:
        if modelo_whisper is None:
            from faster_whisper import WhisperModel
            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute = "float16" if device == "cuda" else "int8"
            print("⏳ Cargando Whisper large-v3 local para teleprompter de alta precisión...")
            modelo_whisper = WhisperModel("large-v3", device=device, compute_type=compute)
            print("✅ Whisper large-v3 listo.")
        return modelo_whisper


# Inicialización en VRAM
print("=====================================================")
print("⏳ Inicializando CosyVoice 3 en NVIDIA GeForce RTX 3090...")
print("=====================================================")
try:
    motor_cosyvoice = obtener_motor_cosyvoice()
    print("✅ ¡CosyVoice 3 cargado con éxito en VRAM!")
except Exception as e:
    print(f"⚠️ Aviso al iniciar CosyVoice 3: {e}")
    motor_cosyvoice = None


# =====================================================================
# 6. RELOJ, SINCRONIZACIÓN Y TELEPROMPTER
# =====================================================================
def crear_sound_resampleado(audio_np, sr, velocidad, offset_ms=0.0):
    """
    Toma un array de audio float32 y devuelve un pygame.mixer.Sound ajustado a la velocidad
    especificada a partir de offset_ms PRESERVANDO EL 100% DEL TONO ORIGINAL (sin efecto ardilla).
    """
    if audio_np is None or len(audio_np) == 0:
        return None

    offset_samples = max(0, min(len(audio_np), int((offset_ms / 1000.0) * sr)))
    sub_audio = audio_np[offset_samples:]
    if len(sub_audio) == 0:
        return None

    v = max(0.4, min(2.5, float(velocidad)))
    if abs(v - 1.0) < 0.01:
        resampled = sub_audio
    else:
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            n_fft = 1024
            hop_length = 256
            window = torch.hann_window(n_fft, device=device)

            if isinstance(sub_audio, np.ndarray):
                audio_t = torch.from_numpy(sub_audio).float()
            else:
                audio_t = sub_audio.float()

            if audio_t.ndim == 1:
                audio_t = audio_t.unsqueeze(0)
            audio_t = audio_t.to(device)

            time_stretch = torchaudio.transforms.TimeStretch(
                n_freq=n_fft // 2 + 1,
                hop_length=hop_length,
                fixed_rate=v
            ).to(device)

            stft_spec = torch.stft(
                audio_t,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window,
                return_complex=True
            )
            stretched_spec = time_stretch(stft_spec, v)
            stretched_audio = torch.istft(
                stretched_spec,
                n_fft=n_fft,
                hop_length=hop_length,
                window=window
            )
            resampled = stretched_audio.squeeze(0).cpu().numpy()
        except Exception as e:
            # Fallback en caso de error en GPU/transform
            print(f"Aviso TimeStretch: {e}")
            up = 100
            down = max(1, int(round(100 * v)))
            resampled = signal.resample_poly(sub_audio, up, down)

    int16_audio = (np.clip(resampled, -1.0, 1.0) * 32767).astype(np.int16)
    if int16_audio.ndim == 1:
        stereo_audio = np.column_stack([int16_audio, int16_audio])
    elif int16_audio.shape[1] == 1:
        stereo_audio = np.repeat(int16_audio, 2, axis=1)
    else:
        stereo_audio = int16_audio

    return pygame.sndarray.make_sound(np.ascontiguousarray(stereo_audio))


class RelojReproduccion:
    """
    Reloj de alta precisión para teleprompter que soporta cambios dinámicos
    de velocidad en caliente durante la reproducción en curso.
    """
    def __init__(self, velocidad_inicial=1.0):
        self.velocidad = max(0.2, float(velocidad_inicial))
        self.t_virtual_base = 0.0      # ms en tiempo de audio original
        self.t_real_base = None        # time.perf_counter() en el último cambio/inicio
        self.pausado_desde = None
        self.acumulado_pausa = 0.0

    def iniciar(self, t_inicio_virtual=0.0, velocidad=None):
        if velocidad is not None:
            self.velocidad = max(0.2, float(velocidad))
        self.t_virtual_base = float(t_inicio_virtual)
        self.t_real_base = time.perf_counter()
        self.pausado_desde = None
        self.acumulado_pausa = 0.0

    def pausar(self):
        if self.t_real_base is not None and self.pausado_desde is None:
            self.pausado_desde = time.perf_counter()

    def reanudar(self):
        if self.pausado_desde is not None:
            self.acumulado_pausa += time.perf_counter() - self.pausado_desde
            self.pausado_desde = None

    def cambiar_velocidad(self, nueva_velocidad):
        if self.t_real_base is None:
            self.velocidad = max(0.2, float(nueva_velocidad))
            return
        t_actual_virtual = self.ms()
        self.t_virtual_base = t_actual_virtual
        self.t_real_base = time.perf_counter()
        self.pausado_desde = self.t_real_base if self.pausado_desde is not None else None
        self.acumulado_pausa = 0.0
        self.velocidad = max(0.2, float(nueva_velocidad))

    def reiniciar(self):
        self.iniciar(0.0, self.velocidad)

    def ms(self):
        if self.t_real_base is None:
            return 0.0
        if self.pausado_desde is not None:
            t_real_efectivo = self.pausado_desde - self.t_real_base - self.acumulado_pausa
        else:
            t_real_efectivo = time.perf_counter() - self.t_real_base - self.acumulado_pausa
        return self.t_virtual_base + (t_real_efectivo * self.velocidad * 1000.0)


def formatear_tiempo_ms(ms):
    segundos_totales = max(0, int(ms / 1000.0))
    minutos = segundos_totales // 60
    segundos = segundos_totales % 60
    return f"{minutos:02d}:{segundos:02d}"


def normalizar_palabra(palabra):
    return re.sub(r"[^\w]", "", palabra, flags=re.UNICODE).lower()


def construir_indices_texto_multilinea(texto, palabras):
    indices = []
    cursor = 0
    for palabra in palabras:
        pos = texto.find(palabra, cursor)
        if pos == -1:
            pos = cursor
        fin = pos + len(palabra)
        cursor = fin

        sub_ant = texto[:pos]
        linea_ini = sub_ant.count("\n") + 1
        ultimo_nl = sub_ant.rfind("\n")
        col_ini = pos if ultimo_nl == -1 else pos - (ultimo_nl + 1)

        sub_fin = texto[:fin]
        linea_fin = sub_fin.count("\n") + 1
        ultimo_nl_fin = sub_fin.rfind("\n")
        col_fin = fin if ultimo_nl_fin == -1 else fin - (ultimo_nl_fin + 1)

        indices.append((f"{linea_ini}.{col_ini}", f"{linea_fin}.{col_fin}"))
    return indices


def _duracion_audio_ms(ruta_audio):
    try:
        sonido = pygame.mixer.Sound(ruta_audio)
        return sonido.get_length() * 1000.0
    except Exception:
        y, sr = sf.read(ruta_audio)
        return (len(y) / sr) * 1000.0


def _rellenar_tiempos_faltantes_con_palabras(tiempos_ms, palabras, duracion_ms):
    n = len(tiempos_ms)
    if n == 0:
        return []

    resultado = list(tiempos_ms)
    if all(t is None for t in resultado):
        pesos = [max(len(p), 1) for p in palabras]
        peso_total = sum(pesos)
        cursor = 0.0
        for i in range(n):
            dur = (pesos[i] / peso_total) * duracion_ms
            resultado[i] = (cursor, cursor + dur)
            cursor += dur
        return resultado

    i = 0
    while i < n:
        if resultado[i] is not None:
            i += 1
            continue

        inicio_bloque = i
        while i < n and resultado[i] is None:
            i += 1
        fin_bloque = i

        t_antes = resultado[inicio_bloque - 1][1] if inicio_bloque > 0 else 0.0
        t_despues = resultado[fin_bloque][0] if fin_bloque < n else duracion_ms
        hueco = max(t_despues - t_antes, 1.0)

        bloque = palabras[inicio_bloque:fin_bloque]
        peso_total = sum(max(len(p), 1) for p in bloque) or len(bloque)
        cursor = t_antes
        for palabra in bloque:
            frac = max(len(palabra), 1) / peso_total
            dur = hueco * frac
            resultado[inicio_bloque] = (cursor, cursor + dur)
            cursor += dur
            inicio_bloque += 1

    return resultado


def _mapear_tiempos_palabras(palabras, indices_txt, detectadas, duracion_ms):
    esp_norm = [normalizar_palabra(p) for p in palabras]
    det_norm = [d["norm"] for d in detectadas]
    tiempos = [None] * len(palabras)

    matcher = SequenceMatcher(None, esp_norm, det_norm)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                det = detectadas[j1 + offset]
                tiempos[i1 + offset] = (det["inicio"], det["fin"])

    tiempos = _rellenar_tiempos_faltantes_con_palabras(tiempos, palabras, duracion_ms)

    segmentos = []
    for idx, (t_ini, t_fin) in enumerate(tiempos):
        segmentos.append({
            "idx_inicio": indices_txt[idx][0],
            "idx_fin": indices_txt[idx][1],
            "t_inicio_ms": max(0.0, t_ini),
            "t_fin_ms": max(t_ini + 30.0, t_fin),
        })
    return segmentos


def segmentos_fallback_ponderado(palabras, indices_txt, duracion_ms):
    if not palabras:
        return []
    pesos = [max(len(p), 1) for p in palabras]
    peso_total = sum(pesos)
    segmentos = []
    t_actual = 0.0
    for i, (p, (idx_ini, idx_fin)) in enumerate(zip(palabras, indices_txt)):
        dur_palabra = (pesos[i] / peso_total) * duracion_ms
        segmentos.append({
            "idx_inicio": idx_ini,
            "idx_fin": idx_fin,
            "t_inicio_ms": t_actual,
            "t_fin_ms": t_actual + dur_palabra,
        })
        t_actual += dur_palabra
    return segmentos


def alinear_teleprompter(ruta_audio, texto_completo, codigo_idioma, avisar=None):
    palabras = texto_completo.split()
    if not palabras:
        return []

    indices_txt = construir_indices_texto_multilinea(texto_completo, palabras)
    duracion_ms = _duracion_audio_ms(ruta_audio)

    try:
        if avisar:
            avisar("🎯 Alineando teleprompter palabra por palabra con Whisper large-v3...")

        modelo = obtener_modelo_whisper()
        idioma_whisper = WHISPER_POR_IDIOMA.get(codigo_idioma, codigo_idioma)

        segmentos_raw, _ = modelo.transcribe(
            ruta_audio,
            language=idioma_whisper,
            word_timestamps=True,
            vad_filter=False,
            beam_size=5,
            best_of=5,
            temperature=0.0,
        )

        detectadas = []
        for seg in segmentos_raw:
            if not seg.words:
                continue
            for palabra in seg.words:
                texto = (palabra.word or "").strip()
                if texto:
                    detectadas.append({
                        "norm": normalizar_palabra(texto),
                        "inicio": palabra.start * 1000.0,
                        "fin": palabra.end * 1000.0,
                    })

        if not detectadas:
            raise ValueError("Whisper no devolvió marcas de tiempo")

        segmentos = _mapear_tiempos_palabras(palabras, indices_txt, detectadas, duracion_ms)
        if avisar:
            avisar(f"✅ Teleprompter sincronizado ({len(segmentos)} palabras alineadas).")
        return segmentos

    except Exception as e:
        print(f"⚠️ Fallback teleprompter: {e}")
        if avisar:
            avisar("⚠️ Usando sincronización estimada ponderada.")
        return segmentos_fallback_ponderado(palabras, indices_txt, duracion_ms)


def indice_por_tiempo_bisect(segmentos, t_ms):
    if not segmentos:
        return 0
    tiempos_fin = [seg["t_fin_ms"] for seg in segmentos]
    idx = bisect.bisect_right(tiempos_fin, t_ms)
    if idx >= len(segmentos):
        return len(segmentos) - 1
    return idx


# =====================================================================
# 7. GENERACIÓN DE AUDIO Y REPRODUCCIÓN NATURAL
# =====================================================================
def centrar_en_indice(index):
    try:
        linea = int(index.split(".")[0])
        total_lineas = int(teleprompter._textbox.index("end-1c").split(".")[0])
        if total_lineas > 4:
            fraccion = max(0.0, min(1.0, (linea - 2) / max(total_lineas, 1)))
            teleprompter._textbox.yview_moveto(fraccion)
        else:
            teleprompter.see(index)
    except Exception:
        teleprompter.see(index)


def reproducir_con_teleprompter(segmentos, id_sesion):
    global estado_audio, reiniciar_solicitado, reloj_reproduccion, audio_numpy_actual, sr_audio_actual
    global cambio_velocidad_solicitado, nueva_velocidad_solicitada

    if id_sesion != sesion_reproduccion_id:
        return

    duracion_ms = duracion_audio_total_ms or 1.0

    try:
        if audio_numpy_actual is None or len(audio_numpy_actual) == 0:
            audio_numpy_actual, sr_audio_actual = sf.read(archivo_actual, dtype="float32")
    except Exception as e:
        app.after(0, lambda: agregar_mensaje(f"⚠️ Error al cargar buffer de audio: {e}", "bot"))
        return

    velocidad_reprod = slider_velocidad_reproductor.get()
    reloj_reproduccion = RelojReproduccion(velocidad_reprod)
    reloj_reproduccion.iniciar(0.0, velocidad_reprod)

    try:
        snd_actual = crear_sound_resampleado(audio_numpy_actual, sr_audio_actual, velocidad_reprod, 0.0)
        if snd_actual is None:
            raise ValueError("No se pudo preparar el audio para reproducción")
        canal_audio.stop()
        canal_audio.play(snd_actual)
        aplicar_volumen()
    except Exception as e:
        app.after(0, lambda: agregar_mensaje(f"⚠️ Error al reproducir audio: {e}", "bot"))
        return

    estado_audio = "reproduciendo"
    cambio_velocidad_solicitado = False
    reiniciar_solicitado = False
    app.after(0, lambda: set_botones_estado("activo"))

    indice_actual = -1

    while True:
        if id_sesion != sesion_reproduccion_id or estado_audio == "detenido":
            break

        if cambio_velocidad_solicitado:
            cambio_velocidad_solicitado = False
            v_nueva = nueva_velocidad_solicitada
            t_actual_ms = min(reloj_reproduccion.ms(), duracion_ms)
            reloj_reproduccion.cambiar_velocidad(v_nueva)
            try:
                snd_nuevo = crear_sound_resampleado(audio_numpy_actual, sr_audio_actual, v_nueva, t_actual_ms)
                if snd_nuevo is not None:
                    canal_audio.stop()
                    canal_audio.play(snd_nuevo)
                    aplicar_volumen()
                    if estado_audio == "pausado":
                        canal_audio.pause()
            except Exception as e:
                print(f"Aviso al cambiar velocidad: {e}")

        if reiniciar_solicitado:
            reiniciar_solicitado = False
            v_actual = slider_velocidad_reproductor.get()
            reloj_reproduccion.iniciar(0.0, v_actual)
            try:
                snd_reinicio = crear_sound_resampleado(audio_numpy_actual, sr_audio_actual, v_actual, 0.0)
                if snd_reinicio is not None:
                    canal_audio.stop()
                    canal_audio.play(snd_reinicio)
                    aplicar_volumen()
            except Exception as e:
                print(f"Aviso al reiniciar: {e}")
            indice_actual = -1
            app.after(0, lambda: teleprompter.tag_remove("resaltado", "1.0", "end"))
            app.after(0, lambda: barra_progreso.set(0.0))
            continue

        if estado_audio == "pausado":
            time.sleep(0.05)
            continue

        t_ms = reloj_reproduccion.ms()
        progreso = min(1.0, max(0.0, t_ms / duracion_ms))
        tiempo_texto = f"{formatear_tiempo_ms(t_ms)} / {formatear_tiempo_ms(duracion_ms)}"

        def actualizar_ui_tiempo(p=progreso, t=tiempo_texto):
            barra_progreso.set(p)
            lbl_tiempo_reproduccion.configure(text=t)

        app.after(0, actualizar_ui_tiempo)

        if not canal_audio.get_busy() and t_ms >= (duracion_ms - 200.0) and estado_audio != "pausado":
            break

        if segmentos:
            nuevo_indice = indice_por_tiempo_bisect(segmentos, t_ms)
            if nuevo_indice != indice_actual:
                indice_actual = nuevo_indice
                seg = segmentos[indice_actual]

                def actualizar_resaltado(s=seg):
                    teleprompter.tag_remove("resaltado", "1.0", "end")
                    teleprompter.tag_add("resaltado", s["idx_inicio"], s["idx_fin"])
                    centrar_en_indice(s["idx_inicio"])

                app.after(0, actualizar_resaltado)

        time.sleep(0.015)

    if id_sesion == sesion_reproduccion_id:
        canal_audio.stop()
        estado_audio = "detenido"
        app.after(0, lambda: set_botones_estado("terminado"))
        app.after(0, lambda: barra_progreso.set(1.0))
        app.after(0, lambda: lbl_tiempo_reproduccion.configure(text=f"{formatear_tiempo_ms(duracion_ms)} / {formatear_tiempo_ms(duracion_ms)}"))


def procesar_y_hablar(texto, codigo_tts, modelo, ruta_voz_ref, estilo_clave, velocidad, es_reinicio=False):
    global estado_audio, archivo_actual, texto_actual, idioma_actual, modelo_actual, voz_actual
    global estilo_actual, velocidad_actual, segmentos_teleprompter, duracion_audio_total_ms, sesion_reproduccion_id

    sesion_reproduccion_id += 1
    mi_sesion = sesion_reproduccion_id

    estado_audio = "generando"
    app.after(0, lambda: set_botones_estado("generando"))

    def avisar(msg):
        app.after(0, lambda m=msg: agregar_mensaje(m, "bot"))

    def restaurar_estado_error():
        global estado_audio
        estado_audio = "detenido"
        estado_final = "terminado" if (archivo_actual and os.path.exists(archivo_actual)) else "inactivo"
        app.after(0, lambda: set_botones_estado(estado_final))

    if not os.path.exists(ruta_voz_ref):
        avisar(f"⚠️ Archivo de voz no encontrado: {os.path.basename(ruta_voz_ref)}")
        restaurar_estado_error()
        return

    # Obtener muestra acústicamente curada (ventana <= 22 segundos normalizada a 0.95 de pico)
    ruta_voz_optima = os.path.abspath(curar_muestra_referencia(ruta_voz_ref))

    try:
        motor_cosy = obtener_motor_cosyvoice(avisar=avisar)
    except Exception as e:
        avisar(f"⚠️ Error al cargar CosyVoice 3: {e}")
        restaurar_estado_error()
        return

    if motor_cosy is None:
        avisar("⚠️ El motor CosyVoice 3 no se cargó correctamente.")
        restaurar_estado_error()
        return

    # Descargar descriptor de Pygame en Windows
    try:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        pygame.mixer.music.unload()
    except Exception:
        pass
    time.sleep(0.08)

    clave_cache = f"{codigo_tts}:{combo_voces.get()}:{estilo_clave}:{velocidad:.2f}"
    generar_nuevo = True

    if archivo_actual and os.path.exists(archivo_actual):
        if texto == texto_actual and clave_cache == idioma_actual:
            generar_nuevo = False

    if generar_nuevo:
        # Preprocesar texto mediante NLP para prosodia natural
        texto_procesado = preprocesar_texto_para_tts(texto, codigo_tts)

        estilo_cfg = ESTILOS_PROSODIA.get(estilo_clave, ESTILOS_PROSODIA["conversacional"])
        avisar(f"⚡ Sintetizando con CosyVoice 3 ({estilo_cfg['etiqueta']} | Voz: {combo_voces.get()})...")

        archivo_anterior = archivo_actual
        archivo_actual = os.path.abspath(os.path.join(CARPETA_TEMP, f"temp_audio_{int(time.time() * 1000)}.wav"))

        with bloqueo_sintesis:
            if mi_sesion != sesion_reproduccion_id:
                return

            t_gen_ini = time.perf_counter()
            try:
                # Registrar identidad de voz en CosyVoice 3 con caché en memoria
                nombre_voz_sel = combo_voces.get()
                spk_id, prompt_text = registrar_voz_en_cosyvoice(motor_cosy, nombre_voz_sel, ruta_voz_optima, avisar=avisar)

                # Dividir el texto procesado en fragmentos semánticos para evitar el límite de atención de CosyVoice 3
                fragmentos = dividir_texto_para_sintesis(texto_procesado, max_palabras=40)
                if not fragmentos:
                    fragmentos = [texto_procesado]

                total_frags = len(fragmentos)
                if total_frags > 1:
                    avisar(f"📄 Procesando {total_frags} párrafos/bloques para síntesis completa sin cortes.")

                outputs = []
                silencio_pausa = torch.zeros(1, int(motor_cosy.sample_rate * 0.20))  # 200ms de pausa natural

                for idx_frag, frag in enumerate(fragmentos, 1):
                    if mi_sesion != sesion_reproduccion_id:
                        return
                    if total_frags > 1:
                        avisar(f"⚡ Generando bloque {idx_frag}/{total_frags}...")

                    frag_outputs = []
                    for out in motor_cosy.inference_zero_shot(
                        frag,
                        prompt_text,
                        ruta_voz_optima,
                        zero_shot_spk_id=spk_id,
                        stream=False,
                        speed=velocidad,
                        text_frontend=False,
                    ):
                        frag_outputs.append(out["tts_speech"])

                    if frag_outputs:
                        outputs.append(torch.concat(frag_outputs, dim=1))
                        if idx_frag < total_frags:
                            outputs.append(silencio_pausa)

                if not outputs:
                    raise RuntimeError("CosyVoice 3 no produjo salida de audio.")

                audio_tensor = torch.concat(outputs, dim=1)
                audio_numpy_actual = audio_tensor.squeeze(0).cpu().numpy()
                sr_audio_actual = motor_cosy.sample_rate
                sf.write(archivo_actual, audio_numpy_actual, sr_audio_actual)

                t_gen_fin = time.perf_counter()
                dur_gen = t_gen_fin - t_gen_ini

                texto_actual = texto
                idioma_actual = clave_cache
                modelo_actual = "cosyvoice3"
                voz_actual = combo_voces.get()
                estilo_actual = estilo_clave
                velocidad_generacion_actual = velocidad

                info_audio = sf.info(archivo_actual)
                rtf = dur_gen / max(info_audio.duration, 0.1)
                avisar(f"✅ Audio generado en {dur_gen:.1f}s ({info_audio.duration:.1f}s de voz | RTF: {rtf:.2f}).")

            except Exception as e:
                avisar(f"⚠️ Error al generar síntesis con CosyVoice 3: {e}")
                restaurar_estado_error()
                return

        # Eliminar archivo temporal anterior
        if archivo_anterior and os.path.exists(archivo_anterior):
            try:
                os.remove(archivo_anterior)
            except Exception:
                pass
    else:
        if es_reinicio:
            avisar("🔄 Reproduciendo audio desde caché local...")

    try:
        duracion_audio_total_ms = _duracion_audio_ms(archivo_actual)

        # Cargar texto en teleprompter con párrafos íntegros
        def actualizar_caja_teleprompter():
            teleprompter.configure(state="normal")
            teleprompter.delete("1.0", "end")
            teleprompter.insert("end", texto_actual)
            teleprompter.configure(state="disabled")
            barra_progreso.set(0.0)
            lbl_tiempo_reproduccion.configure(text=f"00:00 / {formatear_tiempo_ms(duracion_audio_total_ms)}")

        app.after(0, actualizar_caja_teleprompter)

        if generar_nuevo or not segmentos_teleprompter:
            segmentos_teleprompter = alinear_teleprompter(
                archivo_actual, texto_actual, codigo_tts, avisar=avisar
            )

        if mi_sesion != sesion_reproduccion_id:
            return

        reproducir_con_teleprompter(segmentos_teleprompter, mi_sesion)

    except Exception as e:
        avisar(f"⚠️ Error durante la preparación del teleprompter: {e}")
        restaurar_estado_error()


# =====================================================================
# 8. ACCIONES DE CONTROL, GUARDADO Y VOLUMEN
# =====================================================================
def aplicar_volumen():
    vol = 0.0 if audio_muteado else volumen_actual
    canal_audio.set_volume(vol)
    try:
        pygame.mixer.music.set_volume(vol)
    except Exception:
        pass


def ajustar_volumen(valor):
    global volumen_actual, audio_muteado
    volumen_actual = float(valor)
    if audio_muteado and volumen_actual > 0:
        audio_muteado = False
        actualizar_boton_mute()
    aplicar_volumen()
    lbl_volumen_valor.configure(text=f"{int(volumen_actual * 100)}%")


def toggle_mute():
    global audio_muteado
    audio_muteado = not audio_muteado
    aplicar_volumen()
    actualizar_boton_mute()


def actualizar_boton_mute():
    if audio_muteado:
        btn_mute.configure(text="🔇 Mute", fg_color="#E76F51", hover_color="#F4A261")
    else:
        btn_mute.configure(text="🔊 Sonido", fg_color="#5A189A", hover_color="#7B2CBF")


def toggle_pausa():
    global estado_audio, reloj_reproduccion
    if estado_audio == "reproduciendo":
        canal_audio.pause()
        try:
            pygame.mixer.music.pause()
        except Exception:
            pass
        if reloj_reproduccion:
            reloj_reproduccion.pausar()
        estado_audio = "pausado"
        btn_pausa.configure(text="▶️ Reanudar", fg_color="#F4A261", hover_color="#E76F51")
    elif estado_audio == "pausado":
        canal_audio.unpause()
        try:
            pygame.mixer.music.unpause()
        except Exception:
            pass
        if reloj_reproduccion:
            reloj_reproduccion.reanudar()
        estado_audio = "reproduciendo"
        btn_pausa.configure(text="⏸️ Pausa", fg_color="#2A9D8F", hover_color="#21867A")


def hacer_reinicio():
    global reiniciar_solicitado, estado_audio, texto_actual, idioma_actual, modelo_actual
    if estado_audio in ["reproduciendo", "pausado"]:
        reiniciar_solicitado = True
        if estado_audio == "pausado":
            toggle_pausa()
    elif estado_audio == "detenido" and texto_actual != "":
        codigo_tts = idioma_actual.split(":")[0] if ":" in idioma_actual else "es"
        modelo = modelo_actual
        ruta_voz = obtener_ruta_voz(combo_voces.get())
        estilo_clave = combo_estilo_prosodia.get()
        for k, v in ESTILOS_PROSODIA.items():
            if v["etiqueta"] == estilo_clave:
                estilo_clave = k
                break
        hilo = threading.Thread(
            target=procesar_y_hablar,
            args=(texto_actual, codigo_tts, modelo, ruta_voz, estilo_clave, slider_velocidad_generacion.get(), True),
            daemon=True,
        )
        hilo.start()


def hacer_detener():
    global estado_audio, sesion_reproduccion_id
    sesion_reproduccion_id += 1
    estado_audio = "detenido"
    canal_audio.stop()
    try:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
    except Exception:
        pass
    teleprompter.tag_remove("resaltado", "1.0", "end")
    barra_progreso.set(0.0)
    if archivo_actual and os.path.exists(archivo_actual):
        set_botones_estado("terminado")
    else:
        set_botones_estado("inactivo")


def hacer_borrar():
    global archivo_actual, audio_numpy_actual, texto_actual, idioma_actual, modelo_actual, estado_audio
    global segmentos_teleprompter, reloj_reproduccion, duracion_audio_total_ms, sesion_reproduccion_id

    sesion_reproduccion_id += 1
    estado_audio = "detenido"
    canal_audio.stop()
    try:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        pygame.mixer.music.unload()
    except Exception:
        pass
    time.sleep(0.05)

    if archivo_actual and os.path.exists(archivo_actual):
        try:
            os.remove(archivo_actual)
        except Exception:
            pass

    archivo_actual = None
    audio_numpy_actual = None
    texto_actual = ""
    idioma_actual = ""
    modelo_actual = "estandar"
    segmentos_teleprompter = []
    reloj_reproduccion = None
    duracion_audio_total_ms = 0.0

    teleprompter.configure(state="normal")
    teleprompter.delete("1.0", "end")
    teleprompter.insert("end", "El texto que esté siendo leído aparecerá aquí...")
    teleprompter.configure(state="disabled")

    barra_progreso.set(0.0)
    lbl_tiempo_reproduccion.configure(text="00:00 / 00:00")
    set_botones_estado("inactivo")
    agregar_mensaje("🗑️ Archivo temporal borrado. Sistema limpio.", "bot")


def hacer_guardar():
    global archivo_actual, idioma_actual
    if not archivo_actual or not os.path.exists(archivo_actual):
        agregar_mensaje("⚠️ No hay ningún audio en memoria para guardar.", "bot")
        return

    dialogo = ctk.CTkInputDialog(text="Ingresa un nombre o referencia para este audio:", title="Guardar Audio")
    referencia = dialogo.get_input()

    if referencia:
        referencia_limpia = "".join(x for x in referencia if x.isalnum() or x in " _-").strip().replace(" ", "_")
        if not referencia_limpia:
            referencia_limpia = "Audio"

        idioma_abrev = idioma_actual.split(":")[0].upper() if idioma_actual else "ES"
        id_unico = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo_final = f"{referencia_limpia}_{id_unico}_{idioma_abrev}.wav"

        carpeta_idioma = os.path.join(CARPETA_GUARDADOS, idioma_abrev)
        os.makedirs(carpeta_idioma, exist_ok=True)
        ruta_destino = os.path.join(carpeta_idioma, nombre_archivo_final)

        try:
            shutil.copy(archivo_actual, ruta_destino)
            agregar_mensaje(f"💾 ¡Audio guardado con éxito!\nDestino: {ruta_destino}", "bot")
            app.after(500, lambda: agregar_mensaje(f"📂 Carpeta de audios:\n{os.path.abspath(carpeta_idioma)}", "bot"))
        except Exception as e:
            agregar_mensaje(f"⚠️ Error al guardar: {e}", "bot")


def cambiar_tamano_fuente(delta):
    global tamano_fuente_teleprompter
    nuevo_tamano = max(16, min(40, tamano_fuente_teleprompter + delta))
    if nuevo_tamano != tamano_fuente_teleprompter:
        tamano_fuente_teleprompter = nuevo_tamano
        teleprompter.configure(font=("Arial", tamano_fuente_teleprompter, "bold"))
        lbl_tamano_fuente.configure(text=f"{tamano_fuente_teleprompter}pt")


def set_botones_estado(estado):
    if estado == "activo":
        btn_pausa.configure(state="normal", text="⏸️ Pausa", fg_color="#2A9D8F")
        btn_mute.configure(state="normal")
        actualizar_boton_mute()
        btn_reiniciar.configure(state="normal")
        btn_detener.configure(state="normal")
        btn_borrar.configure(state="disabled")
        btn_guardar.configure(state="disabled")
        btn_enviar.configure(state="normal", text="🎙️ Generar y Leer", fg_color="#00A896", hover_color="#028090")
    elif estado == "terminado":
        btn_pausa.configure(state="disabled", text="⏸️ Pausa", fg_color="gray")
        btn_mute.configure(state="normal")
        actualizar_boton_mute()
        btn_reiniciar.configure(state="normal")
        btn_detener.configure(state="disabled")
        btn_borrar.configure(state="normal")
        btn_guardar.configure(state="normal")
        btn_enviar.configure(state="normal", text="🎙️ Generar y Leer", fg_color="#00A896", hover_color="#028090")
    elif estado == "generando":
        btn_pausa.configure(state="disabled", text="⏸️ Pausa", fg_color="gray")
        btn_mute.configure(state="disabled")
        btn_reiniciar.configure(state="disabled")
        btn_detener.configure(state="disabled")
        btn_borrar.configure(state="disabled")
        btn_guardar.configure(state="disabled")
        btn_enviar.configure(state="disabled", text="⏳ Generando...", fg_color="#4A5568")
    else:  # "inactivo" (reposo / audio borrado / inicio sin audio cargado)
        btn_pausa.configure(state="disabled", text="⏸️ Pausa", fg_color="gray")
        btn_mute.configure(state="disabled")
        btn_reiniciar.configure(state="disabled")
        btn_detener.configure(state="disabled")
        btn_borrar.configure(state="disabled")
        btn_guardar.configure(state="disabled")
        btn_enviar.configure(state="normal", text="🎙️ Generar y Leer", fg_color="#00A896", hover_color="#028090")


# =====================================================================
# 9. GESTIÓN Y PREESCUCHA DE VOCES
# =====================================================================
canal_muestra_voz = None

def preescuchar_voz_seleccionada():
    global canal_muestra_voz
    nombre_voz = combo_voces.get()
    ruta_voz = obtener_ruta_voz(nombre_voz)

    if not ruta_voz or not os.path.exists(ruta_voz):
        messagebox.showwarning("Voz no encontrada", f"No se encontró el archivo de audio para '{nombre_voz}'.")
        return

    try:
        sonido = pygame.mixer.Sound(ruta_voz)
        sonido.set_volume(volumen_actual)
        if canal_muestra_voz is not None and canal_muestra_voz.get_busy():
            canal_muestra_voz.stop()
        canal_muestra_voz = sonido.play()
        agregar_mensaje(f"🎧 Escuchando muestra de referencia: {nombre_voz}", "bot")
    except Exception as e:
        messagebox.showerror("Error de Reproducción", f"No se pudo reproducir la muestra:\n{e}")


def eliminar_voz_seleccionada():
    nombre_voz = combo_voces.get()
    if nombre_voz == VOZ_DEFAULT_NOMBRE:
        messagebox.showinfo("Acción no permitida", "La voz original por defecto no puede ser eliminada.")
        return

    confirmacion = messagebox.askyesno(
        "Eliminar Voz",
        f"¿Estás seguro de que deseas eliminar la voz '{nombre_voz}'?\nEsta acción no se puede deshacer.",
    )
    if not confirmacion:
        return

    ruta_voz = os.path.join(CARPETA_VOCES, f"{nombre_voz}.wav")
    ruta_opt = os.path.join(CARPETA_VOCES_OPT, f"{nombre_voz}_opt.wav")
    try:
        if os.path.exists(ruta_voz):
            os.remove(ruta_voz)
        if os.path.exists(ruta_opt):
            os.remove(ruta_opt)
        agregar_mensaje(f"🗑️ Voz '{nombre_voz}' eliminada.", "bot")
        actualizar_selector_voces()
    except Exception as e:
        messagebox.showerror("Error", f"No se pudo eliminar la voz:\n{e}")


def actualizar_selector_voces(voz_a_seleccionar=None):
    voces = listar_voces_disponibles()
    combo_voces.configure(values=voces)
    if voz_a_seleccionar and voz_a_seleccionar in voces:
        combo_voces.set(voz_a_seleccionar)
    elif voces:
        combo_voces.set(voces[0])


def habilitar_menu_contextual(widget):
    """
    Habilita menú contextual al hacer clic derecho (Cortar, Copiar, Pegar, Seleccionar todo, Limpiar)
    con soporte para CTkTextbox, CTkEntry, tk.Text y tk.Entry.
    """
    target = getattr(widget, "_textbox", getattr(widget, "_entry", widget))

    menu = tk.Menu(
        target,
        tearoff=0,
        bg="#1E1E2E",
        fg="#E0E0E0",
        activebackground="#00A896",
        activeforeground="#FFFFFF",
        activeborderwidth=0,
        bd=1,
        relief="solid",
        font=("Arial", 10),
    )

    def cortar():
        try:
            target.event_generate("<<Cut>>")
        except Exception:
            pass
        if hasattr(widget, "event_generate"):
            try:
                widget.event_generate("<KeyRelease>")
            except Exception:
                pass

    def copiar():
        try:
            target.event_generate("<<Copy>>")
        except Exception:
            pass

    def pegar():
        try:
            texto_clip = target.clipboard_get()
            if texto_clip:
                if isinstance(target, tk.Text):
                    try:
                        target.delete("sel.first", "sel.last")
                    except Exception:
                        pass
                    target.insert("insert", texto_clip)
                elif isinstance(target, tk.Entry):
                    try:
                        target.delete("sel.first", "sel.last")
                    except Exception:
                        pass
                    target.insert("insert", texto_clip)
                else:
                    target.event_generate("<<Paste>>")
        except Exception:
            try:
                target.event_generate("<<Paste>>")
            except Exception:
                pass

        if hasattr(widget, "event_generate"):
            try:
                widget.event_generate("<KeyRelease>")
            except Exception:
                pass

    def seleccionar_todo():
        try:
            if isinstance(target, tk.Text):
                target.tag_add("sel", "1.0", "end-1c")
                target.mark_set("insert", "end-1c")
            elif isinstance(target, tk.Entry):
                target.select_range(0, "end")
                target.icursor("end")
        except Exception:
            pass

    def borrar_todo():
        try:
            if isinstance(target, tk.Text):
                target.delete("1.0", "end")
            elif isinstance(target, tk.Entry):
                target.delete(0, "end")
        except Exception:
            pass
        if hasattr(widget, "event_generate"):
            try:
                widget.event_generate("<KeyRelease>")
            except Exception:
                pass

    def mostrar_menu(event):
        menu.delete(0, "end")

        es_editable = True
        try:
            if str(target.cget("state")) == "disabled":
                es_editable = False
        except Exception:
            pass

        tiene_seleccion = False
        try:
            if isinstance(target, tk.Text):
                tiene_seleccion = bool(target.tag_ranges("sel"))
            elif isinstance(target, tk.Entry):
                tiene_seleccion = target.selection_present()
        except Exception:
            pass

        tiene_portapapeles = False
        try:
            tiene_portapapeles = bool(target.clipboard_get())
        except Exception:
            pass

        if es_editable:
            menu.add_command(label="✂️ Cortar", command=cortar, state="normal" if tiene_seleccion else "disabled")
        menu.add_command(label="📋 Copiar", command=copiar, state="normal" if tiene_seleccion else "disabled")
        if es_editable:
            menu.add_command(label="📥 Pegar", command=pegar, state="normal" if tiene_portapapeles else "disabled")
        menu.add_separator()
        menu.add_command(label="🔘 Seleccionar todo", command=seleccionar_todo)
        if es_editable:
            menu.add_command(label="🧹 Limpiar", command=borrar_todo)

        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    target.bind("<Button-3>", mostrar_menu)
    if hasattr(widget, "bind"):
        widget.bind("<Button-3>", mostrar_menu)


def abrir_dialogo_nueva_voz():
    modal = ctk.CTkToplevel(app)
    modal.title("➕ Clonar Nueva Voz")
    modal.geometry("520x430")
    modal.resizable(False, False)
    modal.grab_set()

    lbl_titulo = ctk.CTkLabel(modal, text="🎙️ Crear Nueva Voz para Clonación", font=("Arial", 16, "bold"), text_color="#00FFCC")
    lbl_titulo.pack(pady=(15, 10))

    frame_form = ctk.CTkFrame(modal)
    frame_form.pack(padx=20, pady=10, fill="both", expand=True)

    lbl_nombre = ctk.CTkLabel(frame_form, text="Nombre de la voz (ej: Locutor_Carlos):", font=("Arial", 13, "bold"))
    lbl_nombre.pack(anchor="w", padx=15, pady=(10, 2))

    entry_nombre = ctk.CTkEntry(frame_form, placeholder_text="Ingresa un nombre sin espacios raros...", height=35)
    entry_nombre.pack(fill="x", padx=15, pady=(0, 10))
    habilitar_menu_contextual(entry_nombre)

    lbl_archivo = ctk.CTkLabel(frame_form, text="Muestra de voz (.wav, .mp3, .ogg, .flac, .m4a):", font=("Arial", 13, "bold"))
    lbl_archivo.pack(anchor="w", padx=15, pady=(5, 2))

    ruta_seleccionada = tk.StringVar(value="")

    frame_archivo = ctk.CTkFrame(frame_form, fg_color="transparent")
    frame_archivo.pack(fill="x", padx=15, pady=(0, 10))

    entry_ruta = ctk.CTkEntry(frame_archivo, textvariable=ruta_seleccionada, placeholder_text="Selecciona un archivo...", height=35, state="disabled")
    entry_ruta.pack(side="left", fill="x", expand=True, padx=(0, 10))

    lbl_info_audio = ctk.CTkLabel(frame_form, text="💡 Recomendación: 5 a 15 segundos de voz limpia sin música ni ruido.", font=("Arial", 11), text_color="#AAAAAA")
    lbl_info_audio.pack(padx=15, pady=(0, 10))

    def seleccionar_archivo():
        tipos = [
            ("Archivos de audio", "*.wav *.mp3 *.ogg *.flac *.m4a"),
            ("Todos los archivos", "*.*"),
        ]
        ruta = filedialog.askopenfilename(title="Seleccionar muestra de voz", filetypes=tipos)
        if ruta:
            ruta_seleccionada.set(ruta)
            try:
                y, sr = sf.read(ruta)
                dur = len(y) / sr
                lbl_info_audio.configure(
                    text=f"✅ Duración detectada: {dur:.1f}s. Formato compatible.",
                    text_color="#2A9D8F",
                )
            except Exception:
                lbl_info_audio.configure(
                    text="ℹ️ Archivo seleccionado. Se procesará automáticamente.",
                    text_color="#F4A261",
                )

    btn_examinar = ctk.CTkButton(frame_archivo, text="Examinar...", width=100, height=35, command=seleccionar_archivo)
    btn_examinar.pack(side="right")

    lbl_estado_modal = ctk.CTkLabel(modal, text="", font=("Arial", 12))
    lbl_estado_modal.pack(pady=(0, 10))

    def guardar_voz():
        nombre = entry_nombre.get().strip()
        ruta = ruta_seleccionada.get().strip()

        if not nombre:
            lbl_estado_modal.configure(text="⚠️ Ingresa un nombre para la voz.", text_color="#D90429")
            return
        if not ruta or not os.path.exists(ruta):
            lbl_estado_modal.configure(text="⚠️ Selecciona un archivo de audio válido.", text_color="#D90429")
            return

        lbl_estado_modal.configure(text="⏳ Procesando y normalizando audio a 24 kHz...", text_color="#F4A261")
        modal.update()

        exito, mensaje, _ = validar_y_procesar_audio_voz(ruta, nombre)
        if exito:
            nombre_limpio = "".join(c for c in nombre if c.isalnum() or c in " _-").strip().replace(" ", "_")
            actualizar_selector_voces(nombre_limpio)
            agregar_mensaje(f"🎙️ ¡Nueva voz registrada!: {nombre_limpio}", "bot")
            modal.destroy()
        else:
            lbl_estado_modal.configure(text=f"⚠️ {mensaje}", text_color="#D90429")

    frame_acciones = ctk.CTkFrame(modal, fg_color="transparent")
    frame_acciones.pack(pady=(0, 15))

    btn_guardar_voz = ctk.CTkButton(frame_acciones, text="💾 Guardar Voz", width=140, height=35, font=("Arial", 13, "bold"), fg_color="#2B9348", hover_color="#007F5F", command=guardar_voz)
    btn_guardar_voz.pack(side="left", padx=10)

    btn_cancelar = ctk.CTkButton(frame_acciones, text="Cancelar", width=100, height=35, fg_color="#6C757D", hover_color="#495057", command=modal.destroy)
    btn_cancelar.pack(side="left", padx=10)


# =====================================================================
# 10. ENVÍO DE TEXTO Y TRADUCCIÓN
# =====================================================================
def parsear_idioma_combo(seleccion):
    if seleccion.startswith("AUTO"):
        return "auto", "cosyvoice3"
    codigo_ui = seleccion.split(" - ", 1)[0].strip().lower()
    return MAPA_IDIOMAS.get(codigo_ui, (codigo_ui, "cosyvoice3"))


def resolver_idioma_lectura(texto, seleccion_combo, usar_traduccion, codigo_destino_trad=None):
    if usar_traduccion and codigo_destino_trad:
        return (
            codigo_destino_trad,
            "cosyvoice3",
            f"🔊 Leyendo traducción en {codigo_destino_trad.upper()}",
        )

    codigo_sel, modelo_sel = parsear_idioma_combo(seleccion_combo)

    if codigo_sel == "auto":
        try:
            from langdetect import detect
            codigo = detect(texto.strip())
            detectado = DETECTADO_A_CODIGO.get(codigo, "es")
        except Exception:
            detectado = "es"

        modelo_final = "cosyvoice3"
        aviso = f"🔍 Idioma auto-detectado: {detectado.upper()}"
        return detectado, modelo_final, aviso
    else:
        aviso = f"🔊 Leyendo en {seleccion_combo.split(' - ', 1)[-1]}"
        return codigo_sel, modelo_sel, aviso


def parsear_codigo_traduccion(seleccion):
    prefijo = seleccion.split(" - ", 1)[0].strip().lower()
    mapa = {cod: cod for _, cod in IDIOMAS_TRADUCCION}
    mapa["zh"] = "zh-CN"
    return mapa.get(prefijo, prefijo or "en")


def traducir_con_google(texto, idioma_destino, idioma_origen="auto"):
    from deep_translator import GoogleTranslator
    traductor = GoogleTranslator(source=idioma_origen, target=idioma_destino)
    return traductor.translate(texto)


def actualizar_contador_texto(event=None):
    texto = entrada_texto.get("1.0", "end-1c")
    palabras = len(texto.split())
    caracteres = len(texto)
    lbl_contador.configure(text=f"{palabras} palabras | {caracteres} caracteres")


def limpiar_entrada_texto():
    entrada_texto.delete("1.0", "end")
    actualizar_contador_texto()
    if switch_traducir.get():
        actualizar_panel_traduccion("", "🌐 Escribe para traducir en vivo...")


def actualizar_etiqueta_velocidad_generacion(val):
    global velocidad_generacion_actual
    v = round(float(val), 2)
    velocidad_generacion_actual = v
    texto_v = f"{v:.2f}".rstrip('0').rstrip('.') + 'x' if v != 1.0 else "1.0x"
    lbl_velocidad_gen_valor.configure(text=texto_v)


def actualizar_etiqueta_velocidad_reproductor(val):
    global velocidad_reproduccion_actual, cambio_velocidad_solicitado, nueva_velocidad_solicitada
    v = round(float(val), 2)
    velocidad_reproduccion_actual = v
    texto_v = f"{v:.2f}".rstrip('0').rstrip('.') + 'x' if v != 1.0 else "1.0x"
    lbl_velocidad_rep_valor.configure(text=texto_v)
    if estado_audio in ["reproduciendo", "pausado"]:
        nueva_velocidad_solicitada = v
        cambio_velocidad_solicitado = True


def enviar_mensaje(event=None):
    global estado_audio
    if estado_audio == "generando":
        return "break"
    texto_original = entrada_texto.get("1.0", "end-1c").strip()
    if not texto_original:
        return "break"

    nombre_voz_seleccionada = combo_voces.get()
    ruta_voz = obtener_ruta_voz(nombre_voz_seleccionada)
    if not ruta_voz or not os.path.exists(ruta_voz):
        agregar_mensaje(f"⚠️ No se encontró la voz de referencia: '{nombre_voz_seleccionada}'", "bot")
        return "break"

    usar_traduccion = switch_traducir.get()
    codigo_destino_trad = None
    texto_a_leer = texto_original
    mensaje_usuario = f"Tú:\n{texto_original}"

    if usar_traduccion:
        codigo_destino_trad = parsear_codigo_traduccion(combo_idioma_destino.get())
        texto_traducido = obtener_texto_traducido_actual()
        if not texto_traducido:
            try:
                actualizar_panel_traduccion("", "⏳ Traduciendo antes de leer...")
                texto_traducido = traducir_con_google(texto_original, codigo_destino_trad)
            except Exception as e:
                agregar_mensaje(f"⚠️ No se pudo traducir: {e}", "bot")
                return "break"
        texto_a_leer = texto_traducido
        etiqueta_dest = combo_idioma_destino.get().split(" - ", 1)[-1]
        mensaje_usuario = (
            f"Tú (original):\n{texto_original}\n\n"
            f"🌐 Traducción ({etiqueta_dest}):\n{texto_traducido}"
        )

    codigo_tts, modelo, aviso_idioma = resolver_idioma_lectura(
        texto_a_leer,
        combo_idioma.get(),
        usar_traduccion,
        codigo_destino_trad,
    )

    # Identificar clave de estilo de prosodia
    estilo_sel = combo_estilo_prosodia.get()
    estilo_clave = "conversacional"
    for k, v in ESTILOS_PROSODIA.items():
        if v["etiqueta"] == estilo_sel:
            estilo_clave = k
            break

    velocidad_gen = slider_velocidad_generacion.get()

    if estado_audio != "detenido":
        hacer_detener()

    def iniciar_lectura():
        agregar_mensaje(mensaje_usuario, "user")
        agregar_mensaje(f"{aviso_idioma} | {ESTILOS_PROSODIA[estilo_clave]['etiqueta']} (Gen IA: {velocidad_gen:.2f}x)", "bot")
        procesar_y_hablar(texto_a_leer, codigo_tts, modelo, ruta_voz, estilo_clave, velocidad_gen)

    hilo = threading.Thread(target=iniciar_lectura, daemon=True)
    hilo.start()
    return "break"


def agregar_mensaje(texto, tipo):
    caja_chat.configure(state="normal")
    caja_chat.insert("end", texto + "\n\n")
    caja_chat.configure(state="disabled")
    caja_chat.see("end")


def limpiar_chat():
    caja_chat.configure(state="normal")
    caja_chat.delete("1.0", "end")
    caja_chat.insert("end", "🧹 Historial de chat limpiado. ¡Listo para nuevos textos!\n\n")
    caja_chat.configure(state="disabled")


def actualizar_panel_traduccion(texto="", estado=""):
    def _actualizar():
        caja_traduccion_vivo.configure(state="normal")
        caja_traduccion_vivo.delete("1.0", "end")
        if texto:
            caja_traduccion_vivo.insert("end", texto)
        caja_traduccion_vivo.configure(state="disabled")
        if estado:
            lbl_estado_traduccion.configure(text=estado)

    app.after(0, _actualizar)


def _traducir_en_segundo_plano(texto, idioma_destino, seq):
    global traduccion_seq
    try:
        resultado = traducir_con_google(texto, idioma_destino)
        if seq != traduccion_seq:
            return
        actualizar_panel_traduccion(resultado, "🌐 Traducción en vivo (Google)")
    except Exception as e:
        if seq != traduccion_seq:
            return
        actualizar_panel_traduccion("", f"⚠️ Error: {e}")


def ejecutar_traduccion_vivo():
    global traduccion_seq
    if not switch_traducir.get():
        return

    texto = entrada_texto.get("1.0", "end-1c").strip()
    if not texto:
        actualizar_panel_traduccion("", "🌐 Escribe para traducir en vivo...")
        return

    traduccion_seq += 1
    seq = traduccion_seq
    idioma_destino = parsear_codigo_traduccion(combo_idioma_destino.get())
    actualizar_panel_traduccion("", "⏳ Traduciendo...")

    hilo = threading.Thread(
        target=_traducir_en_segundo_plano,
        args=(texto, idioma_destino, seq),
        daemon=True,
    )
    hilo.start()


def programar_traduccion_vivo(event=None):
    global job_traduccion_id
    actualizar_contador_texto()
    if not switch_traducir.get():
        return
    if job_traduccion_id is not None:
        app.after_cancel(job_traduccion_id)
    job_traduccion_id = app.after(450, ejecutar_traduccion_vivo)


def al_cambiar_switch_traduccion():
    if switch_traducir.get():
        combo_idioma_destino.configure(state="normal")
        if not frame_traduccion_caja.winfo_ismapped():
            frame_traduccion_caja.pack(pady=(0, 5), padx=10, fill="x", before=frame_entrada)
        lbl_estado_traduccion.configure(text="🌐 Conectado a Google Translate")
        ejecutar_traduccion_vivo()
    else:
        frame_traduccion_caja.pack_forget()
        combo_idioma_destino.configure(state="disabled")
        lbl_estado_traduccion.configure(text="🌐 Activa el interruptor para traducir")
        actualizar_panel_traduccion()


def al_cambiar_idioma_destino(_event=None):
    if switch_traducir.get():
        ejecutar_traduccion_vivo()


def obtener_texto_traducido_actual():
    if not switch_traducir.get():
        return None
    texto = caja_traduccion_vivo.get("1.0", "end").strip()
    if texto and not texto.startswith("⏳") and not texto.startswith("⚠️"):
        return texto
    return None


# =====================================================================
# 11. INTERFAZ GRÁFICA DE USUARIO
# =====================================================================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.geometry("980x960")
app.minsize(820, 750)
app.title("Lector y Teleprompter con CosyVoice 3 (RTX 3090 Offline)")

# ----------------- PANEL SUPERIOR (VOZ, IDIOMA Y PROSODIA) -----------------
frame_superior = ctk.CTkFrame(app, corner_radius=8)
frame_superior.pack(pady=(10, 5), padx=10, fill="x")

# Fila 1: Selección y Gestión de Voces
frame_voces_fila = ctk.CTkFrame(frame_superior, fg_color="transparent")
frame_voces_fila.pack(fill="x", padx=10, pady=(8, 4))

lbl_voz = ctk.CTkLabel(frame_voces_fila, text="🎙️ Voz Clonada:", font=("Arial", 13, "bold"), text_color="#00FFCC")
lbl_voz.pack(side="left", padx=(0, 8))

voces_iniciales = listar_voces_disponibles()
combo_voces = ctk.CTkComboBox(frame_voces_fila, values=voces_iniciales, width=220)
combo_voces.pack(side="left", padx=(0, 8))
if voces_iniciales:
    combo_voces.set(voces_iniciales[0])

btn_oir_voz = ctk.CTkButton(
    frame_voces_fila,
    text="🎧 Escuchar",
    width=90,
    font=("Arial", 12, "bold"),
    fg_color="#3A0CA3",
    hover_color="#4361EE",
    command=preescuchar_voz_seleccionada,
)
btn_oir_voz.pack(side="left", padx=(0, 6))

btn_nueva_voz = ctk.CTkButton(
    frame_voces_fila,
    text="➕ Nueva Voz",
    width=100,
    font=("Arial", 12, "bold"),
    fg_color="#2B9348",
    hover_color="#007F5F",
    command=abrir_dialogo_nueva_voz,
)
btn_nueva_voz.pack(side="left", padx=(0, 6))

btn_eliminar_voz = ctk.CTkButton(
    frame_voces_fila,
    text="🗑️",
    width=35,
    font=("Arial", 12, "bold"),
    fg_color="#6C757D",
    hover_color="#D90429",
    command=eliminar_voz_seleccionada,
)
btn_eliminar_voz.pack(side="left")

# Fila 2: Idioma, Estilo de Prosodia y Velocidad
frame_prosodia_fila = ctk.CTkFrame(frame_superior, fg_color="transparent")
frame_prosodia_fila.pack(fill="x", padx=10, pady=(4, 8))

lbl_idioma = ctk.CTkLabel(frame_prosodia_fila, text="🌐 Acento:", font=("Arial", 13, "bold"))
lbl_idioma.pack(side="left", padx=(0, 6))

combo_idioma = ctk.CTkComboBox(frame_prosodia_fila, values=IDIOMAS_VOZ_OPCIONES, width=220)
combo_idioma.pack(side="left", padx=(0, 12))
combo_idioma.set(IDIOMAS_VOZ_OPCIONES[0])

lbl_estilo = ctk.CTkLabel(frame_prosodia_fila, text="🎭 Prosodia:", font=("Arial", 13, "bold"))
lbl_estilo.pack(side="left", padx=(0, 6))

opciones_estilos = [v["etiqueta"] for v in ESTILOS_PROSODIA.values()]
combo_estilo_prosodia = ctk.CTkComboBox(frame_prosodia_fila, values=opciones_estilos, width=190)
combo_estilo_prosodia.pack(side="left", padx=(0, 12))
combo_estilo_prosodia.set(opciones_estilos[0])

lbl_velocidad_gen = ctk.CTkLabel(frame_prosodia_fila, text="⚡ Ritmo IA:", font=("Arial", 13, "bold"))
lbl_velocidad_gen.pack(side="left", padx=(0, 6))

slider_velocidad_generacion = ctk.CTkSlider(
    frame_prosodia_fila,
    from_=0.8,
    to=1.2,
    width=95,
    number_of_steps=8,
    command=actualizar_etiqueta_velocidad_generacion,
)
slider_velocidad_generacion.set(1.0)
slider_velocidad_generacion.pack(side="left", padx=(0, 6))

lbl_velocidad_gen_valor = ctk.CTkLabel(frame_prosodia_fila, text="1.0x", font=("Arial", 12, "bold"), width=35)
lbl_velocidad_gen_valor.pack(side="left")

# ----------------- PANEL DEL TELEPROMPTER -----------------
frame_tele_header = ctk.CTkFrame(app, fg_color="transparent")
frame_tele_header.pack(fill="x", padx=15, pady=(5, 0))

lbl_tele = ctk.CTkLabel(frame_tele_header, text="🎙️ TELEPROMPTER EN VIVO:", font=("Arial", 13, "bold"), text_color="#00FFCC")
lbl_tele.pack(side="left")

frame_fuente = ctk.CTkFrame(frame_tele_header, fg_color="transparent")
frame_fuente.pack(side="right")

lbl_tamano_lbl = ctk.CTkLabel(frame_fuente, text="Tamaño:", font=("Arial", 11), text_color="#AAAAAA")
lbl_tamano_lbl.pack(side="left", padx=(0, 6))

btn_fuente_menos = ctk.CTkButton(frame_fuente, text="A-", width=32, height=24, font=("Arial", 11, "bold"), command=lambda: cambiar_tamano_fuente(-2))
btn_fuente_menos.pack(side="left", padx=2)

lbl_tamano_fuente = ctk.CTkLabel(frame_fuente, text=f"{tamano_fuente_teleprompter}pt", font=("Arial", 11, "bold"), width=40)
lbl_tamano_fuente.pack(side="left")

btn_fuente_mas = ctk.CTkButton(frame_fuente, text="A+", width=32, height=24, font=("Arial", 11, "bold"), command=lambda: cambiar_tamano_fuente(2))
btn_fuente_mas.pack(side="left", padx=2)

teleprompter = ctk.CTkTextbox(
    app,
    height=165,
    font=("Arial", tamano_fuente_teleprompter, "bold"),
    wrap="word",
    fg_color="#121218",
    text_color="#555566",
    corner_radius=8,
)
teleprompter.pack(pady=5, padx=10, fill="x")
teleprompter.tag_config("resaltado", foreground="#00FFCC")
teleprompter.insert("end", "El texto que esté siendo leído aparecerá aquí...")
teleprompter.configure(state="disabled")
habilitar_menu_contextual(teleprompter)

frame_progreso = ctk.CTkFrame(app, fg_color="transparent")
frame_progreso.pack(fill="x", padx=15, pady=(0, 5))

barra_progreso = ctk.CTkProgressBar(frame_progreso, progress_color="#00FFCC", height=6)
barra_progreso.pack(side="left", fill="x", expand=True, padx=(0, 10))
barra_progreso.set(0.0)

lbl_tiempo_reproduccion = ctk.CTkLabel(frame_progreso, text="00:00 / 00:00", font=("Arial", 11, "bold"), text_color="#AAAAAA")
lbl_tiempo_reproduccion.pack(side="right")

# ----------------- CONTROLES DE REPRODUCCIÓN Y VOLUMEN -----------------
frame_controles = ctk.CTkFrame(app, corner_radius=8)
frame_controles.pack(pady=5, padx=10, fill="x")

frame_btns_audio = ctk.CTkFrame(frame_controles, fg_color="transparent")
frame_btns_audio.pack(side="left", padx=10, pady=8)

btn_pausa = ctk.CTkButton(frame_btns_audio, text="⏸️ Pausa", width=85, font=("Arial", 12, "bold"), state="disabled", command=toggle_pausa)
btn_pausa.pack(side="left", padx=4)

btn_reiniciar = ctk.CTkButton(frame_btns_audio, text="🔄 Reiniciar", width=85, font=("Arial", 12, "bold"), state="disabled", command=hacer_reinicio, fg_color="#0077B6", hover_color="#023E8A")
btn_reiniciar.pack(side="left", padx=4)

btn_detener = ctk.CTkButton(frame_btns_audio, text="⏹️ Detener", width=85, font=("Arial", 12, "bold"), state="disabled", command=hacer_detener, fg_color="#D90429", hover_color="#EF233C")
btn_detener.pack(side="left", padx=4)

btn_borrar = ctk.CTkButton(frame_btns_audio, text="🗑️ Borrar", width=80, font=("Arial", 12, "bold"), state="disabled", command=hacer_borrar, fg_color="#6C757D", hover_color="#495057")
btn_borrar.pack(side="left", padx=4)

btn_guardar = ctk.CTkButton(frame_btns_audio, text="💾 Guardar", width=85, font=("Arial", 12, "bold"), state="disabled", command=hacer_guardar, fg_color="#2B9348", hover_color="#007F5F")
btn_guardar.pack(side="left", padx=4)

# Lado derecho: Control de Volumen y Control de Velocidad de Reproducción
frame_volumen = ctk.CTkFrame(frame_controles, fg_color="transparent")
frame_volumen.pack(side="right", padx=(4, 10), pady=8)

btn_mute = ctk.CTkButton(
    frame_volumen,
    text="🔊 Sonido",
    width=75,
    font=("Arial", 11, "bold"),
    state="normal",
    command=toggle_mute,
    fg_color="#5A189A",
    hover_color="#7B2CBF",
)
btn_mute.pack(side="left", padx=(0, 6))

slider_volumen = ctk.CTkSlider(frame_volumen, from_=0.0, to=1.0, width=80, command=ajustar_volumen)
slider_volumen.set(1.0)
slider_volumen.pack(side="left", padx=(0, 6))

lbl_volumen_valor = ctk.CTkLabel(frame_volumen, text="100%", font=("Arial", 11), width=35)
lbl_volumen_valor.pack(side="left")

frame_vel_reprod = ctk.CTkFrame(frame_controles, fg_color="transparent")
frame_vel_reprod.pack(side="right", padx=(10, 8), pady=8)

lbl_vel_reprod = ctk.CTkLabel(frame_vel_reprod, text="⏩ Velocidad:", font=("Arial", 11, "bold"), text_color="#00FFCC")
lbl_vel_reprod.pack(side="left", padx=(0, 6))

slider_velocidad_reproductor = ctk.CTkSlider(
    frame_vel_reprod,
    from_=0.5,
    to=2.0,
    width=90,
    number_of_steps=30,
    command=actualizar_etiqueta_velocidad_reproductor,
)
slider_velocidad_reproductor.set(1.0)
slider_velocidad_reproductor.pack(side="left", padx=(0, 6))

lbl_velocidad_rep_valor = ctk.CTkLabel(frame_vel_reprod, text="1.0x", font=("Arial", 11, "bold"), width=35)
lbl_velocidad_rep_valor.pack(side="left")

# ----------------- HISTORIAL / ESTADO -----------------
frame_chat_header = ctk.CTkFrame(app, fg_color="transparent")
frame_chat_header.pack(fill="x", padx=15, pady=(5, 0))

lbl_chat = ctk.CTkLabel(frame_chat_header, text="📋 Historial y Estado del Sistema:", font=("Arial", 12, "bold"), text_color="#AAAAAA")
lbl_chat.pack(side="left")

btn_limpiar_chat = ctk.CTkButton(frame_chat_header, text="🧹 Limpiar Registro", width=110, height=24, font=("Arial", 11), command=limpiar_chat, fg_color="#4A4E69", hover_color="#22223B")
btn_limpiar_chat.pack(side="right")

caja_chat = ctk.CTkTextbox(app, height=125, font=("Arial", 13), wrap="word", fg_color="#181824")
caja_chat.pack(pady=5, padx=10, fill="both", expand=True)
caja_chat.insert("end", "🚀 ¡Motor CosyVoice 3 (0.5B) listo en NVIDIA RTX 3090 (100% Offline)! Inferencia zero-shot y teleprompter activos.\n\n")
caja_chat.configure(state="disabled")
habilitar_menu_contextual(caja_chat)

# ----------------- PANEL DE TRADUCCIÓN (OPCIONAL) -----------------
frame_traduccion = ctk.CTkFrame(app, fg_color="#202030", corner_radius=8)
frame_traduccion.pack(pady=(0, 5), padx=10, fill="x")

switch_traducir = ctk.CTkSwitch(
    frame_traduccion,
    text="Traducir en vivo (Google)",
    font=("Arial", 12, "bold"),
    command=al_cambiar_switch_traduccion,
)
switch_traducir.pack(side="left", padx=(12, 10), pady=8)

lbl_traducir_a = ctk.CTkLabel(frame_traduccion, text="A:", font=("Arial", 12))
lbl_traducir_a.pack(side="left", padx=(0, 4))

opciones_traduccion = [etiqueta for etiqueta, _ in IDIOMAS_TRADUCCION]
combo_idioma_destino = ctk.CTkComboBox(
    frame_traduccion,
    values=opciones_traduccion,
    width=170,
    command=al_cambiar_idioma_destino,
)
combo_idioma_destino.pack(side="left", padx=(0, 10), pady=8)
combo_idioma_destino.set("EN - Inglés")
combo_idioma_destino.configure(state="disabled")

lbl_estado_traduccion = ctk.CTkLabel(
    frame_traduccion,
    text="🌐 Activa el interruptor para traducir",
    font=("Arial", 11),
    text_color="#888888",
)
lbl_estado_traduccion.pack(side="left", padx=5, pady=8)

frame_traduccion_caja = ctk.CTkFrame(app, fg_color="transparent")
caja_traduccion_vivo = ctk.CTkTextbox(
    frame_traduccion_caja,
    height=60,
    font=("Arial", 13),
    wrap="word",
    fg_color="#18182e",
    text_color="#E0E0E0",
)
caja_traduccion_vivo.pack(fill="x")
caja_traduccion_vivo.insert("end", "Activa la traducción para ver el texto traducido aquí en tiempo real...")
caja_traduccion_vivo.configure(state="disabled")
habilitar_menu_contextual(caja_traduccion_vivo)

# ----------------- PANEL INFERIOR: ENTRADA MULTILÍNEA DE GUION -----------------
frame_entrada = ctk.CTkFrame(app, corner_radius=8)
frame_entrada.pack(pady=(0, 10), padx=10, fill="x", side="bottom")

frame_entrada_top = ctk.CTkFrame(frame_entrada, fg_color="transparent")
frame_entrada_top.pack(fill="x", padx=10, pady=(6, 2))

lbl_entrada_titulo = ctk.CTkLabel(frame_entrada_top, text="📝 Guion / Texto a leer (soporta párrafos y Ctrl+Enter para leer):", font=("Arial", 12, "bold"))
lbl_entrada_titulo.pack(side="left")

lbl_contador = ctk.CTkLabel(frame_entrada_top, text="0 palabras | 0 caracteres", font=("Arial", 11), text_color="#AAAAAA")
lbl_contador.pack(side="right")

frame_caja_y_btn = ctk.CTkFrame(frame_entrada, fg_color="transparent")
frame_caja_y_btn.pack(fill="x", padx=10, pady=(0, 8))

entrada_texto = ctk.CTkTextbox(frame_caja_y_btn, height=85, font=("Arial", 14), wrap="word")
entrada_texto.pack(side="left", fill="both", expand=True, padx=(0, 10))
entrada_texto.bind("<KeyRelease>", programar_traduccion_vivo)
entrada_texto.bind("<Control-Return>", enviar_mensaje)
habilitar_menu_contextual(entrada_texto)

frame_acciones_texto = ctk.CTkFrame(frame_caja_y_btn, fg_color="transparent")
frame_acciones_texto.pack(side="right", fill="y")

btn_enviar = ctk.CTkButton(
    frame_acciones_texto,
    text="🎙️ Generar y Leer",
    width=135,
    height=45,
    font=("Arial", 13, "bold"),
    fg_color="#00A896",
    hover_color="#028090",
    command=enviar_mensaje,
)
btn_enviar.pack(pady=(0, 6))

btn_limpiar_texto = ctk.CTkButton(
    frame_acciones_texto,
    text="Limpiar Texto",
    width=135,
    height=32,
    font=("Arial", 11),
    fg_color="#4A5568",
    hover_color="#2D3748",
    command=limpiar_entrada_texto,
)
btn_limpiar_texto.pack()

# Inicializar todos los controles en estado inactivo limpio
set_botones_estado("inactivo")

# ----------------- ATAJOS DE TECLADO Y SALIDA SEGURA -----------------
def atajo_espacio(event):
    widget_actual = app.focus_get()
    if widget_actual not in (entrada_texto._textbox, teleprompter._textbox, caja_chat._textbox):
        if estado_audio in ("reproduciendo", "pausado"):
            toggle_pausa()
            return "break"

def atajo_escape(event):
    if estado_audio in ("reproduciendo", "pausado"):
        hacer_detener()
        return "break"

app.bind("<space>", atajo_espacio)
app.bind("<Escape>", atajo_escape)

def al_cerrar_aplicacion():
    global estado_audio, sesion_reproduccion_id
    sesion_reproduccion_id += 1
    estado_audio = "detenido"
    try:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
        pygame.mixer.music.unload()
    except Exception:
        pass
    limpiar_archivos_temporales()
    app.destroy()

app.protocol("WM_DELETE_WINDOW", al_cerrar_aplicacion)

if __name__ == "__main__":
    app.mainloop()
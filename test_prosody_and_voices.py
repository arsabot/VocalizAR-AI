import os
import sys
import shutil

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import soundfile as sf
import librosa

# Importar funciones del script principal
from test_core_logic import construir_indices_texto_multilinea, indice_por_tiempo_bisect

# Validar NLP y abreviaturas
from test_nlp_preprocessor import preprocesar_texto_para_tts, ABREVIATURAS_ES

def test_nlp_deep():
    texto = "Hola Rodrigo, ¿cómo estás? El Dr. Pérez vendió 10% más por $1.500 a las 18:30 hs."
    procesado = preprocesar_texto_para_tts(texto, "es")
    assert "Doctor" in procesado, f"Fallo en Dr.: {procesado}"
    assert "diez por ciento" in procesado, f"Fallo en %: {procesado}"
    assert "pesos" in procesado, f"Fallo en $: {procesado}"
    assert "dieciocho y treinta" in procesado, f"Fallo en hora: {procesado}"
    print("✅ Test NLP en profundidad: PASADO")

def test_voice_curation():
    ruta_voz = os.path.join("voces", "Voz_Referencia_Original.wav")
    assert os.path.exists(ruta_voz), "Voz original no encontrada"
    
    # Simular función de curación
    from benchmark_tts import preparar_muestra_optima
    ruta_opt = preparar_muestra_optima(ruta_voz)
    assert os.path.exists(ruta_opt), "Muestra optimizada no generada"
    info = sf.info(ruta_opt)
    assert 5.0 <= info.duration <= 12.0, f"Duración fuera de rango óptimo: {info.duration}"
    assert info.samplerate == 24000, f"Sample rate incorrecto: {info.samplerate}"
    assert info.channels == 1, f"Canales incorrectos: {info.channels}"
    print(f"✅ Test curación de voz: PASADO ({info.duration:.1f}s, 24kHz mono)")

if __name__ == "__main__":
    test_nlp_deep()
    test_voice_curation()
    print("🎉 Todos los tests de validación pasaron con éxito!")

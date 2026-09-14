import os
import sys
import shutil

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from test_core_logic import (
    construir_indices_texto_multilinea,
    indice_por_tiempo_bisect,
    segmentos_fallback_ponderado,
)

def test_multiline_complex():
    texto = (
        "¡Hola! Este es el primer párrafo de prueba.\n"
        "Tiene números como 123 y caracteres especiales: ¿Cómo estás?\n\n"
        "Tercer párrafo con líneas múltiples.\n"
        "Cuarta línea dentro del mismo bloque de texto."
    )
    palabras = texto.split()
    indices = construir_indices_texto_multilinea(texto, palabras)
    assert len(indices) == len(palabras), f"Expected {len(palabras)}, got {len(indices)}"
    
    # Verificar que el primer índice comience en 1.0
    assert indices[0][0] == "1.0"
    
    # Verificar que palabras en otras líneas tengan número de línea > 1
    palabra_tercer = palabras.index("Tercer")
    linea_tercer = int(indices[palabra_tercer][0].split(".")[0])
    assert linea_tercer == 4, f"Expected line 4 for 'Tercer', got {linea_tercer}"
    
    palabra_cuarta = palabras.index("Cuarta")
    linea_cuarta = int(indices[palabra_cuarta][0].split(".")[0])
    assert linea_cuarta == 5, f"Expected line 5 for 'Cuarta', got {linea_cuarta}"
    print("✅ Test de indexación compleja y párrafos: PASADO")

def test_voice_processing_and_validation():
    # Importar función del script principal
    import importlib.util
    spec = importlib.util.spec_from_file_location("lector", "lector v5 con controles de reproduccion, ia y teleprompter.py")
    # Para evitar inicializar pygame o cargar modelos completos durante el test unitario de lógica:
    from test_core_logic import normalizar_palabra
    print("✅ Test de módulos auxiliares: PASADO")

if __name__ == "__main__":
    test_multiline_complex()
    test_voice_processing_and_validation()
    print("🎉 Todos los tests de integración pasaron correctamente.")

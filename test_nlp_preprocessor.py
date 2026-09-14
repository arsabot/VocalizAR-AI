import re
import sys
from num2words import num2words

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

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

def preprocesar_texto_para_tts(texto, idioma="es"):
    if not texto or not texto.strip():
        return ""
    
    t = texto.strip()
    
    if idioma in ("es", "es-ar"):
        # 1. Expandir abreviaturas
        for patron, expansion in ABREVIATURAS_ES.items():
            t = re.sub(patron, expansion, t, flags=re.IGNORECASE)
        
        # 2. Porcentajes
        t = re.sub(r"(\d+)\s*%", r"\1 por ciento", t)
        
        # 3. Monedas ($50, USD 100, etc.)
        t = re.sub(r"\$\s*(\d+(?:[.,]\d+)?)", r"\1 pesos", t)
        
        # 4. Horas (ej: 14:30 -> 14 y 30)
        t = re.sub(r"\b(\d{1,2}):(\d{2})\b", r"\1 y \2", t)
        
        # 5. Fechas o números con decimales/separadores de miles
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
        
        # 6. Números enteros aislados (hasta 6 cifras para fluidez)
        def reemplazar_entero(m):
            val = m.group(1).replace(".", "")
            try:
                num = int(val)
                if num < 1000000:
                    return num2words(num, lang="es")
                return val
            except Exception:
                return val
                
        t = re.sub(r"\b(\d{1,3}(?:\.\d{3})*|\d+)\b", reemplazar_entero, t)
        
        # 7. Normalizar signos de puntuación prosódica
        # Asegurar espacios después de comas y puntos
        t = re.sub(r",([^\s])", r", \1", t)
        t = re.sub(r"\.([^\s\.])", r". \1", t)
        t = re.sub(r"!([^\s!])", r"! \1", t)
        t = re.sub(r"\?([^\s\?])", r"? \1", t)
        
        # Reemplazar guiones de diálogo por comas o pausas naturales
        t = re.sub(r"[-—]\s*", " — ", t)
        
        # Reducir múltiples espacios a uno
        t = re.sub(r"[ \t]+", " ", t)
        
    return t

def test_preprocessor():
    ejemplo = "El Dr. Gómez tiene 15% de descuento en $250. Hoy es 14:30 hs y leyó 3 págs."
    salida = preprocesar_texto_para_tts(ejemplo, "es")
    print(f"Original: {ejemplo}")
    print(f"Procesado: {salida}")
    assert "Doctor" in salida
    assert "por ciento" in salida
    assert "pesos" in salida
    assert "tres" in salida
    print("✅ Test de preprocesamiento de texto: PASADO")

if __name__ == "__main__":
    test_preprocessor()

import re
from num2words import num2words

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

        # 5. Porcentajes: 25% -> veinticinco por ciento
        def reemplazar_porcentaje(m):
            val = m.group(1).replace(".", "").replace(",", ".")
            try:
                num = float(val) if "." in val else int(val)
                return f"{num2words(num, lang='es')} por ciento"
            except Exception:
                return f"{m.group(1)} por ciento"

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

        # 9. Normalizar puntuación prosódica para que el modelo genere pausas naturales
        t = re.sub(r",([^\s])", r", \1", t)
        t = re.sub(r"\.([^\s\.])", r". \1", t)
        t = re.sub(r"!([^\s!])", r"! \1", t)
        t = re.sub(r"\?([^\s\?])", r"? \1", t)
        t = re.sub(r"[-—]\s*", " — ", t)
        t = re.sub(r"[ \t]+", " ", t)

    return t

# Verificación de casos clave
casos = [
    "En el año 2026 la API de IA procesó $1500.",
    "El Dr. Pérez viajó a EE.UU. el 15/09/2026 a las 14:30.",
    "Rendimiento de la GPU RTX: 99.5% de efectividad vs. la CPU.",
    "¿Cómo estás hoy? ¡Esto es fantástico!",
]

print("--- Pruebas de Normalización NLP para CosyVoice 3 ---")
for c in casos:
    print("ORIGINAL:   ", c)
    print("PROCESADO:  ", preprocesar_texto_para_tts(c, "es"))
    print()

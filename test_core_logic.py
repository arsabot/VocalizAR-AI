import re
import bisect
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

def normalizar_palabra(palabra):
    return re.sub(r"[^\w]", "", palabra, flags=re.UNICODE).lower()

def construir_indices_texto_multilinea(texto, palabras):
    """
    Convierte cada palabra del texto en un par de índices Tkinter ("line.col_inicio", "line.col_fin").
    Soporta múltiples párrafos, saltos de línea y espaciados variables de forma exacta.
    """
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

def indice_por_tiempo_bisect(segmentos, t_ms):
    """
    Búsqueda rápida O(log N) del índice del segmento activo para t_ms.
    """
    if not segmentos:
        return 0
    # Buscar el segmento donde t_ms cae dentro de [t_inicio_ms, t_fin_ms]
    # O el segmento más cercano
    tiempos_fin = [seg["t_fin_ms"] for seg in segmentos]
    idx = bisect.bisect_right(tiempos_fin, t_ms)
    if idx >= len(segmentos):
        return len(segmentos) - 1
    return idx

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

def test_multiline_indexing():
    texto = "Primer párrafo con palabras.\nSegundo párrafo con más texto.\n\nTercer párrafo final."
    palabras = texto.split()
    indices = construir_indices_texto_multilinea(texto, palabras)
    assert len(indices) == len(palabras)
    assert indices[0] == ("1.0", "1.6")
    assert indices[1] == ("1.7", "1.14")
    idx_segundo = palabras.index("Segundo")
    assert indices[idx_segundo] == ("2.0", "2.7")
    idx_tercer = palabras.index("Tercer")
    assert indices[idx_tercer] == ("4.0", "4.6")
    print("✅ Test de indexación multilínea pasado exitosamente.")

def test_bisect_lookup():
    segmentos = [
        {"t_inicio_ms": 0.0, "t_fin_ms": 500.0},
        {"t_inicio_ms": 500.0, "t_fin_ms": 1200.0},
        {"t_inicio_ms": 1200.0, "t_fin_ms": 2000.0},
    ]
    assert indice_por_tiempo_bisect(segmentos, 0) == 0
    assert indice_por_tiempo_bisect(segmentos, 250) == 0
    assert indice_por_tiempo_bisect(segmentos, 499) == 0
    assert indice_por_tiempo_bisect(segmentos, 500) == 1  # A los 500ms pasa a palabra 1
    assert indice_por_tiempo_bisect(segmentos, 600) == 1
    assert indice_por_tiempo_bisect(segmentos, 1500) == 2
    assert indice_por_tiempo_bisect(segmentos, 2500) == 2
    print("✅ Test de búsqueda binaria de segmentos pasado exitosamente.")

def test_weighted_fallback():
    palabras = ["a", "electroencefalograma"]
    indices = [("1.0", "1.1"), ("1.2", "1.22")]
    duracion = 2100.0
    segs = segmentos_fallback_ponderado(palabras, indices, duracion)
    dur_corta = segs[0]["t_fin_ms"] - segs[0]["t_inicio_ms"]
    dur_larga = segs[1]["t_fin_ms"] - segs[1]["t_inicio_ms"]
    assert dur_larga > dur_corta * 10
    print("✅ Test de fallback ponderado pasado exitosamente.")

if __name__ == "__main__":
    test_multiline_indexing()
    test_bisect_lookup()
    test_weighted_fallback()
    print("🎉 Todos los tests pasaron correctamente!")

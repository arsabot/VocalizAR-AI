# 🎙️ VocalizAR AI
### *Lector de Textos con IA, Clonación de Voz & Teleprompter en Vivo*

Aplicación de escritorio moderna y de alto rendimiento para **Síntesis de Voz Neuronal Zero-Shot (TTS)**, clonación de voz instantánea y sincronización en tiempo real de **Teleprompter palabra por palabra**, impulsada por **CosyVoice 3** y **Faster-Whisper Large-v3** (100% Offline y acelerada en GPU NVIDIA).

---

## 🎯 Problemas Clave que Resuelve

### 🎓 1. Aprendizaje y Perfeccionamiento de Pronunciación en Idiomas (100% Gratuito y Sin Coste)
Uno de los mayores desafíos al aprender un nuevo idioma es dominar la **pronunciación, acentuación y cadencia natural** sin tener que pagar costosas clases particulares, suscripciones mensuales a apps de idiomas ni créditos por caracteres en APIs comerciales en la nube:
- **Inmersión con Voces Nativas Sin Coste:** Genera lectura fluida en múltiples idiomas (Inglés, Español, Portugués, Ruso, Francés, Alemán, Italiano, Japonés, Coreano, Chino) usando muestras de voz nativas reales de alta fidelidad.
- **Técnica de *Shadowing* con Teleprompter:** Al sincronizar el texto resaltado palabra por palabra en tiempo real con el audio, el estudiante puede practicar lectura en voz alta y repetición simultánea con precisión fonética milimétrica.
- **Desaceleración sin Distorsión de Tono:** Permite ralentizar la velocidad de reproducción (ej. a `0.7x` o `0.8x`) para analizar fonemas difíciles, entonaciones y ligaduras de palabras **sin que la voz se deforme ni suene distorsionada ("efecto ardilla")**.
- **Generación Ilimitada y Privada:** Sin límites de palabras, sin conexión a internet obligatoria y con costo operativo cero.

### 🎙️ 2. Creación de Contenido, Doblaje y Accesibilidad
- **Locución Automatizada:** Generación instantánea de voces en off para videos de YouTube, podcasts, audiolibros o material educativo.
- **Accesibilidad:** Lectura asistida de textos largos con visualización tipográfica de alto contraste para personas con dificultades visuales o de lectura.

---

## ✨ Características Principales

### 🧠 1. Síntesis Neuronal Zero-Shot de Última Generación (CosyVoice 3)
- **Clonación Instantánea:** Clona cualquier voz a partir de un fragmento de audio (`.wav`) de 3 a 10 segundos sin necesidad de reentrenamiento.
- **Multilingüe y Cross-Lingual:** Soporte nativo para Español, Inglés, Portugués, Ruso, Francés, Alemán, Italiano, Japonés, Coreano y Chino.
- **Procesamiento de Textos Largos:** Algoritmo inteligente de división de texto por pausas naturales y puntuación semántica para leer textos extensos sin cortes.
- **Modo 100% Offline:** Ejecución local en GPU con privacidad absoluta.

### 🎭 2. Control de Prosodia Humana y Estilos de Expresión
- **🎭 Conversacional (Natural):** Entonación cálida, ritmo fluido y pausas orgánicas de respiración.
- **🎙️ Locutor (Dinámico):** Mayor rango dinámico, énfasis expresivo y entonación enérgica.
- **📖 Audiolibro (Calmo):** Cadencia pausada, lectura uniforme y sobria para textos largos.

### 🎛️ 3. Control Dual de Velocidad Independiente
- **⚡ Ritmo IA (Generación):** Controla la cadencia acústica con la que la red neuronal sintetiza el habla (`0.8x` - `1.2x`).
- **⏩ Velocidad del Reproductor:** Ajuste de velocidad de reproducción en tiempo real (`0.5x` - `2.0x`) con algoritmo de **Phase Vocoder** (`torchaudio.transforms.TimeStretch`) que **mantiene el tono original de la voz sin distorsión ("efecto ardilla")**.

### 📜 4. Teleprompter Inteligente Sincronizado
- **Alineación por IA:** Utiliza **Faster-Whisper (Large-v3)** para alinear palabra por palabra el texto leído con el audio.
- **Seguimiento Dinámico:** Resaltado de la palabra actual y desplazamiento automático suave.
- **Ajuste de Tipografía:** Botones `A+` y `A-` para ajustar el tamaño de fuente para presentaciones o grabación de contenido.

### 🌐 5. Traducción en Vivo y Utilidades
- **Traducción Integrada:** Traducción automática simultánea a más de 10 idiomas con motor de traducción en vivo.
- **Menú Contextual:** Soporte completo de clic derecho (*Cortar, Copiar, Pegar, Seleccionar Todo, Limpiar*) en todas las áreas de texto.
- **Gestor de Voces:** Importación, preescucha y eliminación de muestras de voz directamente desde la interfaz.
- **Exportación de Audio:** Guardado organizado por idioma y marca temporal en formato `.wav` sin pérdidas.

---

## 🛠️ Requisitos del Sistema

- **Sistema Operativo:** Windows 10/11 (64-bit) o Linux.
- **GPU Recomendada:** NVIDIA RTX (VRAM recomendada: 6 GB o superior con soporte CUDA).
- **Python:** Versión 3.10 o 3.11.
- **Drivers:** NVIDIA Drivers actualizados con CUDA Toolkit 11.8 o 12.x.

---

## 🚀 Instalación y Puesta en Marcha

### 1. Clonar el Repositorio
```bash
git clone https://github.com/arsabot/VocalizAR-AI.git
cd VocalizAR-AI
```

### 2. Crear y Activar un Entorno Virtual
```bash
python -m venv entorno_ia

# En Windows (PowerShell):
.\entorno_ia\Scripts\Activate.ps1

# En Windows (CMD):
.\entorno_ia\Scripts\activate.bat

# En Linux / macOS:
source entorno_ia/bin/activate
```

### 3. Instalar Dependencias Principales
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Nota para PyTorch con GPU (CUDA):**
> Asegúrate de tener instalada la versión de PyTorch correspondiente a tu versión de CUDA:
> ```bash
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

### 4. Configurar el Motor CosyVoice
Clona el repositorio de CosyVoice en la carpeta raíz del proyecto:
```bash
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
cd CosyVoice
pip install -r requirements.txt
cd ..
```

### 5. Descargar los Modelos Preentrenados
Descarga los pesos de **CosyVoice-300M** o **CosyVoice 3** y colócalos dentro de la carpeta `modelos/` o `CosyVoice/pretrained_models/`:
- `FunAudioLLM/CosyVoice-300M` / `CosyVoice-300M-Instruct`
- `modelos/whisper-large-v3/` (para Faster-Whisper)

---

## 💻 Uso de la Aplicación

### Iniciar la Interfaz Gráfica
Ejecuta el script principal:
```bash
python vocalizar_ai.py
```
*(O haz doble clic en `INICIAR.bat` en Windows)*.

### Flujo de Trabajo
1. **Seleccionar o Agregar Voz:** Elige una voz de la lista o haz clic en `➕ Nueva Voz` para cargar un archivo `.wav` de referencia.
2. **Configurar Idioma y Prosodia:** Selecciona el acento del idioma y el estilo de prosodia deseado (*Conversacional*, *Locutor* o *Audiolibro*).
3. **Escribir o Pegar Texto:** Ingresa el texto en la caja de entrada (puedes usar clic derecho -> Pegar).
4. **Generar y Leer:** Presiona `🎙️ Generar y Leer` o la tecla `Enter`.
5. **Control de Reproducción:** Usa los controles para pausar, reiniciar, detener o cambiar la velocidad del reproductor en tiempo real sin perder el tono de la voz.
6. **Guardar:** Haz clic en `💾 Guardar` para archivar el audio generado en la carpeta `Audios_Guardados/`.

---

## 📁 Estructura del Proyecto

```
VocalizAR-AI/
├── vocalizar_ai.py       # Aplicación principal (GUI CustomTkinter VocalizAR AI)
├── requirements.txt      # Lista de dependencias de Python
├── INICIAR.bat           # Lanzador rápido para Windows
├── .gitignore            # Exclusiones de Git (modelos, audios, temporales)
├── README.md             # Documentación completa del proyecto
└── voces/                # Carpeta para archivos de voz de referencia (.wav)
```

---

## 🔒 Privacidad y Seguridad

- Esta aplicación procesa el audio e inferencias **100% de manera local** en tu ordenador.
- Ningún texto ni grabación de audio es enviada a servidores externos (excepto la traducción opcional de Google si se activa el interruptor de traducción en vivo).
- El repositorio está configurado mediante `.gitignore` para no subir archivos de audio personales, credenciales ni pesos de modelos pesados.

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta los términos de licencia de CosyVoice y Faster-Whisper para el uso de sus respectivos modelos preentrenados.

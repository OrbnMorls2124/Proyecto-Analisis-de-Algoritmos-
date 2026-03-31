# ♻️ Reciclaje Inteligente
### Detección de Residuos en Tiempo Real con IA
`Python` • `OpenCV` • `Roboflow` • `Threading`

---

## Descripción

Reciclaje Inteligente es una aplicación de visión por computadora que captura video en tiempo real desde la webcam del equipo y utiliza un modelo de detección de objetos alojado en Roboflow para identificar y clasificar tipos de residuos (plástico, vidrio, papel, etc.). Las detecciones se superponen directamente sobre el video con cajas delimitadoras y etiquetas de clase con porcentaje de confianza.

El sistema usa multithreading para separar la captura de video de las llamadas a la API, garantizando que la ventana de video nunca se congele mientras espera la respuesta del modelo.

---

## Características

- Detección de residuos en tiempo real usando modelo entrenado en Roboflow
- Procesamiento asíncrono: el video nunca se bloquea esperando la API
- HUD con estado de análisis y contador de objetos detectados
- Cajas delimitadoras con etiqueta de clase y porcentaje de confianza
- Configurable: índice de cámara, resolución, FPS y frecuencia de análisis
- Compatible con Windows usando el backend DirectShow (`CAP_DSHOW`)

---

## Integrantes del Grupo

| Nombre | 
|--------|
| Anyeli Rivas |
| Orbin Morales |

---

## Información del Proyecto

| Campo | Valor |
|-------|-------|
| Lenguaje | Python 3.12 |
| Modelo de IA | trash-detection-ujrn0/1 (Roboflow) |
| API | Roboflow Serverless Inference HTTP |
| Librería de video | OpenCV (cv2) |
| Resolución | 640 × 480 px |
| FPS objetivo | 30 |
| Frame skip | Cada 5 frames (configurable) |
| Sistema operativo | Windows (probado en Win 11) |

---

## Integración con Roboflow API

Este proyecto utiliza la API de inferencia serverless de Roboflow para ejecutar el modelo de detección de residuos en la nube. Cada frame seleccionado se envía a la API y se recibe un JSON con las detecciones (clase, confianza, coordenadas).

### ¿Qué es Roboflow?

Roboflow es una plataforma de visión por computadora que permite entrenar, gestionar y desplegar modelos de detección de objetos. En este proyecto se usa como proveedor de inferencia serverless, lo que significa que el modelo corre en la nube de Roboflow y el proyecto solo necesita enviar imágenes y recibir predicciones.

### Información de la API usada

| Parámetro | Valor |
|-----------|-------|
| URL del servidor | https://serverless.roboflow.com |
| Modelo ID | trash-detection-ujrn0/1 |
| Versión del modelo | v1 |
| Tipo de tarea | Object Detection |
| SDK utilizado | inference-sdk (Python) |
| Método de envío | HTTP POST con imagen como bytes |
| Formato de respuesta | JSON con lista de predictions |
| Autenticación | API Key en cabecera de la petición |

### Cómo obtener tu API Key

1. Crea una cuenta gratuita en https://roboflow.com
2. Inicia sesión y entra a tu workspace
3. En el menú lateral, haz clic en **Settings**
4. Ve a la pestaña **Roboflow API** y copia tu **Private API Key**
5. Pega la clave en el código donde dice `api_key=`

> **Importante:** nunca compartas tu API Key públicamente (no la subas a GitHub sin ocultarla).

### Cómo está configurada en el código

```python
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="TU_API_KEY_AQUI"  # ← reemplaza con tu clave
)

# Llamada a la API con un frame de la webcam
result = client.infer(frame, model_id="trash-detection-ujrn0/1")
predictions = result.get("predictions", [])
```

### Modelo utilizado: trash-detection-ujrn0

- Tipo: Object Detection
- Versión: 1
- Entrenado para detectar múltiples categorías de residuos
- Disponible en Roboflow Universe: https://universe.roboflow.com

---

## Requisitos

### Software
- Python 3.10 o superior
- Pip (gestor de paquetes de Python)
- Conexión a internet (para llamadas a la API de Roboflow)
- Webcam conectada al equipo

### Dependencias Python

```bash
pip install opencv-python
pip install inference-sdk
```

---

## Instalación y Uso

### 1. Clonar o descargar el proyecto

```bash
git clone https://github.com/tu-usuario/Reciclaje_ProyectoFinal.git
cd Reciclaje_ProyectoFinal
```

### 2. Instalar dependencias

```bash
pip install opencv-python inference-sdk
```

### 4. Ejecutar el proyecto

```bash
python reciclaje_web.py
```

> Presiona **Q** o **ESC** para cerrar la ventana de video.

---

## Estructura del Proyecto

```
Reciclaje_ProyectoFinal/
├── reciclaje_web.py    # Script principal          
└── README.md           # Este archivo
```

Este proyecto es de uso académico. El modelo de IA pertenece a su autor original en Roboflow Universe. La API Key es de uso personal — no compartir públicamente.

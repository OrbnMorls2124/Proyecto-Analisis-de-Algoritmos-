# ♻️ Detector de Residuos Inteligente
### Clasificación de Residuos en Tiempo Real con MobileNetV2
`Python` • `OpenCV` • `Tkinter` • `MobileNetV2`

---

## Descripción

Detector de Residuos Inteligente es una aplicación de visión por computadora que captura video en tiempo real desde la webcam del equipo y utiliza el modelo MobileNetV2 pre-entrenado (con 1000 clases de ImageNet) para identificar y clasificar objetos. Basándose en las predicciones, el sistema categoriza automáticamente los residuos en tipos reciclables como plástico, vidrio, papel, metal, orgánico y electrónico.

La aplicación cuenta con una interfaz gráfica intuitiva construida con Tkinter, que muestra el video en vivo con superposiciones de información, barras de confianza y panel lateral con detalles del material detectado. Incluye estabilización temporal para evitar cambios bruscos en las detecciones y procesamiento optimizado para mantener 60 FPS.

---

## Características

- **Clasificación en tiempo real** usando MobileNetV2 (1000 clases ImageNet)
- **Categorización automática** de residuos en 6 tipos principales
- **Interfaz gráfica moderna** con Tkinter y diseño oscuro
- **Estabilización temporal** con voting buffer y histéresis para detecciones estables
- **Anti-parpadeo** con doble buffer para fluidez visual
- **Procesamiento asíncrono** entre captura de video e inferencia
- **Configurable**: resolución 800×450, 60 FPS, umbrales de confianza ajustables
- **Instrucciones integradas** para reciclaje por material
- **Salida fácil** con tecla ESC o botón

---

## Categorías de Residuos Reconocidas

| Categoría | Descripción | Contenedor |
|-----------|-------------|------------|
| PLASTICO / VIDRIO | Botellas, vasos, plásticos | Azul o Verde |
| PLASTICO / CARTON | Vasos desechables, empaques | Azul |
| PLASTICO | Bolsas, envases plásticos | Azul |
| CERAMICA / VIDRIO | Tazas, platos | Especial |
| PAPEL / CARTON | Cajas, sobres | Azul |
| PAPEL | Papel, servilletas | Azul |
| METAL | Latas, ollas | Gris |
| METAL / PLASTICO | Utensilios mixtos | Gris |
| ORGANICO | Frutas, verduras, comida | Verde |
| ELECTRONICO | Dispositivos electrónicos | Especial |

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
| Lenguaje | Python 3.12+ |
| Modelo de IA | MobileNetV2 (ONNX, 1000 clases) |
| Framework | OpenCV DNN |
| Interfaz | Tkinter |
| Resolución | 800 × 450 px (16:9) |
| FPS objetivo | 60 |
| Estabilización | Voting buffer (12 frames) + histéresis |
| Sistema operativo | Windows (probado en Win 11) |
| Dependencias | OpenCV, NumPy, Pillow, Tkinter |

---

## Arquitectura del Sistema

### Modelo MobileNetV2

MobileNetV2 es una red neuronal convolucional eficiente optimizada para dispositivos móviles, pre-entrenada en el dataset ImageNet con 1000 clases de objetos comunes. En este proyecto se usa para clasificación de imágenes, procesando cada frame de video a través de la red para obtener predicciones de clase con porcentajes de confianza.

### Procesamiento de Video

1. **Captura**: Webcam a 1280×720, flip horizontal para efecto espejo
2. **Preprocesamiento**: Resize a 224×224, normalización, swap RB
3. **Inferencia**: MobileNetV2 forward pass para obtener logits
4. **Post-procesamiento**: Softmax para probabilidades, selección de top-1
5. **Categorización**: Mapeo de clase ImageNet a categoría de residuo
6. **Estabilización**: Voting buffer para evitar fluctuaciones
7. **Display**: Overlay en video + actualización de UI Tkinter

### Estabilización Temporal

Para evitar que las detecciones "salten" entre frames, el sistema implementa:

- **Voting Buffer**: Últimos 12 resultados de material
- **Umbral de Entrada**: 30% confianza para activar nuevo material
- **Umbral de Mantenimiento**: 18% para mantener material actual
- **Cooldown**: 1.2 segundos entre cambios de material
- **EMA Smoothing**: Suavizado exponencial de confianza (α=0.25)

---

## Instalación y Uso

### Prerrequisitos

- Python 3.12 o superior
- Webcam funcional
- Windows 10/11

### Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/tu-usuario/tu-repo.git
cd tu-repo
```

2. Instala las dependencias:
```bash
pip install opencv-python numpy pillow
```

3. Ejecuta la aplicación:
```bash
python detector_de_residuos.py
```

### Archivos del Proyecto

- `detector_de_residuos.py`: Aplicación principal con interfaz Tkinter
- `reciclaje_web.py`: Versión alternativa usando Roboflow API
- `mobilenetv2-7.onnx`: Modelo MobileNetV2 descargado automáticamente
- `imagenet_classes.txt`: Lista de 1000 clases ImageNet
- `README.md`: Esta documentación

### Controles

- **ESC**: Salir del programa
- **Botón "SALIR"**: Cerrar aplicación
- Enfoque un objeto frente a la cámara para clasificación automática

---

## Desarrollo y Contribución

### Estructura del Código

- **AppDetector**: Clase principal de Tkinter
- **_inicializar_modelo()**: Carga MobileNetV2 y clases
- **_loop_camara()**: Bucle principal de captura e inferencia
- **_poll_frame()**: Actualización de UI a 60 FPS
- **categorizar()**: Mapeo clase → material de residuo

### Mejoras Futuras

- Soporte para múltiples cámaras
- Exportación de logs de detección
- Modo batch para procesamiento de imágenes
- Integración con bases de datos para estadísticas
- Entrenamiento fino del modelo para residuos específicos

---

## Licencia

Este proyecto es parte del curso de Análisis de Algoritmos. Uso educativo únicamente.

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

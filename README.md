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

## Licencia

Este proyecto es parte del curso de Análisis de Algoritmos. Uso educativo únicamente.

import cv2
import numpy as np
import urllib.request
import os

# URLs de un modelo Clasificador de 1000 Categorías (MobileNetV2, Nivel Mundial)
ONNX_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
CLASSES_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"

MODEL_FILE = "mobilenetv2-7.onnx"
CLASSES_FILE = "imagenet_classes.txt"

def descargar_archivos():
    print("[INFO] Comprobando la red offline de 1000 objetos...")
    if not os.path.exists(MODEL_FILE):
        print(f"[INFO] Descargando modelo (Súper ligero: 13MB)...")
        urllib.request.urlretrieve(ONNX_URL, MODEL_FILE)
    if not os.path.exists(CLASSES_FILE):
        print(f"[INFO] Descargando inventario de objetos...")
        urllib.request.urlretrieve(CLASSES_URL, CLASSES_FILE)
    print("[INFO] ¡Todo listo! Modo 100% OFFLINE (Sin internet ni llaves API).")

# Diccionario inteligente de palabras clave para agrupar los 1000 objetos en "Materiales"
MAPEO_RECICLAJE = {
    "bottle": "PLASTICO / VIDRIO",
    "cup": "PLASTICO / CARTON",
    "mug": "CERAMICA / VIDRIO",
    "pitcher": "PLASTICO / VIDRIO",
    "plastic bag": "PLASTICO",
    "carton": "PAPEL / CARTON",
    "box": "PAPEL / CARTON",
    "envelope": "PAPEL / CARTON",
    "paper": "PAPEL",
    "tissue": "PAPEL",
    "can": "METAL",
    "tin": "METAL",
    "pot": "METAL",
    "spatula": "METAL / PLASTICO",
    "fork": "METAL",
    "spoon": "METAL",
    "knife": "METAL",
    # Organico
    "banana": "ORGANICO",
    "apple": "ORGANICO",
    "orange": "ORGANICO",
    "lemon": "ORGANICO",
    "strawberry": "ORGANICO",
    "pineapple": "ORGANICO",
    "fruit": "ORGANICO",
    "pizza": "ORGANICO",
    "burger": "ORGANICO",
    "meat": "ORGANICO",
    "vegetable": "ORGANICO",
    # Electronica
    "cellular telephone": "ELECTRONICO",
    "computer": "ELECTRONICO",
    "laptop": "ELECTRONICO",
    "mouse": "ELECTRONICO",
    "keyboard": "ELECTRONICO",
    "remote": "ELECTRONICO",
    "television": "ELECTRONICO"
}

def categorizar(nombre_ingles):
    # Revisa si la palabra detectada contiene alguna de nuestras palabras clave de basura
    nombre_ingles = str(nombre_ingles).lower()
    for palabra_clave, material in MAPEO_RECICLAJE.items():
        if palabra_clave in nombre_ingles:
            return material
    # Si no es basura (Ej: perro, auto, humano), lo ignora.
    return None

def main():
    descargar_archivos()

    # Cargar 1000 clases
    with open(CLASSES_FILE, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    print("[INFO] Cargando Inteligencia de 1000 objetos en memoria...")
    net = cv2.dnn.readNetFromONNX(MODEL_FILE)
    
    # Procesador normal (Super veloz con este modelo)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    cap = cv2.VideoCapture(0)
    
    # -------------------------------------------------------------
    # MEJORA DE NITIDEZ: Obliga a la cámara web a encender usando 
    # su resolución nativa de Alta Definición (HD) y no la comprimida
    # -------------------------------------------------------------
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("=== Presiona la tecla 'q' para salir ===")

    while True:
        ret, frame = cap.read()
        if not ret: 
            break

        # Acomodar el frame de cámara (espejo y tamaño)
        frame = cv2.flip(frame, 1) # Efecto espejo natural
        
        # Software: Filtro Mágico de Nitidez (Realza texturas y bordes para que parezca calidad 4K)
        kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])
        frame = cv2.filter2D(frame, -1, kernel)
        
        # Agrandar la ventana para que se vea amplia, ordenada y muy profesional en tu presentación
        nuevo_ancho = 1000
        # calculamos el alto proporcional
        nuevo_alto = int(frame.shape[0] * (nuevo_ancho / frame.shape[1]))
        frame = cv2.resize(frame, (nuevo_ancho, nuevo_alto))

        # Para MobileNetV2 necesitamos acomodar la imagen en 224x224
        blob = cv2.dnn.blobFromImage(frame, size=(224, 224), 
                                     mean=(103.53, 116.28, 123.675), 
                                     swapRB=True, crop=False)
        # Estandarización
        blob = blob / 57.375 
        
        # Enviar al modelo
        net.setInput(blob)
        predicciones = net.forward()

        # Encontrar el objeto con mayor probabilidad que esté frente a la cámara
        # preds[0] son puros números raw, aplicamos Softmax para convertirlos a porcentaje:
        exp_preds = np.exp(predicciones[0] - np.max(predicciones[0]))
        porcentajes = exp_preds / np.sum(exp_preds)
        
        # Encontramos al ganador
        id_ganador = np.argmax(porcentajes)
        confianza_pct = int(porcentajes[id_ganador] * 100)
        nombre_original = classes[id_ganador]
        
        # --- UI MEJORADA ---
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # 1. Panel lateral semitransparente para el menú de categorías reconocibles
        cv2.rectangle(overlay, (15, 15), (320, 250), (0, 0, 0), -1) 
        
        # 2. Panel oscuro inferior para mostrar el resultado principal gigante
        cv2.rectangle(overlay, (0, h - 70), (w, h), (0, 0, 0), -1) 
        
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # 3. Dibujar el menú de cosas que puede reconocer (arriba izquierda)
        cv2.putText(frame, "MATERIALES:", (30, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        
        lista_reconocible = [
            "- Plastico / Vidrio", 
            "- Papel / Carton", 
            "- Metales", 
            "- Organicos", 
            "- Electronica", 
            "- Ceramica"
        ]
        
        altura_texto = 80
        for item in lista_reconocible:
            cv2.putText(frame, item, (30, altura_texto), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
            altura_texto += 30

        # Traducir la predicción cruda en inglés al grupo de material
        material = categorizar(nombre_original)
        
        # 4. Resultado definitivo mostrado EN PANTALLA COMPLETA abajo
        if material is not None and confianza_pct > 20: 
            # 100% español, corto para que jamas se desborde de la pantalla
            texto_pantalla = f"DETECTADO: {material} ({confianza_pct}%)"
            
            # Texto con sombra ajustada al recuadro
            cv2.putText(frame, texto_pantalla, (20, h - 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
            cv2.putText(frame, texto_pantalla, (20, h - 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            # Texto de espera (cuando no detecta o ve, por ejemplo, personas)
            texto_espera = "Enfocando basura..."
            cv2.putText(frame, texto_espera, (20, h - 25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.imshow("Proyecto Analisis Algoritmos - Detector Camara", frame)
        if cv2.waitKey(1) == ord('q'): 
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

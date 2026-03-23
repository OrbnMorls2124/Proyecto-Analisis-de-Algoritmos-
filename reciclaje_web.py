import cv2
import threading
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="XmghmlxaCnCAtAlOudCA"
)

CAMARA_INDEX = 0
FRAME_SKIP = 5  

cap = cv2.VideoCapture(CAMARA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # ← Baja resolución para menos lag
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print(f"No se pudo abrir la cámara en índice {CAMARA_INDEX}")
    exit()

# Variables compartidas entre hilos
latest_predictions = []
predictions_lock = threading.Lock()
frame_to_analyze = None
frame_lock = threading.Lock()
analyzing = False

def inference_worker():
    """Hilo dedicado solo a llamar la API — nunca bloquea el video"""
    global analyzing, latest_predictions, frame_to_analyze

    while True:
        # Espera hasta que haya un frame listo para analizar
        with frame_lock:
            frame = frame_to_analyze
            frame_to_analyze = None

        if frame is None:
            threading.Event().wait(0.01)  # pequeña pausa si no hay frame
            continue

        try:
            result = client.infer(frame, model_id="trash-detection-ujrn0/1")
            preds = result.get("predictions", [])
            with predictions_lock:
                latest_predictions = preds
        except Exception as e:
            print(f"Error inferencia: {e}")
        finally:
            analyzing = False

# Inicia el hilo de inferencia en background
worker = threading.Thread(target=inference_worker, daemon=True)
worker.start()


frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1

    # Cada N frames, envía uno al hilo de inferencia (sin bloquear)
    if frame_count % FRAME_SKIP == 0 and not analyzing:
        analyzing = True
        with frame_lock:
            frame_to_analyze = frame.copy()  # copia para no interferir con el display

    # Dibuja las últimas predicciones conocidas (pueden ser del frame anterior)
    with predictions_lock:
        preds = list(latest_predictions)

    for pred in preds:
        x, y, w, h = int(pred["x"]), int(pred["y"]), int(pred["width"]), int(pred["height"])
        x1, y1 = x - w // 2, y - h // 2
        x2, y2 = x + w // 2, y + h // 2
        cls = pred["class"]
        conf = pred["confidence"] * 100

        # Caja verde con etiqueta
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(frame, (x1, y1 - 25), (x2, y1), (0, 255, 0), -1)  # fondo etiqueta
        cv2.putText(frame, f"{cls} {conf:.1f}%", (x1 + 4, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # HUD superior
    cv2.putText(frame, "Reciclaje Inteligente", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
    estado = "Analizando..." if analyzing else f"{len(preds)} objeto(s) detectado(s)"
    cv2.putText(frame, estado, (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Detección de Residuos", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
print("Sesión cerrada.")

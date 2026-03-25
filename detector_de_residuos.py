import cv2
import numpy as np
import urllib.request
import os
import threading
import queue
import tkinter as tk
from tkinter import font as tkfont
from PIL import Image, ImageTk
import time
from collections import deque, Counter

# ─────────────────────────────────────────────
#  URLs del modelo MobileNetV2 (1000 categorías)
# ─────────────────────────────────────────────
ONNX_URL    = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
CLASSES_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
MODEL_FILE   = "mobilenetv2-7.onnx"
CLASSES_FILE = "imagenet_classes.txt"

# ─────────────────────────────────────────────
#  Paleta de colores por categoría de material
# ─────────────────────────────────────────────
COLORES_MATERIAL = {
    "PLASTICO / VIDRIO":  "#00D4FF",
    "PLASTICO / CARTON":  "#00BFFF",
    "PLASTICO":           "#1E90FF",
    "CERAMICA / VIDRIO":  "#9B59B6",
    "PAPEL / CARTON":     "#F39C12",
    "PAPEL":              "#E67E22",
    "METAL":              "#BDC3C7",
    "METAL / PLASTICO":   "#95A5A6",
    "ORGANICO":           "#2ECC71",
    "ELECTRONICO":        "#E74C3C",
}

COLOR_ESPERA  = "#556677"
COLOR_FONDO   = "#0A0E14"
COLOR_PANEL   = "#0D1117"
COLOR_BORDE   = "#1C2333"
COLOR_TITULO  = "#58A6FF"
COLOR_TEXTO   = "#C9D1D9"
COLOR_SUBTEXTO= "#8B949E"

# ─────────────────────────────────────────────
#  Mapeo keyword → categoría de reciclaje
# ─────────────────────────────────────────────
MAPEO_RECICLAJE = {
    "bottle":            "PLASTICO / VIDRIO",
    "cup":               "PLASTICO / CARTON",
    "mug":               "CERAMICA / VIDRIO",
    "pitcher":           "PLASTICO / VIDRIO",
    "plastic bag":       "PLASTICO",
    "carton":            "PAPEL / CARTON",
    "box":               "PAPEL / CARTON",
    "envelope":          "PAPEL / CARTON",
    "paper":             "PAPEL",
    "tissue":            "PAPEL",
    "can":               "METAL",
    "tin":               "METAL",
    "pot":               "METAL",
    "spatula":           "METAL / PLASTICO",
    "fork":              "METAL",
    "spoon":             "METAL",
    "knife":             "METAL",
    "banana":            "ORGANICO",
    "apple":             "ORGANICO",
    "orange":            "ORGANICO",
    "lemon":             "ORGANICO",
    "strawberry":        "ORGANICO",
    "pineapple":         "ORGANICO",
    "fruit":             "ORGANICO",
    "pizza":             "ORGANICO",
    "burger":            "ORGANICO",
    "meat":              "ORGANICO",
    "vegetable":         "ORGANICO",
    "cellular telephone":"ELECTRONICO",
    "computer":          "ELECTRONICO",
    "laptop":            "ELECTRONICO",
    "mouse":             "ELECTRONICO",
    "keyboard":          "ELECTRONICO",
    "remote":            "ELECTRONICO",
    "television":        "ELECTRONICO",
}

ICONOS_MATERIAL = {
    "PLASTICO / VIDRIO":  "♻  Reciclable",
    "PLASTICO / CARTON":  "♻  Reciclable",
    "PLASTICO":           "♻  Reciclable",
    "CERAMICA / VIDRIO":  "⚠  Especial",
    "PAPEL / CARTON":     "♻  Reciclable",
    "PAPEL":              "♻  Reciclable",
    "METAL":              "♻  Reciclable",
    "METAL / PLASTICO":   "♻  Reciclable",
    "ORGANICO":           "🌿  Compostable",
    "ELECTRONICO":        "⚡  E-waste",
}

INSTRUCCION_MATERIAL = {
    "PLASTICO / VIDRIO":  "Depositar en contenedor AZUL o VERDE",
    "PLASTICO / CARTON":  "Depositar en contenedor AZUL",
    "PLASTICO":           "Depositar en contenedor AZUL",
    "CERAMICA / VIDRIO":  "Centro de acopio especial",
    "PAPEL / CARTON":     "Depositar en contenedor AZUL",
    "PAPEL":              "Depositar en contenedor AZUL",
    "METAL":              "Depositar en contenedor GRIS",
    "METAL / PLASTICO":   "Depositar en contenedor GRIS",
    "ORGANICO":           "Depositar en contenedor VERDE",
    "ELECTRONICO":        "Centro de acopio de electrónicos",
}

def descargar_archivos(callback_log):
    callback_log("Verificando archivos del modelo...")
    if not os.path.exists(MODEL_FILE):
        callback_log("Descargando MobileNetV2 (~13MB)...")
        urllib.request.urlretrieve(ONNX_URL, MODEL_FILE)
        callback_log("Modelo descargado.")
    if not os.path.exists(CLASSES_FILE):
        callback_log("Descargando clases ImageNet...")
        urllib.request.urlretrieve(CLASSES_URL, CLASSES_FILE)
        callback_log("Clases descargadas.")
    callback_log("Sistema listo. Iniciando cámara...")

def categorizar(nombre_ingles):
    nombre_ingles = str(nombre_ingles).lower()
    for clave, material in MAPEO_RECICLAJE.items():
        if clave in nombre_ingles:
            return material
    return None


# ═══════════════════════════════════════════════
#  APLICACIÓN PRINCIPAL TKINTER
# ═══════════════════════════════════════════════
class AppDetector(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Detector de Reciclaje ─ Proyecto Análisis de Algoritmos")
        self.configure(bg=COLOR_FONDO)
        self.resizable(False, False)

        # Estado de detección compartido entre hilos
        self.material_actual   = None
        self.confianza_actual  = 0
        self.clase_raw         = ""
        self.running           = True
        self.net               = None
        self.classes           = []
        self.fps_val           = 0
        self._last_time        = time.time()
        self._frame_count      = 0

        # ── Estabilización temporal (anti-saltos) ─────────────────────────
        # Voting buffer: guarda los últimos 12 resultados de material.
        # El material mostrado es el más frecuente en esa ventana.
        self._voto_buffer      = deque(maxlen=12)
        # Confianza promedio suavizada (evita que el % salte cada frame)
        self._conf_suavizada   = 0.0
        # Histéresis: material que se está mostrando actualmente
        self._material_estable = None
        # Cooldown: timestamp de la última vez que se aceptó un cambio de material
        self._ultimo_cambio    = 0.0
        # Umbral de confianza para ENTRAR en un material (más exigente)
        self._UMBRAL_ENTRADA   = 30
        # Umbral para MANTENERSE en el material actual (más permisivo)
        self._UMBRAL_MANTENER  = 18
        # Segundos mínimos antes de aceptar cambio a un material DISTINTO
        self._COOLDOWN_SEG     = 1.2

        # ── Anti-parpadeo ──────────────────────────────────────────────
        # Cola de tamaño 1: el hilo de cámara deposita frames procesados
        # (como numpy arrays RGB), el hilo principal los consume a 30 fps.
        # Tamaño 1 = siempre el frame más reciente, sin acumulación.
        self._frame_queue = queue.Queue(maxsize=1)
        # Doble buffer: guardamos referencia al PhotoImage ANTERIOR para
        # que Tkinter no lo destruya mientras lo está dibujando.
        self._photo_buf = [None, None]
        self._photo_idx = 0

        self._construir_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_cerrar)

        # Lanzar hilo de carga + cámara
        threading.Thread(target=self._inicializar_modelo, daemon=True).start()

        # Iniciar el loop de refresco de UI a 30 fps (hilo principal)
        self._poll_frame()

    # ────────────────────────────────
    #  CONSTRUCCIÓN DE LA UI
    # ────────────────────────────────
    def _construir_ui(self):
        # ── Fuentes ──
        self.f_titulo   = tkfont.Font(family="Consolas", size=13, weight="bold")
        self.f_label    = tkfont.Font(family="Consolas", size=10)
        self.f_material = tkfont.Font(family="Consolas", size=22, weight="bold")
        self.f_sub      = tkfont.Font(family="Consolas", size=9)
        self.f_conf     = tkfont.Font(family="Consolas", size=11)
        self.f_log      = tkfont.Font(family="Consolas", size=8)

        # ── Layout principal: video | panel ──
        self.frame_video = tk.Frame(self, bg=COLOR_FONDO, width=800, height=600)
        self.frame_video.pack(side=tk.LEFT, padx=(12, 6), pady=12)
        self.frame_video.pack_propagate(False)

        self.canvas_video = tk.Label(
            self.frame_video, bg="#000000",
            relief="flat", bd=0
        )
        self.canvas_video.pack(expand=True, fill=tk.BOTH)

        self.panel = tk.Frame(self, bg=COLOR_PANEL, width=310)
        self.panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 12), pady=12)
        self.panel.pack_propagate(False)

        self._construir_panel()

    def _bloque(self, parent, titulo, pady_top=10):
        """Crea un bloque de sección con título y separador."""
        tk.Label(
            parent, text=titulo, bg=COLOR_PANEL,
            fg=COLOR_TITULO, font=self.f_label, anchor="w"
        ).pack(fill=tk.X, padx=14, pady=(pady_top, 2))
        tk.Frame(parent, bg=COLOR_BORDE, height=1).pack(fill=tk.X, padx=14)

    def _construir_panel(self):
        p = self.panel

        # ── Encabezado ──
        tk.Label(
            p, text="♻  RECYCLE·SCAN",
            bg=COLOR_PANEL, fg=COLOR_TITULO,
            font=self.f_titulo, anchor="w"
        ).pack(fill=tk.X, padx=14, pady=(18, 0))

        tk.Label(
            p, text="MobileNetV2 · 1000 clases ",
            bg=COLOR_PANEL, fg=COLOR_SUBTEXTO,
            font=self.f_sub, anchor="w"
        ).pack(fill=tk.X, padx=14, pady=(2, 10))

        tk.Label(
            p, text="Anyeli Rivas · Orbin Morales ",
            bg=COLOR_PANEL, fg=COLOR_SUBTEXTO,
            font=self.f_sub, anchor="w"
        ).pack(fill=tk.X, padx=14, pady=(2, 10))

        tk.Frame(p, bg=COLOR_BORDE, height=1).pack(fill=tk.X, padx=14)

        # ── Sección: DETECCIÓN ──
        self._bloque(p, "  MATERIAL DETECTADO", pady_top=14)

        self.lbl_material = tk.Label(
            p, text="---",
            bg=COLOR_PANEL, fg=COLOR_ESPERA,
            font=self.f_material, wraplength=280, justify="center"
        )
        self.lbl_material.pack(fill=tk.X, padx=14, pady=(10, 0))

        self.lbl_icono = tk.Label(
            p, text="Esperando objeto...",
            bg=COLOR_PANEL, fg=COLOR_SUBTEXTO,
            font=self.f_sub
        )
        self.lbl_icono.pack(pady=(4, 0))

        self.lbl_instruccion = tk.Label(
            p, text="",
            bg=COLOR_PANEL, fg=COLOR_TEXTO,
            font=self.f_sub, wraplength=270, justify="center"
        )
        self.lbl_instruccion.pack(fill=tk.X, padx=14, pady=(4, 0))

        # ── Barra de confianza ──
        self._bloque(p, "  CONFIANZA", pady_top=14)

        self.lbl_confianza_pct = tk.Label(
            p, text="0%",
            bg=COLOR_PANEL, fg=COLOR_TEXTO, font=self.f_conf
        )
        self.lbl_confianza_pct.pack(pady=(6, 2))

        barra_bg = tk.Frame(p, bg="#1C2333", height=12, relief="flat")
        barra_bg.pack(fill=tk.X, padx=14, pady=(0, 4))
        barra_bg.pack_propagate(False)

        self.barra_fill = tk.Frame(barra_bg, bg=COLOR_ESPERA, height=12)
        self.barra_fill.place(x=0, y=0, width=0, height=12)
        self._barra_bg = barra_bg  # guardar referencia para calcular ancho

        self.lbl_clase_raw = tk.Label(
            p, text="clase: ---",
            bg=COLOR_PANEL, fg=COLOR_SUBTEXTO, font=self.f_log
        )
        self.lbl_clase_raw.pack(padx=14, anchor="w")

        # ── Sección: CATEGORÍAS ──
        self._bloque(p, "  CATEGORÍAS RECONOCIBLES", pady_top=14)

        categorias = [
            ("PLASTICO / VIDRIO",  "●"),
            ("PAPEL / CARTON",     "●"),
            ("METAL",              "●"),
            ("ORGANICO",           "●"),
            ("ELECTRONICO",        "●"),
            ("CERAMICA / VIDRIO",  "●"),
        ]
        self.cat_labels = {}
        for mat, dot in categorias:
            color = COLORES_MATERIAL.get(mat, "#555")
            row = tk.Frame(p, bg=COLOR_PANEL)
            row.pack(fill=tk.X, padx=14, pady=1)
            lbl_dot = tk.Label(row, text=dot, bg=COLOR_PANEL, fg=color, font=self.f_sub)
            lbl_dot.pack(side=tk.LEFT)
            lbl_txt = tk.Label(row, text=f"  {mat}", bg=COLOR_PANEL, fg=COLOR_SUBTEXTO, font=self.f_sub, anchor="w")
            lbl_txt.pack(side=tk.LEFT, fill=tk.X)
            self.cat_labels[mat] = (lbl_dot, lbl_txt)

        # ── Sección: SISTEMA ──
        self._bloque(p, "  SISTEMA", pady_top=14)

        row_fps = tk.Frame(p, bg=COLOR_PANEL)
        row_fps.pack(fill=tk.X, padx=14, pady=(6, 0))
        tk.Label(row_fps, text="FPS", bg=COLOR_PANEL, fg=COLOR_SUBTEXTO, font=self.f_sub, width=12, anchor="w").pack(side=tk.LEFT)
        self.lbl_fps = tk.Label(row_fps, text="--", bg=COLOR_PANEL, fg=COLOR_TEXTO, font=self.f_sub)
        self.lbl_fps.pack(side=tk.LEFT)

        row_cam = tk.Frame(p, bg=COLOR_PANEL)
        row_cam.pack(fill=tk.X, padx=14, pady=(3, 0))
        tk.Label(row_cam, text="Cámara", bg=COLOR_PANEL, fg=COLOR_SUBTEXTO, font=self.f_sub, width=12, anchor="w").pack(side=tk.LEFT)
        self.lbl_cam = tk.Label(row_cam, text="Iniciando...", bg=COLOR_PANEL, fg=COLOR_TEXTO, font=self.f_sub)
        self.lbl_cam.pack(side=tk.LEFT)

        row_modelo = tk.Frame(p, bg=COLOR_PANEL)
        row_modelo.pack(fill=tk.X, padx=14, pady=(3, 0))
        tk.Label(row_modelo, text="Modelo", bg=COLOR_PANEL, fg=COLOR_SUBTEXTO, font=self.f_sub, width=12, anchor="w").pack(side=tk.LEFT)
        self.lbl_modelo = tk.Label(row_modelo, text="Cargando...", bg=COLOR_PANEL, fg=COLOR_TEXTO, font=self.f_sub)
        self.lbl_modelo.pack(side=tk.LEFT)

        # ── Log ──
        self._bloque(p, "  LOG", pady_top=14)

        self.log_text = tk.Text(
            p, height=6, bg="#080C10", fg=COLOR_SUBTEXTO,
            font=self.f_log, relief="flat", bd=0,
            state="disabled", wrap="word"
        )
        self.log_text.pack(fill=tk.X, padx=14, pady=(6, 0))

        # ── Botón salir ──
        tk.Button(
            p, text="⏻  SALIR",
            bg="#1C2333", fg=COLOR_TEXTO,
            font=self.f_label, relief="flat", bd=0,
            activebackground="#E74C3C", activeforeground="#FFFFFF",
            cursor="hand2", command=self._on_cerrar
        ).pack(side=tk.BOTTOM, fill=tk.X, padx=14, pady=14)

    # ────────────────────────────────
    #  LÓGICA DE MODELO + CÁMARA (hilo separado)
    # ────────────────────────────────
    def _log(self, msg):
        self.after(0, self._append_log, msg)

    def _append_log(self, msg):
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, f"› {msg}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def _inicializar_modelo(self):
        descargar_archivos(self._log)

        with open(CLASSES_FILE, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        self._log("Cargando red neuronal en memoria...")
        self.net = cv2.dnn.readNetFromONNX(MODEL_FILE)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self._log("Modelo listo. ¡Enfocá un objeto!")

        self.after(0, lambda: self.lbl_modelo.configure(text="MobileNetV2 ✓", fg="#2ECC71"))
        self._loop_camara()

    def _loop_camara(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Obtener resolución real
        real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.after(0, lambda: self.lbl_cam.configure(text=f"{real_w}×{real_h} ✓", fg="#2ECC71"))

        # Kernel de nitidez
        kernel = np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.filter2D(frame, -1, kernel)

            # ── Inferencia ──
            blob = cv2.dnn.blobFromImage(
                frame, size=(224, 224),
                mean=(103.53, 116.28, 123.675),
                swapRB=True, crop=False
            )
            blob = blob / 57.375
            self.net.setInput(blob)
            preds = self.net.forward()

            exp_p = np.exp(preds[0] - np.max(preds[0]))
            porcentajes = exp_p / np.sum(exp_p)
            idx = np.argmax(porcentajes)
            confianza = int(porcentajes[idx] * 100)
            nombre = self.classes[idx] if self.classes else ""
            material = categorizar(nombre)

            # ── Estabilización temporal ────────────────────────────────
            # 1. Alimentar el voting buffer con el material de este frame
            self._voto_buffer.append(material)  # None si no es basura

            # 2. Contar votos en la ventana deslizante
            conteo   = Counter(v for v in self._voto_buffer if v is not None)
            ganador  = conteo.most_common(1)[0][0] if conteo else None
            n_votos  = conteo[ganador] if ganador else 0
            n_frames = len(self._voto_buffer)

            # El ganador necesita al menos el 50% de los votos para ser válido
            candidato = ganador if (n_votos / max(n_frames, 1) >= 0.50) else None

            # 3. Histéresis + cooldown
            ahora_cambio = time.time()
            if candidato is None:
                # Sin candidato claro → limpiar estado estable
                self._material_estable = None
            elif candidato == self._material_estable:
                # Mismo material → mantenerlo con umbral bajo
                if confianza < self._UMBRAL_MANTENER:
                    self._material_estable = None
            else:
                # Candidato diferente → exigir umbral alto + cooldown
                tiempo_ok = (ahora_cambio - self._ultimo_cambio) >= self._COOLDOWN_SEG
                if confianza >= self._UMBRAL_ENTRADA and tiempo_ok:
                    self._material_estable = candidato
                    self._ultimo_cambio    = ahora_cambio

            # 4. Suavizar la confianza (EMA α=0.25 → responde pero no salta)
            alpha = 0.25
            conf_raw = confianza if self._material_estable else 0
            self._conf_suavizada = alpha * conf_raw + (1 - alpha) * self._conf_suavizada

            # 5. Publicar estado para el hilo de UI
            self.material_actual  = self._material_estable
            self.confianza_actual = int(self._conf_suavizada)
            self.clase_raw        = nombre

            # ── Calcular FPS ──
            self._frame_count += 1
            ahora = time.time()
            if ahora - self._last_time >= 1.0:
                self.fps_val      = self._frame_count
                self._frame_count = 0
                self._last_time   = ahora

            # ── Preparar frame para Tkinter (sin overlay OpenCV) ──
            display = cv2.resize(frame, (800, 600))

            # Dibujar borde de color según material
            color_borde = COLORES_MATERIAL.get(self.material_actual, "#556677")
            borde_bgr   = self._hex_to_bgr(color_borde)
            cv2.rectangle(display, (0, 0), (799, 599), borde_bgr, 3)

            # Etiqueta flotante sobre el video
            if self.material_actual:
                etiqueta = f" {self.material_actual} · {confianza}% "
                (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
                cv2.rectangle(display, (10, 10), (14 + tw, 14 + th + 8), borde_bgr, -1)
                cv2.putText(display, etiqueta, (12, 14 + th),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            else:
                cv2.putText(display, "Enfocando basura...", (12, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 100, 120), 2)

            # ── Depositar frame RGB en cola (hilo principal crea ImageTk) ──
            # Crear ImageTk en hilo secundario causa race conditions en Tcl/Tk
            # que se manifiestan como parpadeo. La conversión se hace en _poll_frame.
            rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            try:
                self._frame_queue.put_nowait(rgb)
            except queue.Full:
                pass  # descartamos; la UI no alcanzó a consumir el frame anterior

        cap.release()

    def _hex_to_bgr(self, hex_color):
        hex_color = hex_color.lstrip("#")
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return (b, g, r)

    def _poll_frame(self):
        """Corre en el hilo principal a ~30 fps. Consume frames de la cola
        y crea el ImageTk aquí para evitar parpadeo (Tcl/Tk no es thread-safe)."""
        try:
            rgb = self._frame_queue.get_nowait()
            img = Image.fromarray(rgb)
            # Doble buffer: alternamos entre dos slots para que el GC
            # no destruya la imagen anterior mientras Tkinter la dibuja.
            self._photo_idx = 1 - self._photo_idx
            self._photo_buf[self._photo_idx] = ImageTk.PhotoImage(image=img)
            self.canvas_video.configure(image=self._photo_buf[self._photo_idx])
        except queue.Empty:
            pass

        # Actualizar panel lateral en cada tick (no solo cuando llega frame)
        self._actualizar_panel()

        if self.running:
            # ~33 ms = 30 fps; suficiente para fluidez sin sobrecargar el event loop
            self.after(33, self._poll_frame)

    def _actualizar_panel(self):
        mat = self.material_actual
        color = COLORES_MATERIAL.get(mat, COLOR_ESPERA) if mat else COLOR_ESPERA

        # Material principal
        self.lbl_material.configure(
            text=mat if mat else "---",
            fg=color
        )

        # Icono + instrucción
        self.lbl_icono.configure(
            text=ICONOS_MATERIAL.get(mat, "Esperando objeto...") if mat else "Esperando objeto...",
            fg=color if mat else COLOR_SUBTEXTO
        )
        self.lbl_instruccion.configure(
            text=INSTRUCCION_MATERIAL.get(mat, "") if mat else ""
        )

        # Barra de confianza
        self.lbl_confianza_pct.configure(
            text=f"{self.confianza_actual}%",
            fg=color
        )
        self.barra_fill.configure(bg=color)
        ancho_barra = int(self._barra_bg.winfo_width() * self.confianza_actual / 100)
        self.barra_fill.place(x=0, y=0, width=max(ancho_barra, 0), height=12)

        # Clase raw
        self.lbl_clase_raw.configure(text=f"clase: {self.clase_raw}")

        # Resaltar categoría activa
        for mat_key, (lbl_dot, lbl_txt) in self.cat_labels.items():
            activo = mat and mat_key in mat
            lbl_dot.configure(fg=COLORES_MATERIAL.get(mat_key, "#555") if activo else "#2A3344")
            lbl_txt.configure(fg=COLOR_TEXTO if activo else COLOR_SUBTEXTO,
                              font=(self.f_sub.actual("family"), self.f_sub.actual("size"),
                                    "bold" if activo else "normal"))

        # FPS
        self.lbl_fps.configure(text=f"{self.fps_val} fps")

    # ────────────────────────────────
    #  CIERRE LIMPIO
    # ────────────────────────────────
    def _on_cerrar(self):
        self.running = False
        self.after(200, self.destroy)


if __name__ == "__main__":
    app = AppDetector()
    app.mainloop()

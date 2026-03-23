# Proyecto Final - Análisis de Algoritmos

Hola. Este es el repositorio de mi proyecto final para la materia de Análisis de Algoritmos. Básicamente hice un programa que usa la cámara de la compu para detectar qué tipo de basura le estás poniendo enfrente (plástico, cartón, metal, vidrio, orgánico, etc.) y saber cómo clasificarla correctamente.

## Qué le agregué al código
Estuve probando distintas opciones y al final estructuré el programa para que corra todo de forma 100% local (sin tener que usar internet ni APIs). 

- Detección de objetos: Usé una red neuronal con formato ONNX (MobileNet) que ya conoce miles de objetos de la vida diaria. Lo que hice en el código (detector_de_residuos.py) fue armar un diccionario que toma lo que ve la cámara y lo agrupa matemáticamente en las 6 categorías de reciclaje principales que ocupamos para la clase.
- Diseño de pantalla: En lugar de poner los clásicos cuadros de detección que brincan por todos lados, hice un layout directamente en la pantalla de la cámara con rectángulos semitransparentes para mostrar qué residuo detectó de una forma más limpia.

## Cómo probar el proyecto desde cero

Si quieres bajar mi trabajo y probarlo en tu propia computadora, aquí te explico los pasos exactos que debes seguir para que no falles:

1. **Bájate el código de mi GitHub:**
   Abre una terminal o CMD en tu computadora, entra en la carpeta donde quieras guardar el proyecto y pega este comando:
   ```bash
   git clone -b orbin-dev https://github.com/OrbnMorls2124/Proyecto-Analisis-de-Algoritmos-.git
   ```
   *(Nota: Yo usé la rama 'orbin-dev', así que el comando de arriba ya te baja la versión final derechito).*

2. **Entra a la carpeta del proyecto:**
   Muévete a la carpeta que se acaba de crear:
   ```bash
   cd "Proyecto-Analisis-de-Algoritmos-"
   ```

3. **Instala las librerías necesarias:**
   El programa necesita OpenCV para manejar la cámara y Numpy para las matemáticas de la IA. Solo tienes que pegar este comando en tu terminal para instalarlas:
   ```bash
   pip install opencv-python numpy
   ```

4. **Inicia el detector:**
   Ahora que ya tienes todo, solo escribe esto para encenderlo:
   ```bash
   python detector_de_residuos.py
   ```

5. **Tips de uso:**
   - **La primera vez:** No te asustes si tarda unos segundos extras en abrir; el programa bajará automáticamente el archivo de la inteligencia artificial (pesa como 13MB) de internet. Esto solo pasa una vez.
   - **En el lente:** Pon botellas, manzanas, libros o lo que tengas de reciclaje cerca y verás la clasificación en la parte de abajo de la ventana.
   - **Para salir:** Si ya te cansaste de probarlo, presiona la tecla **'q'** y se cerrará todo solito.

¡Y listo! Con eso ya puedes ver funcionando mi detector de residuos. 🌍

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# Cargar el modelo de detección de objetos desde TensorFlow Hub
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# Función para detectar personas
def detect_persons(frame):
    # Preprocesar la imagen
    img_tensor = tf.convert_to_tensor(frame)
    img_tensor = tf.image.convert_image_dtype(img_tensor, tf.uint8)[tf.newaxis, ...]

    # Realizar la detección
    detections = detector(img_tensor)

    # Obtener las cajas de detección, clases y puntajes
    boxes = detections['detection_boxes'].numpy()[0]
    classes = detections['detection_classes'].numpy()[0].astype(np.int32)
    scores = detections['detection_scores'].numpy()[0]

    # Filtrar las detecciones para encontrar solo personas
    person_boxes = []
    for i in range(len(scores)):
        if scores[i] > 0.5 and classes[i] == 1:  # El 1 indica la clase "persona"
            person_boxes.append(boxes[i])

    return person_boxes

# Función para obtener el centro de una caja delimitadora
def get_person_center(box, frame_shape):
    h, w, _ = frame_shape
    ymin, xmin, ymax, xmax = box
    x_center = int((xmin + xmax) * w / 2)
    y_center = int((ymin + ymax) * h / 2)
    return (x_center, y_center)

# Captura de video
cap = cv2.VideoCapture(2 + cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar el frame.")
        break

    # Detección de personas en el frame
    person_boxes = detect_persons(frame)

    # Dibujar las cajas delimitadoras y mostrar el centro de la persona detectada
    for box in person_boxes:
        person_center = get_person_center(box, frame.shape)
        
        # Dibujar la caja alrededor de la persona detectada
        ymin, xmin, ymax, xmax = box
        h, w, _ = frame.shape
        cv2.rectangle(frame, (int(xmin * w), int(ymin * h)), (int(xmax * w), int(ymax * h)), (0, 255, 0), 2)
        cv2.putText(frame, "Persona", (int(xmin * w), int(ymin * h) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Dibujar el centro de la persona detectada
        cv2.circle(frame, person_center, 5, (0, 0, 255), -1)

    # Mostrar el frame con las detecciones
    cv2.imshow('input', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Presionar 'ESC' para salir
        break

cap.release()
cv2.destroyAllWindows()

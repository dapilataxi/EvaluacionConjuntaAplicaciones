import cv2
import cv2.aruco as aruco
import numpy as np
import pickle
import tensorflow as tf
import tensorflow_hub as hub
import pygame
import threading

# Inicializar pygame para el manejo de sonido
pygame.mixer.init()

# Cargar el archivo de sonido
sound = pygame.mixer.Sound('audio.mp3')  # Reemplaza 'alarma.wav' con el archivo de sonido que desees

# Variable para controlar si la alarma está sonando
alarm_sounding = False

# Función para reproducir la alarma en un hilo separado
def sound_alarm():
    global alarm_sounding
    sound.play(-1)  # Reproduce en bucle
    while alarm_sounding:
        pygame.time.wait(100)  # Espera brevemente para evitar alta carga en el CPU
    sound.stop()  # Detiene el sonido cuando `alarm_sounding` es False

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

# Función para ordenar los puntos de los marcadores ArUco
def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

# Cargar la matriz de la cámara y los coeficientes de distorsión desde los archivos .pkl
with open("cameraMatrix.pkl", "rb") as f:
    camera_matrix = pickle.load(f)

with open("dist.pkl", "rb") as f:
    dist_coeffs = pickle.load(f)

# Diccionario ArUco y parámetros de detección
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
parameters = aruco.DetectorParameters()

# IDs de los marcadores a detectar
marker_ids = [7, 8, 9, 10]

# Variable para almacenar los centros de los marcadores entre fotogramas
marker_centers = None

# Captura de video
cap = cv2.VideoCapture(2 + cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar el frame.")
        break

    # Deshacer la distorsión de la imagen
    h, w = frame.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=parameters)

    if ids is not None:
        ids_list = ids.flatten()
        marker_centers = []

        for i in range(len(ids)):
            if ids[i] in marker_ids:
                center = np.mean(corners[i][0], axis=0)
                marker_centers.append(center)

        if len(marker_centers) == 4:
            marker_centers = np.array(marker_centers, dtype=np.float32)
            marker_centers = order_points(marker_centers)

    # Si los centros de los marcadores se detectaron alguna vez, dibujar la zona restringida
    if marker_centers is not None:
        # Convertir marker_centers a un array de NumPy si no lo es
        marker_centers_np = np.array(marker_centers, dtype=np.float32)
        cv2.polylines(frame, [marker_centers_np.astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)

    # Detección de personas en el frame
    person_boxes = detect_persons(frame)
    person_in_area = False

    for box in person_boxes:
        person_center = get_person_center(box, frame.shape)

        # Dibujar la caja alrededor de la persona detectada
        ymin, xmin, ymax, xmax = box
        h, w, _ = frame.shape
        cv2.rectangle(frame, (int(xmin * w), int(ymin * h)), (int(xmax * w), int(ymax * h)), (0, 255, 0), 2)
        cv2.putText(frame, "Persona", (int(xmin * w), int(ymin * h) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Dibujar el centro de la persona detectada
        cv2.circle(frame, person_center, 5, (0, 0, 255), -1)

        # Verificar si la persona está dentro de la zona restringida
        if marker_centers is not None and cv2.pointPolygonTest(marker_centers_np, person_center, False) >= 0:
            cv2.putText(frame, "ALERTA: Persona en area prohibida", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            person_in_area = True

    # Controlar el sonido de la alarma
    if person_in_area and not alarm_sounding:
        alarm_sounding = True
        threading.Thread(target=sound_alarm).start()
    elif not person_in_area and alarm_sounding:
        alarm_sounding = False

    # Mostrar el frame con las detecciones
    cv2.imshow('input', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Presionar 'ESC' para salir
        break

cap.release()
cv2.destroyAllWindows()
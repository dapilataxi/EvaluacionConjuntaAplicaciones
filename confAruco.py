import cv2
import cv2.aruco as aruco
import numpy as np
import pickle

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

# Captura de video
cap = cv2.VideoCapture(2 + cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar el frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=parameters)

    if ids is not None:
        ids_list = ids.flatten()
        marker_centers = []

        for i in range(len(ids)):
            if ids[i] in marker_ids:
                # Eliminar o comentar la siguiente línea para no dibujar los ejes
                # rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[i], 100, camera_matrix, dist_coeffs)
                # cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec[0], tvec[0], 50)

                center = np.mean(corners[i][0], axis=0)
                marker_centers.append(center)

        if len(marker_centers) == 4:
            marker_centers = np.array(marker_centers, dtype=np.float32)
            marker_centers = order_points(marker_centers)

            cv2.polylines(frame, [marker_centers.astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=2)

    cv2.imshow('input', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Presionar 'ESC' para salir
        break

cap.release()
cv2.destroyAllWindows()

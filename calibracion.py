import numpy as np
import cv2 as cv
import glob
import os
import pickle

# Parámetros del tablero de ajedrez
chessboardSize = (8, 6)
frameSize = (1280, 720)

# Criterios de terminación
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Preparar puntos de objeto en 3D, como (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

# Escalar las coordenadas según el tamaño de los cuadrados del tablero (en milímetros)
size_of_chessboard_squares_mm = 20
objp = objp * size_of_chessboard_squares_mm

# Arrays para almacenar puntos de objeto y puntos de imagen desde todas las imágenes
objpoints = []  # Puntos 3D en el espacio real
imgpoints = []  # Puntos 2D en el plano de la imagen

# Cargar las imágenes de la carpeta especificada
images = glob.glob('FotosCalib2/*.jpg')

# Procesar cada imagen
for image in images:

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Encontrar las esquinas del tablero de ajedrez
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # Si se encuentran, agregar puntos de objeto y puntos de imagen (después de refinarlos)
    if ret:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Dibujar y mostrar las esquinas
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(100)

cv.destroyAllWindows()

# Si se encontraron puntos suficientes, proceder a la calibración
if len(objpoints) > 0 and len(imgpoints) > 0:
    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

    if ret:
        print("Calibración exitosa.")
        print("Matriz de la cámara:")
        print(cameraMatrix)
        print("Coeficientes de distorsión:")
        print(dist)

        # Verificación de los resultados de la calibración en una imagen
        img = cv.imread(images[0])
        h, w = img.shape[:2]
        newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

        # Corregir la distorsión en la primera imagen
        dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        # Mostrar la imagen corregida
        cv.imshow('Imagen corregida', dst)
        cv.waitKey(0)
        cv.destroyAllWindows()

        # Guardar la imagen corregida
        cv.imwrite('calibrated_image.jpg', dst)

        # Calcular el error de reproyección
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
            error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
            total_error += error
        print("Error de reproyección total: ", total_error / len(objpoints))

    else:
        print("Calibración fallida.")
else:
    print("No se encontraron suficientes puntos para la calibración. Verifica las imágenes y el tamaño del tablero.")


#########CALIBRACION############################################
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

# Guardar el resultado de la calibración de la cámara para su uso posterior (no nos preocuparemos por rvecs / tvecs)
pickle.dump((cameraMatrix, dist), open("calibration.pkl", "wb"), protocol=2)
pickle.dump(cameraMatrix, open("cameraMatrix.pkl", "wb"), protocol=2)
pickle.dump(dist, open("dist.pkl", "wb"), protocol=2)


##########DESISTORSION############################################

img = cv.imread('FotosCalib2/foto0.jpg')
h, w = img.shape[:2]
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(cameraMatrix, dist, (w, h), 1, (w, h))

# Deshacer la distorsión
dst = cv.undistort(img, cameraMatrix, dist, None, newCameraMatrix)

# recortar la imagen
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult1.jpg', dst)

# Deshacer con Remapeo
mapx, mapy = cv.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (w, h), 5)
dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

# recortar la imagen
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('caliResult2.jpg', dst)

# Error de reproyección
mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print("error total: {}".format(mean_error / len(objpoints)))
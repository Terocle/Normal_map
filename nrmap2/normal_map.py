import numpy as np
from PIL import Image
import cv2


def generate_normal_map(image: Image.Image, strength=5.0, depth=1.0, operator="Sobel", invert=False):
    gray = np.array(image.convert("L"), dtype=np.float32)

    # Выбор оператора
    if operator == "Sobel":
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    elif operator == "Scharr":
        dx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        dy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    elif operator == "Prewitt":
        kernelx = np.array([[1, 0, -1],
                            [1, 0, -1],
                            [1, 0, -1]], dtype=np.float32)
        kernely = np.array([[1, 1, 1],
                            [0, 0, 0],
                            [-1, -1, -1]], dtype=np.float32)
        dx = cv2.filter2D(gray, -1, kernelx)
        dy = cv2.filter2D(gray, -1, kernely)
        dy = -dy
        dx = -dx
    else:
        raise ValueError(f"Unknown operator: {operator}")

    dx *= strength / 255.0
    dy *= strength / 255.0
    dz = np.ones_like(dx) * depth

    # Вектор нормали
    norm = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    nx = dx / norm
    ny = dy / norm
    nz = dz / norm

    # Инвертирование
    if invert:
        nx = -nx
        ny = -ny
        nz = -nz

    normal_map = np.stack([(nx + 1) / 2, (ny + 1) / 2, (nz + 1) / 2], axis=-1)
    return normal_map.clip(0, 1)

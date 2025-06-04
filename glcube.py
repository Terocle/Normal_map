from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtGui import QImage
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image
import numpy as np
import math


class GLCubeWidget(QOpenGLWidget):
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.image = image  # одна и та же текстура для всех граней
        self.texture_id = None
        self.last_pos = None
        self.x_rot = 20
        self.y_rot = 30
        self.zoom = -6.0

    def initializeGL(self):
        try:
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)
            glEnable(GL_COLOR_MATERIAL)
            glClearColor(0.1, 0.1, 0.1, 1.0)

            light_position = [0.0, 0.0, 2.0, 1.0]
            glLightfv(GL_LIGHT0, GL_POSITION, light_position)

            if self.image:
                self.texture_id = self.bind_texture(self.image)
            else:
                print("No image provided to GLCubeWidget.")

        except Exception as e:
            print("Error in initializeGL:", e)

    def bind_texture(self, pil_image):
        try:
            image = pil_image.convert("RGB").transpose(Image.FLIP_TOP_BOTTOM)
            image_data = image.tobytes("raw", "RGB")
            width, height = image.size

            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0,
                         GL_RGB, GL_UNSIGNED_BYTE, image_data)

            return texture_id
        except Exception as e:
            print("Error in bind_texture:", e)
            return 0

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, w / h if h != 0 else 1.0, 1.0, 100.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, self.zoom)
        glRotatef(self.x_rot, 1.0, 0.0, 0.0)
        glRotatef(self.y_rot, 0.0, 1.0, 0.0)

        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)

        self.draw_cube()

        glDisable(GL_TEXTURE_2D)

    def draw_cube(self):
        size = 1.0
        faces = [
            [0, 0, 1], [0, 0, -1], [-1, 0, 0],
            [1, 0, 0], [0, 1, 0], [0, -1, 0]
        ]

        vertices = [
            # Front face
            ((-size, -size, size), (0, 0)),
            ((size, -size, size), (1, 0)),
            ((size, size, size), (1, 1)),
            ((-size, size, size), (0, 1)),
            # Back face
            ((-size, -size, -size), (1, 0)),
            ((-size, size, -size), (1, 1)),
            ((size, size, -size), (0, 1)),
            ((size, -size, -size), (0, 0)),
            # Left face
            ((-size, -size, -size), (0, 0)),
            ((-size, -size, size), (1, 0)),
            ((-size, size, size), (1, 1)),
            ((-size, size, -size), (0, 1)),
            # Right face
            ((size, -size, -size), (1, 0)),
            ((size, size, -size), (1, 1)),
            ((size, size, size), (0, 1)),
            ((size, -size, size), (0, 0)),
            # Top face
            ((-size, size, -size), (0, 1)),
            ((-size, size, size), (0, 0)),
            ((size, size, size), (1, 0)),
            ((size, size, -size), (1, 1)),
            # Bottom face
            ((-size, -size, -size), (1, 1)),
            ((size, -size, -size), (0, 1)),
            ((size, -size, size), (0, 0)),
            ((-size, -size, size), (1, 0)),
        ]

        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glBegin(GL_QUADS)
        for i in range(0, len(vertices), 4):
            normal = faces[i // 4]
            for j in range(4):
                pos, tex = vertices[i + j]
                glTexCoord2f(*tex)
                glNormal3fv(normal)
                glVertex3f(*pos)
        glEnd()

    def normalize(self, v):
        length = math.sqrt(sum(x ** 2 for x in v))
        return tuple(x / length for x in v)

    def mousePressEvent(self, event):
        self.last_pos = event.pos()

    def mouseMoveEvent(self, event):
        if self.last_pos is not None:
            dx = event.x() - self.last_pos.x()
            dy = event.y() - self.last_pos.y()
            self.x_rot += dy
            self.y_rot += dx
            self.update()
            self.last_pos = event.pos()

    def wheelEvent(self, event):
        delta = event.angleDelta().y() / 120  # шаг: ±1 за щелчок
        self.zoom += delta * 0.5  # чувствительность приближения
        self.update()


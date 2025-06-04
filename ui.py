import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QFileDialog, QVBoxLayout, QWidget, QHBoxLayout, QSlider,
    QComboBox, QCheckBox, QGroupBox, QGridLayout, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon
from normal_map import generate_normal_map
import numpy as np
from PIL import Image
from glcube import GLCubeWidget



class NormalMapApp:
    def __init__(self):
        # ...
        self.app = QApplication(sys.argv)
        self.window = QMainWindow()
        self.window.setWindowIcon(QIcon("icon.ico"))
        self.window.setWindowTitle("Normal Map Generator")
        self.window.setFixedSize(1480, 700)

        self.image = None
        self.normal_map = None
        self.strength = 5
        self.depth = 1.0
        self.operator = "Sobel"
        self.invert = False
        self.intensity = 1.0
        self.ambient = 0.2
        self.radius = 300
        self.light_pos = (200, 200)

        self.original_label = QLabel()
        self.original_label.setFixedSize(320, 320)
        self.original_label.setObjectName("imageLabel")
        self.original_label.setStyleSheet("border: 1px solid gray; background-color: #eee;")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setText("Original Image")

        self.normal_label = QLabel()
        self.normal_label.setFixedSize(320, 320)
        self.normal_label.setObjectName("imageLabel")
        self.normal_label.setStyleSheet("border: 1px solid gray; background-color: #eee;")
        self.normal_label.setAlignment(Qt.AlignCenter)
        self.normal_label.setText("Normal Map")

        self.visualization_label = QLabel()
        self.visualization_label.setFixedSize(600, 600)
        self.visualization_label.setStyleSheet("border: 1px solid gray; background-color: black;")
        self.visualization_label.setMouseTracking(True)
        self.visualization_label.mouseMoveEvent = self.update_light_position

        self.visualization_gl_widget = None

        self.load_button = QPushButton("Загрузить")
        self.save_button = QPushButton("Сохранить")

        self.strength_slider = QSlider(Qt.Horizontal)
        self.strength_slider.setMinimum(1)
        self.strength_slider.setMaximum(10)
        self.strength_slider.setValue(5)
        self.strength_slider.valueChanged.connect(self.update_strength)

        self.depth_slider = QSlider(Qt.Horizontal)
        self.depth_slider.setMinimum(1)
        self.depth_slider.setMaximum(10)
        self.depth_slider.setValue(1)
        self.depth_slider.valueChanged.connect(self.update_depth)

        self.operator_combo = QComboBox()
        self.operator_combo.addItems(["Sobel", "Scharr", "Prewitt"])
        self.operator_combo.currentTextChanged.connect(self.update_operator)

        self.invert_checkbox = QCheckBox("Инвертировать")
        self.invert_checkbox.stateChanged.connect(self.update_invert)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["2D", "3D"])
        self.mode_combo.currentTextChanged.connect(self.update_mode)

        self.intensity_slider = QSlider(Qt.Horizontal)
        self.intensity_slider.setMinimum(1)
        self.intensity_slider.setMaximum(20)
        self.intensity_slider.setValue(10)
        self.intensity_slider.valueChanged.connect(self.update_light_params)

        self.ambient_slider = QSlider(Qt.Horizontal)
        self.ambient_slider.setMinimum(0)
        self.ambient_slider.setMaximum(10)
        self.ambient_slider.setValue(2)
        self.ambient_slider.valueChanged.connect(self.update_light_params)

        self.radius_slider = QSlider(Qt.Horizontal)
        self.radius_slider.setMinimum(50)
        self.radius_slider.setMaximum(1000)
        self.radius_slider.setValue(300)
        self.radius_slider.valueChanged.connect(self.update_light_params)

        self.load_button.clicked.connect(self.load_image)
        self.save_button.clicked.connect(self.save_normal_map)

        left_images_layout = QVBoxLayout()
        left_images_layout.addWidget(self.original_label, alignment=Qt.AlignHCenter)
        left_images_layout.addSpacing(10)
        self.load_button.setFixedWidth(320)
        left_images_layout.addWidget(self.load_button, alignment=Qt.AlignHCenter)

        center_images_layout = QVBoxLayout()
        center_images_layout.addWidget(self.normal_label, alignment=Qt.AlignHCenter)
        center_images_layout.addSpacing(10)
        self.save_button.setFixedWidth(320)
        center_images_layout.addWidget(self.save_button, alignment=Qt.AlignHCenter)

        images_layout = QHBoxLayout()
        images_layout.addLayout(left_images_layout)
        images_layout.addLayout(center_images_layout)

        params_box = QGroupBox("Параметры карты нормалей")
        params_layout = QGridLayout()
        params_layout.addWidget(QLabel("Сила нормалей:"), 0, 0)
        params_layout.addWidget(self.strength_slider, 0, 1)
        params_layout.addWidget(QLabel("Гладкость:"), 1, 0)
        params_layout.addWidget(self.depth_slider, 1, 1)
        params_layout.addWidget(QLabel("Оператор:"), 2, 0)
        params_layout.addWidget(self.operator_combo, 2, 1)
        params_layout.addWidget(self.invert_checkbox, 3, 0, 1, 2)
        params_layout.addWidget(QLabel("Режим:"), 4, 0)
        params_layout.addWidget(self.mode_combo, 4, 1)
        params_box.setLayout(params_layout)

        light_box = QGroupBox("Параметры света")
        light_layout = QGridLayout()
        light_layout.addWidget(QLabel("Интенсивность:"), 0, 0)
        light_layout.addWidget(self.intensity_slider, 0, 1)
        light_layout.addWidget(QLabel("Окружение:"), 1, 0)
        light_layout.addWidget(self.ambient_slider, 1, 1)
        light_layout.addWidget(QLabel("Радиус:"), 2, 0)
        light_layout.addWidget(self.radius_slider, 2, 1)
        light_box.setLayout(light_layout)

        left_panel = QVBoxLayout()
        left_panel.addLayout(images_layout)
        left_panel.addWidget(params_box)
        left_panel.addWidget(light_box)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_panel)
        self.visualization_container = QVBoxLayout()
        self.visualization_container.addWidget(self.visualization_label)
        main_layout.addLayout(self.visualization_container)

        container = QWidget()
        container.setLayout(main_layout)
        self.window.setCentralWidget(container)

        dark_style = """
            QLabel {
                color: #dddddd;
                font-size: 12pt;
            }
            QLabel#imageLabel {
                font-weight: bold;
                color: #888888;
                font-size: 14pt;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #dddddd;
                font-family: Segoe UI, sans-serif;
                font-size: 12pt;
            }
            QGroupBox {
                border: 1px solid #5c5c5c;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
            QPushButton {
                background-color: #3c3f41;
                color: #ffffff;
                border: 1px solid #5c5c5c;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #505357;
            }
            QPushButton:pressed {
                background-color: #2c2f31;
            }
            QSlider::groove:horizontal {
                height: 6px;
                background: #3c3f41;
            }
            QSlider::handle:horizontal {
                background: #dddddd;
                border: 1px solid #5c5c5c;
                width: 12px;
                margin: -5px 0;
                border-radius: 3px;
            }
            QComboBox, QCheckBox, QLabel {
                background-color: transparent;
            }
            QComboBox {
                background-color: #3c3f41;
                color: #ffffff;
                border: 1px solid #5c5c5c;
                padding: 4px;
            }
            QComboBox::drop-down {
                border-left: 1px solid #5c5c5c;
                background: #3c3f41;
            }
            QComboBox QAbstractItemView {
                background-color: #2b2b2b;
                color: #ffffff;
                selection-background-color: #505357;
            }
        """
        self.app.setStyleSheet(dark_style)

    def update_light_position(self, event):
        self.light_pos = (event.x(), event.y())
        self.render_light_effect()

    def render_light_effect(self):
        if self.image is None or self.normal_map is None:
            return

        label_size = self.visualization_label.size()
        width = label_size.width()
        height = label_size.height()

        iw, ih = self.image.size
        scale = min(width / iw, height / ih)
        new_w, new_h = int(iw * scale), int(ih * scale)
        image_resized = self.image.resize((new_w, new_h), Image.LANCZOS)
        image_padded = Image.new("RGB", (width, height), (0, 0, 0))
        offset = ((width - new_w) // 2, (height - new_h) // 2)
        image_padded.paste(image_resized, offset)
        resized_image = np.array(image_padded) / 255.0
        nh, nw = self.normal_map.shape[:2]
        scale = min(width / nw, height / nh)
        new_w, new_h = int(nw * scale), int(nh * scale)
        normals_resized = Image.fromarray((self.normal_map * 255).astype(np.uint8)).resize((new_w, new_h),
                                                                                           Image.LANCZOS)
        normals_padded = Image.new("RGB", (width, height), (127, 127, 255))
        offset = ((width - new_w) // 2, (height - new_h) // 2)
        normals_padded.paste(normals_resized, offset)
        resized_normals = np.array(normals_padded) / 255.0
        normals = resized_normals * 2 - 1

        lx, ly = self.light_pos
        x = np.arange(width)
        y = np.arange(height)
        xx, yy = np.meshgrid(x, y)
        light_dir = np.stack([(lx - xx), (ly - yy), np.full_like(xx, 100)], axis=-1)
        light_dist = np.linalg.norm(light_dir, axis=-1, keepdims=True)
        light_dir = light_dir / (light_dist + 1e-6)

        dot = np.sum(normals * light_dir, axis=-1, keepdims=True)
        dot = np.clip(dot, 0, 1)

        attenuation = np.clip(1.0 - (light_dist[..., 0] / self.radius), 0, 1)[..., None]
        brightness = self.ambient + self.intensity * dot * attenuation
        brightness = np.clip(brightness, 0, 1)

        shaded = (resized_image * brightness) * 255
        shaded = shaded.astype(np.uint8)

        qimage = QImage(shaded, width, height, width * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.visualization_label.setPixmap(pixmap)

    def update_strength(self, value):
        self.strength = value
        self.process_image()
        if self.mode_combo.currentText() == "3D":
            self.update_mode("3D")
        else:
            self.render_light_effect()

    def update_depth(self, value):
        self.depth = value / 10.0
        self.process_image()
        if self.mode_combo.currentText() == "3D":
            self.update_mode("3D")
        else:
            self.render_light_effect()

    def update_operator(self, text):
        self.operator = text
        self.process_image()
        if self.mode_combo.currentText() == "3D":
            self.update_mode("3D")
        else:
            self.render_light_effect()

    def update_invert(self, state):
        self.invert = state == Qt.Checked
        self.process_image()
        if self.mode_combo.currentText() == "3D":
            self.update_mode("3D")
        else:
            self.render_light_effect()

    def update_light_params(self):
        self.intensity = self.intensity_slider.value() / 10.0
        self.ambient = self.ambient_slider.value() / 10.0
        self.radius = self.radius_slider.value()

        if self.mode_combo.currentText() == "3D":
            self.update_mode("3D")
        else:
            self.render_light_effect()

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self.window, "Open Image")
        if path:
            image = Image.open(path).convert("RGB")
            width, height = image.size
            if width < 64 or height < 64 or width > 2048 or height > 2048:
                QMessageBox.warning(self.window, "Invalid Image Size",
                                    "Image must be between 64x64 and 2048x2048 pixels.")
                return
            self.image = image  # <--- ВАЖНО: сохраняем PIL.Image
            pixmap = QPixmap(path).scaled(320, 320, Qt.KeepAspectRatio)
            self.original_label.setPixmap(pixmap)
            self.process_image()
            if self.mode_combo.currentText() == "2D":
                self.render_light_effect()
            else:
                self.update_mode("3D")

    def process_image(self):
        if self.image is None:
            QMessageBox.warning(self.window, "Нет изображения", "Сначала загрузите изображение.")
            return
        self.normal_map = generate_normal_map(
            self.image,
            strength=self.strength,
            depth=self.depth,
            operator=self.operator,
            invert=self.invert
        )
        normal_image = Image.fromarray((self.normal_map * 255).astype(np.uint8))
        qimage = QImage(
            normal_image.tobytes(), normal_image.width, normal_image.height,
            normal_image.width * 3, QImage.Format_RGB888
        )
        pixmap = QPixmap.fromImage(qimage).scaled(320, 320, Qt.KeepAspectRatio)
        self.normal_label.setPixmap(pixmap)

    def save_normal_map(self):
        if self.normal_map is not None:
            path, _ = QFileDialog.getSaveFileName(self.window, "Save Normal Map", filter="PNG Files (*.png)")
            if path:
                normal_image = Image.fromarray((self.normal_map * 255).astype(np.uint8))
                normal_image.save(path)

    def visualize(self):
        if self.normal_map is not None:
            self.render_light_effect()
        else:
            QMessageBox.warning(self.window, "Нет изображения", "Сначала загрузите изображение.")

    def get_shaded_image_for_3d(self) -> Image.Image:
        if self.image is None or self.normal_map is None:
            return None

        width, height = 512, 512

        # Пропорционально ресайзим исходное изображение
        iw, ih = self.image.size
        side = min(iw, ih)
        left = (iw - side) // 2
        top = (ih - side) // 2
        image_cropped = self.image.crop((left, top, left + side, top + side))
        image_resized = image_cropped.resize((width, height), Image.LANCZOS)
        resized_image = np.array(image_resized) / 255.0

        # Обрезаем нормаль аналогично изображению
        normal_image = Image.fromarray((self.normal_map * 255).astype(np.uint8))
        normal_cropped = normal_image.crop((left, top, left + side, top + side))
        normal_resized = normal_cropped.resize((width, height), Image.LANCZOS)

        normals = np.array(normal_resized) / 255.0
        normals = normals * 2 - 1  # [-1,1]

        # Центр света, слабая интенсивность
        lx, ly = width // 2, height // 2
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        light_dir = np.stack([(lx - xx), (ly - yy), np.full_like(xx, 100)], axis=-1)
        light_dist = np.linalg.norm(light_dir, axis=-1, keepdims=True)
        light_dir = light_dir / (light_dist + 1e-6)

        dot = np.sum(normals * light_dir, axis=-1, keepdims=True)
        dot = np.clip(dot, 0, 1)

        attenuation = np.clip(1.0 - (light_dist[..., 0] / self.radius), 0, 1)[..., None]
        brightness = self.ambient + self.intensity * dot * attenuation

        brightness = np.clip(brightness, 0, 1)

        shaded = (resized_image * brightness) * 255
        shaded = shaded.astype(np.uint8)
        return Image.fromarray(shaded)

    def update_mode(self, text):
        if text == "2D":
            # Удаляем OpenGL-виджет, если он есть
            if self.visualization_gl_widget:
                self.visualization_container.removeWidget(self.visualization_gl_widget)
                self.visualization_gl_widget.hide()
                self.visualization_gl_widget.deleteLater()
                self.visualization_gl_widget = None
            self.visualization_label.show()
            self.render_light_effect()



        elif text == "3D":

            if self.image is None or self.normal_map is None:
                QMessageBox.warning(self.window, "Ошибка",
                                    "Сначала загрузите изображение и сгенерируйте карту нормалей.")
                self.mode_combo.setCurrentText("2D")
                return

            # Удаляем старый OpenGL-виджет, если он есть
            if self.visualization_gl_widget:
                self.visualization_container.removeWidget(self.visualization_gl_widget)
                self.visualization_gl_widget.hide()
                self.visualization_gl_widget.deleteLater()

            # --- Временно пропускаем render_light_effect() ---
            # Просто возьмём изображение, уменьшим до 512x512
            base_image = self.get_shaded_image_for_3d()
            if base_image is None:
                QMessageBox.critical(self.window, "Ошибка", "Не удалось создать текстуру для 3D.")
                return

            x_rot, y_rot, zoom = 20, 30, -6.0
            if self.visualization_gl_widget:
                x_rot = self.visualization_gl_widget.x_rot
                y_rot = self.visualization_gl_widget.y_rot
                zoom = self.visualization_gl_widget.zoom

            # --- Создаём виджет куба ---
            from glcube import GLCubeWidget
            self.visualization_gl_widget = GLCubeWidget(base_image)
            self.visualization_gl_widget.x_rot = x_rot
            self.visualization_gl_widget.y_rot = y_rot
            self.visualization_gl_widget.zoom = zoom
            self.visualization_gl_widget.setFixedSize(600, 600)
            self.visualization_container.addWidget(self.visualization_gl_widget)
            self.visualization_label.hide()
            self.visualization_gl_widget.show()

    def run(self):
        self.window.show()
        self.app.exec_()

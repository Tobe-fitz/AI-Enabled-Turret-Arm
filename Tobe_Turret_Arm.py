import sys
import cv2
import serial
import time
from ultralytics import YOLO
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QSlider, QCheckBox, QHBoxLayout, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, QPoint
from PyQt5.QtGui import QImage, QPixmap, QMouseEvent

# ---------------------- SERIAL CONFIG ----------------------
SERIAL_PORT = 'COM3'
BAUD_RATE = 9600

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Wait for Arduino reset
    print("Serial connection established.")
except Exception as e:
    print(f"Serial connection failed: {e}")
    ser = None  # Allows GUI to still run without Arduino

# ---------------------- YOLO MODEL ----------------------
model = YOLO("yolov8n.pt")

# ---------------------- SERVO SETTINGS ----------------------
SERVO_X_MIN, SERVO_X_MAX = 0, 180
SERVO_Y_MIN, SERVO_Y_MAX = 0, 180
servo_x_angle = 90  # Pan center
servo_y_angle = 90  # Tilt center

PAN_OFFSET = 0
TILT_OFFSET = 0

# ---------------------- CAMERA ----------------------
# Try camera 1 first (as in your code), then fall back to 0 if needed.
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap.release()
    cap = cv2.VideoCapture(0)
FRAME_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
FRAME_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

# ---------------------- SEND ANGLES TO ARDUINO ----------------------
def send_servo_angles(x_angle, y_angle):
    if ser and ser.is_open:
        x_byte = max(0, min(180, int(x_angle + PAN_OFFSET)))
        y_byte = max(0, min(180, int(y_angle + TILT_OFFSET)))
        ser.write(bytes([255, x_byte, y_byte]))

def send_fire_command():
    if ser and ser.is_open:
        ser.write(bytes([250]))  # 🔥 Special byte for FIRE
    print("🔥 FIRE button pressed!")

# ---------------------- INTERACTIVE VIDEO LABEL ----------------------
class InteractiveVideoLabel(QLabel):
    """
    Captures mouse events on the video area and maintains a user-selected
    target rectangle in FRAME (camera) coordinates. Supports:
      - Drag to create a box
      - Drag inside to move
      - Drag corners/edges to resize
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setScaledContents(True)  # Enables simple scale mapping
        self.selection_enabled = False
        self.frame_size = (FRAME_WIDTH, FRAME_HEIGHT)
        self.rect = None  # (x1,y1,x2,y2) in frame coords
        self.mode = None  # 'creating' | 'moving' | 'resizing'
        self.start_pt = None  # frame coords at mouse down
        self.resize_handle = None  # which handle is used
        self.handle_size = 12  # px in frame coords (will be scaled visually)

    # ---- Helpers: coord mapping ----
    def _label_to_frame(self, x, y):
        fw, fh = self.frame_size
        lw = max(self.width(), 1)
        lh = max(self.height(), 1)
        fx = int(x * fw / lw)
        fy = int(y * fh / lh)
        return fx, fy

    def _normalize_rect(self, r):
        x1, y1, x2, y2 = r
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        # clamp to frame
        fw, fh = self.frame_size
        x1 = max(0, min(fw-1, x1))
        y1 = max(0, min(fh-1, y1))
        x2 = max(0, min(fw-1, x2))
        y2 = max(0, min(fh-1, y2))
        return (x1, y1, x2, y2)

    def _inside_rect(self, x, y, r):
        x1, y1, x2, y2 = r
        return x1 <= x <= x2 and y1 <= y <= y2

    def _near(self, x, y, hx, hy, tol):
        return abs(x - hx) <= tol and abs(y - hy) <= tol

    def _hit_test_handles(self, x, y):
        if not self.rect: return None
        x1, y1, x2, y2 = self.rect
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        hs = self.handle_size
        handles = {
            'tl': (x1, y1), 'tr': (x2, y1),
            'bl': (x1, y2), 'br': (x2, y2),
            'tm': (cx, y1), 'bm': (cx, y2),
            'ml': (x1, cy), 'mr': (x2, cy),
        }
        for name, (hx, hy) in handles.items():
            if self._near(x, y, hx, hy, tol=hs):
                return name
        return None

    # ---- Public API ----
    def clear_selection(self):
        self.rect = None
        self.mode = None
        self.start_pt = None
        self.resize_handle = None

    def set_selection_enabled(self, enabled: bool):
        self.selection_enabled = enabled

    def set_frame_size(self, w, h):
        self.frame_size = (w, h)

    # ---- Events ----
    def mousePressEvent(self, event: QMouseEvent):
        if not self.selection_enabled:
            return
        if event.button() != Qt.LeftButton:
            return
        fx, fy = self._label_to_frame(event.x(), event.y())
        if self.rect:
            handle = self._hit_test_handles(fx, fy)
            if handle:
                self.mode = 'resizing'
                self.resize_handle = handle
                self.start_pt = (fx, fy)
                return
            if self._inside_rect(fx, fy, self.rect):
                self.mode = 'moving'
                self.start_pt = (fx, fy)
                return
        # Start creating a new rectangle
        self.mode = 'creating'
        self.rect = (fx, fy, fx, fy)
        self.start_pt = (fx, fy)

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self.selection_enabled or not self.mode:
            return
        fx, fy = self._label_to_frame(event.x(), event.y())
        if self.mode == 'creating':
            x1, y1 = self.start_pt
            self.rect = self._normalize_rect((x1, y1, fx, fy))
        elif self.mode == 'moving' and self.rect:
            x1, y1, x2, y2 = self.rect
            sx, sy = self.start_pt
            dx, dy = fx - sx, fy - sy
            self.rect = self._normalize_rect((x1 + dx, y1 + dy, x2 + dx, y2 + dy))
            self.start_pt = (fx, fy)
        elif self.mode == 'resizing' and self.rect:
            x1, y1, x2, y2 = self.rect
            if self.resize_handle == 'tl':
                self.rect = self._normalize_rect((fx, fy, x2, y2))
            elif self.resize_handle == 'tr':
                self.rect = self._normalize_rect((x1, fy, fx, y2))
            elif self.resize_handle == 'bl':
                self.rect = self._normalize_rect((fx, y1, x2, fy))
            elif self.resize_handle == 'br':
                self.rect = self._normalize_rect((x1, y1, fx, fy))
            elif self.resize_handle == 'tm':
                self.rect = self._normalize_rect((x1, fy, x2, y2))
            elif self.resize_handle == 'bm':
                self.rect = self._normalize_rect((x1, y1, x2, fy))
            elif self.resize_handle == 'ml':
                self.rect = self._normalize_rect((fx, y1, x2, y2))
            elif self.resize_handle == 'mr':
                self.rect = self._normalize_rect((x1, y1, fx, y2))

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.mode = None
            self.resize_handle = None
            self.start_pt = None


# ---------------------- MAIN GUI CLASS ----------------------
class TurretGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.tracking_enabled = False

        self.setWindowTitle("YOLO Turret Tracking System")
        self.showMaximized()  # ✅ Open full screen

        # 🌑 Global dark theme
        self.setStyleSheet("""
            QWidget {
                background-color: black;
                color: white;
                font-size: 14px;
            }
            QPushButton {
                background-color: #222;
                color: white;
                border: 1px solid #555;
                border-radius: 6px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #444;
            }
            QPushButton:pressed {
                background-color: #666;
            }
            QPushButton#fireButton {
                background-color: red;
                color: white;
                font-size: 18px;
                font-weight: bold;
                border: 2px solid darkred;
                border-radius: 10px;
                padding: 12px 24px;
            }
            QPushButton#fireButton:hover {
                background-color: darkred;
            }
        """)

        # ✅ Set your logo paths here (forward slashes for Windows)
        left_logo_path = "C:/Users/LAWSON/Desktop/PHOTO-2025-08-18-15-11-22.jpeg"
        right_logo_path = "C:/Users/LAWSON/Desktop/NNlogo.png"

        # Logo labels
        self.left_logo = QLabel(self)
        self.right_logo = QLabel(self)
        self.set_logo(self.left_logo, left_logo_path)
        self.set_logo(self.right_logo, right_logo_path)

        # Top logo layout
        logo_layout = QHBoxLayout()
        logo_layout.addWidget(self.left_logo, alignment=Qt.AlignLeft)
        logo_layout.addStretch()
        logo_layout.addWidget(self.right_logo, alignment=Qt.AlignRight)

        # Video display (centralized and larger) — interactive label
        self.video_label = InteractiveVideoLabel(self)
        self.video_label.setFixedSize(1000, 700)  # ✅ Bigger frame
        self.video_label.setStyleSheet("background-color: black;")

        # Tracking toggle
        self.track_checkbox = QCheckBox("Enable Tracking")
        self.track_checkbox.stateChanged.connect(self.toggle_tracking)

        # NEW: Select Target button (enables/disables mouse selection)
        self.select_button = QPushButton("Select Target")
        self.select_button.setCheckable(True)
        self.select_button.toggled.connect(self.on_select_target_toggled)

        # Servo sliders
        self.slider_x = QSlider(Qt.Horizontal)
        self.slider_x.setRange(SERVO_X_MIN, SERVO_X_MAX)
        self.slider_x.setValue(servo_x_angle)
        self.slider_x.valueChanged.connect(self.update_x_angle)

        self.slider_y = QSlider(Qt.Horizontal)
        self.slider_y.setRange(SERVO_Y_MIN, SERVO_Y_MAX)
        self.slider_y.setValue(servo_y_angle)
        self.slider_y.valueChanged.connect(self.update_y_angle)

        # Servo angle labels
        self.label_x = QLabel(f"Pan: {servo_x_angle}")
        self.label_y = QLabel(f"Tilt: {servo_y_angle}")
        self.label_x.setStyleSheet("color: cyan; font-weight: bold;")
        self.label_y.setStyleSheet("color: lime; font-weight: bold;")

        # Buttons
        self.reset_button = QPushButton("Reset Position")
        self.reset_button.clicked.connect(self.reset_position)

        self.fire_button = QPushButton("🔥 FIRE")
        self.fire_button.setObjectName("fireButton")
        self.fire_button.clicked.connect(send_fire_command)

        self.stop_button = QPushButton("Stop")
        self.pause_button = QPushButton("Pause")
        self.resume_button = QPushButton("Resume")
        self.calibrate_button = QPushButton("Calibrate")
        self.save_button = QPushButton("Save Settings")
        self.load_button = QPushButton("Load Settings")
        self.settings_button = QPushButton("Settings")
        self.help_button = QPushButton("Help")
        self.exit_button = QPushButton("Exit")

        # --- Layout ---
        main_layout = QVBoxLayout()
        main_layout.addLayout(logo_layout)

        content_layout = QHBoxLayout()

        # Left sidebar
        left_sidebar = QVBoxLayout()
        left_sidebar.addWidget(self.track_checkbox)
        left_sidebar.addWidget(self.select_button)
        left_sidebar.addWidget(self.reset_button)
        left_sidebar.addWidget(self.fire_button)
        left_sidebar.addWidget(self.stop_button)
        left_sidebar.addWidget(self.pause_button)
        left_sidebar.addWidget(self.resume_button)

        # ✅ Center video
        video_layout = QVBoxLayout()
        video_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)

        # Right sidebar
        right_sidebar = QVBoxLayout()
        right_sidebar.addWidget(self.label_x)
        right_sidebar.addWidget(self.slider_x)
        right_sidebar.addWidget(self.label_y)
        right_sidebar.addWidget(self.slider_y)
        right_sidebar.addWidget(self.calibrate_button)
        right_sidebar.addWidget(self.save_button)
        right_sidebar.addWidget(self.load_button)
        right_sidebar.addWidget(self.settings_button)
        right_sidebar.addWidget(self.help_button)
        right_sidebar.addWidget(self.exit_button)

        # Add all to horizontal layout
        content_layout.addLayout(left_sidebar, 1)
        content_layout.addLayout(video_layout, 5)
        content_layout.addLayout(right_sidebar, 1)

        main_layout.addLayout(content_layout)
        self.setLayout(main_layout)

        # ✅ Expand buttons and sliders to fill sidebar space
        for widget in [
            self.track_checkbox, self.select_button, self.reset_button, self.fire_button,
            self.stop_button, self.pause_button, self.resume_button,
            self.calibrate_button, self.save_button, self.load_button,
            self.settings_button, self.help_button, self.exit_button,
            self.slider_x, self.slider_y
        ]:
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Timer for video update
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~33 FPS

    # --- Helper: Set logos ---
    def set_logo(self, label, path):
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(60, 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            label.setPixmap(pixmap)

    # --- GUI Controls ---
    def toggle_tracking(self, state):
        self.tracking_enabled = state == Qt.Checked

    def on_select_target_toggled(self, checked: bool):
        self.video_label.set_selection_enabled(checked)
        if not checked:
            # Keep the last box; user can re-enable to adjust. Do nothing on uncheck.
            pass

    def update_x_angle(self, value):
        global servo_x_angle
        servo_x_angle = value
        self.label_x.setText(f"Pan: {servo_x_angle}")
        send_servo_angles(servo_x_angle, servo_y_angle)

    def update_y_angle(self, value):
        global servo_y_angle
        servo_y_angle = value
        self.label_y.setText(f"Tilt: {servo_y_angle}")
        send_servo_angles(servo_x_angle, servo_y_angle)

    def reset_position(self):
        global servo_x_angle, servo_y_angle
        servo_x_angle, servo_y_angle = 90, 90
        self.slider_x.setValue(90)
        self.slider_y.setValue(90)
        send_servo_angles(servo_x_angle, servo_y_angle)

    # --- Main Tracking Loop ---
    def update_frame(self):
        global servo_x_angle, servo_y_angle

        if not cap.isOpened():
            return

        ret, frame = cap.read()
        if not ret:
            return

        # Update label's knowledge of the frame size for proper mouse mapping
        h, w = frame.shape[:2]
        self.video_label.set_frame_size(w, h)

        if self.tracking_enabled:
            results = model(frame, stream=True)
            for result in results:
                person_boxes = []
                for box in result.boxes:
                    cls = int(box.cls[0])
                    if model.names[cls] == "person":
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        area = (x2 - x1) * (y2 - y1)
                        person_boxes.append((area, (x1, y1, x2, y2)))

                if person_boxes:
                    # Largest person
                    _, (x1, y1, x2, y2) = max(person_boxes, key=lambda b: b[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, ), 3)
                    label = "CLOSEST TARGET SELECTED AND READY TO FIRE"
                    # Text position (clamp inside image)
                    ty = max(20, y1 - 10)
                    cv2.putText(frame, label, (x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 3, cv2.LINE_AA)

                    # Turret control (deadzone + limited per-frame change)
                    dx = cx - w // 2
                    dy = cy - h // 2
                    dx_norm = dx / (w / 2)
                    dy_norm = dy / (h / 2)

                    DEAD_ZONE_X = 30
                    DEAD_ZONE_Y = 30
                    MAX_PAN_CHANGE = 2.0
                    MAX_TILT_CHANGE = 2.0

                    if abs(dx) > DEAD_ZONE_X:
                        servo_x_angle += dx_norm * MAX_PAN_CHANGE
                    if abs(dy) > DEAD_ZONE_Y:
                        servo_y_angle += dy_norm * MAX_TILT_CHANGE

                    servo_x_angle = max(SERVO_X_MIN, min(SERVO_X_MAX, servo_x_angle))
                    servo_y_angle = max(SERVO_Y_MIN, min(SERVO_Y_MAX, servo_y_angle))

                    self.slider_x.setValue(int(servo_x_angle))
                    self.slider_y.setValue(int(servo_y_angle))
                    send_servo_angles(servo_x_angle, servo_y_angle)

        # ---- Overlay: user-selected target box (always shown if present) ----
        if self.video_label.rect:
            x1, y1, x2, y2 = self.video_label.rect
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            label = "TARGET SELECTED AND READY TO FIRE"
            # Text position (clamp inside image)
            ty = max(20, y1 - 10)
            cv2.putText(frame, label, (x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3, cv2.LINE_AA)

            # Optional: draw resize handles
            def hdot(px, py):
                cv2.rectangle(frame, (px-4, py-4), (px+4, py+4), (0, 0, 255), -1)
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            for (px, py) in [(x1, y1), (x2, y1), (x1, y2), (x2, y2),
                             (cx, y1), (cx, y2), (x1, cy), (x2, cy)]:
                hdot(px, py)

        # Display frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_image = QImage(rgb_frame.data, w, h, 3*w, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

# ---------------------- MAIN ----------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TurretGUI()
    window.show()
    sys.exit(app.exec_())


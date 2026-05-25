"""
noark_gui_pyside6.py
--------------------
PySide6 port of the NOARK Force Control GUI.
Run on the Pi:

    pip install PySide6 opencv-python-headless pillow
    python noark_gui_pyside6.py

Controls:
    Click / drag canvas  → set force direction
    Slider               → force magnitude (0–24 N)
    Reset Encoders       → zero encoders
    Send to Teensy       → toggle continuous force sending at 60Hz
    E-STOP               → zero force immediately
"""

import sys, os, math, time, threading, socket
import cv2
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QSlider, QFrame, QSizePolicy,
)
from PySide6.QtGui import (
    QPainter, QPen, QColor, QBrush, QFont, QPixmap, QImage,
    QPolygonF, QPainterPath,
)
from PySide6.QtCore import Qt, QTimer, QPointF, Signal, QObject, QRectF

sys.path.insert(0, '/home/sujith/Documents/NOARK_backbone')
from camera_pose_gui import MainClass
from vaideesh.control_implementation_with_GUI.pyteensy_gui import TeensyPort

# ── calibration paths ─────────────────────────────────────────────────────
CAM_TOML   = "/home/sujith/Documents/NOARK_backbone/notebooks/calibration/output/good.toml"
TABLE_TOML = "/home/sujith/Documents/NOARK_backbone/estimator/charuco_pose/charuco_pose.toml"

# ── geometry (table-frame metres, x-z plane) ─────────────────────────────
P1 = (-0.495, -0.773)   # left pulley
P3 = ( 0.495, -0.773)   # right pulley
ML = (-0.065, -0.707)   # left motor
MR = ( 0.065, -0.707)   # right motor
R_SPOOL  = 0.033         # motor spool radius (m)
MAX_F    = 24.0           # N
T_MIN    = 0.07 / R_SPOOL # minimum tension from 0.07 Nm pretension = ~2.12 N
MAX_TORQUE_RATE = 0.013   # Nm per cycle at 60Hz → ~0.8 Nm/s

# ── UDP load cell config ──────────────────────────────────────────────────
LC_UDP_PORT = 5005        # Pi listens on this port for load cell data
LC_STALE_MS = 100         # ms — warn if data older than this

# ── world bounds ──────────────────────────────────────────────────────────
WX0, WX1 = -0.50,  0.50
WZ0, WZ1 = -0.80,  0.00

# ── colours ───────────────────────────────────────────────────────────────
C_BG      = QColor("#0d0d0d")
C_BG2     = QColor("#141414")
C_SIDE    = QColor("#161616")
C_GRID    = QColor("#1e1e1e")
C_RAIL    = QColor("#3a3a3a")
C_MUTED   = QColor("#555")
C_FG      = QColor("#ccc")
C_BLUE    = QColor("#50b4ff")
C_GREEN   = QColor("#44cc88")
C_AMBER   = QColor("#ffbb00")
C_RED     = QColor("#f05050")
C_WHITE   = QColor("#ffffff")
C_SEP     = QColor("#2a2a2a")


# ─────────────────────────────────────────────────────────────────────────
# Kinematics helper
# ─────────────────────────────────────────────────────────────────────────
def solve_tensions(nx, nz, Fx, Fz):
    def unit(dx, dz):
        l = math.hypot(dx, dz)
        return (dx / l, dz / l) if l > 1e-9 else (0.0, 0.0)

    u1x, u1z = unit(P1[0] - nx, P1[1] - nz)
    u3x, u3z = unit(P3[0] - nx, P3[1] - nz)
    det = u1x * u3z - u3x * u1z
    if abs(det) < 1e-6:
        return None
    return {
        "T1": (Fx * u3z - Fz * u3x) / det,
        "T3": (u1x * Fz - u1z * Fx) / det,
    }


# ─────────────────────────────────────────────────────────────────────────
# Load Cell UDP Receiver
# ─────────────────────────────────────────────────────────────────────────
class LoadCellReceiver:
    """
    Listens for UDP packets from laptop running loadcell_sender.py.
    Expects CSV format: "lc_x,lc_y,lc_z,timestamp"
    lc_x → NOARK X axis
    lc_y → NOARK Z axis
    """
    def __init__(self, port=LC_UDP_PORT):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("0.0.0.0", port))
        self.sock.settimeout(0.01)
        self.lc_x      = 0.0   # kg — maps to NOARK X
        self.lc_y      = 0.0   # kg — maps to NOARK Z
        self.lc_z      = 0.0   # kg — unused axis
        self.timestamp = 0.0
        self.stale     = True
        self._running  = True
        threading.Thread(target=self._read_loop, daemon=True).start()
        print(f"[loadcell] UDP receiver listening on port {port}")

    def _read_loop(self):
        while self._running:
            try:
                data, _ = self.sock.recvfrom(1024)
                line    = data.decode().strip()
                parts   = line.split(",")
                if len(parts) == 4:
                    self.lc_x      = float(parts[0])
                    self.lc_y      = float(parts[1])
                    self.lc_z      = float(parts[2])
                    self.timestamp = float(parts[3])
                    self.stale     = False
                elif len(parts) == 3:
                    # fallback: no timestamp
                    self.lc_x  = float(parts[0])
                    self.lc_y  = float(parts[1])
                    self.lc_z  = float(parts[2])
                    self.stale = False
            except:
                pass

    def is_stale(self):
        if self.timestamp == 0.0:
            return True
        age_ms = (time.time() - self.timestamp) * 1000
        return age_ms > LC_STALE_MS

    def stop(self):
        self._running = False
        self.sock.close()


# ─────────────────────────────────────────────────────────────────────────
# Shared state (thread-safe via lock)
# ─────────────────────────────────────────────────────────────────────────
class State:
    def __init__(self):
        self.lock        = threading.Lock()

        # NOARK pose
        self.noark_x     = 0.0
        self.noark_z     = -0.4
        self.has_noark   = False

        # Encoders
        self.enc1        = 0.0
        self.enc2        = 0.0

        # Force command
        self.dir_x       = 0.0
        self.dir_z       = -1.0
        self.force_mag   = 12.0

        # Motor torque history for rate limiting
        self.prev_tau1   = 0.0
        self.prev_tau2   = 0.0

        # Camera frame
        self.video_frame = None

        # Load cell raw (kg)
        self.lc_x        = 0.0
        self.lc_y        = 0.0
        self.lc_z        = 0.0
        self.lc_stale    = True

        # Commanded force (N)
        self.cmd_fx      = 0.0
        self.cmd_fz      = 0.0
        self.cmd_mag     = 0.0
        self.cmd_dir     = 0.0   # degrees

        # Measured force from load cell (N)
        self.meas_fx     = 0.0
        self.meas_fz     = 0.0
        self.meas_mag    = 0.0
        self.meas_dir    = 0.0   # degrees

        # Errors
        self.err_mag     = 0.0
        self.err_dir     = 0.0   # degrees

STATE = State()


# ─────────────────────────────────────────────────────────────────────────
# Workspace canvas — draws the 2-D table view
# ─────────────────────────────────────────────────────────────────────────
class WorkspaceCanvas(QWidget):
    directionChanged = Signal(float, float)   # (dir_x, dir_z)

    def __init__(self, state: State, parent=None):
        super().__init__(parent)
        self.state = state
        self.setMinimumHeight(240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

    def _w2c(self, wx, wz):
        w, h = self.width(), self.height()
        cx = (wx - WX0) / (WX1 - WX0) * w
        cy = (wz - WZ0) / (WZ1 - WZ0) * h
        return QPointF(cx, cy)

    def _c2w(self, cx, cy):
        w, h = self.width(), self.height()
        return (WX0 + cx / w * (WX1 - WX0),
                WZ0 + cy / h * (WZ1 - WZ0))

    def _update_dir(self, event):
        s = self.state
        with s.lock:
            np_ = self._w2c(s.noark_x, s.noark_z)
        dx = event.position().x() - np_.x()
        dy = event.position().y() - np_.y()
        length = math.hypot(dx, dy)
        if length < 4:
            return
        self.directionChanged.emit(dx / length, dy / length)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._update_dir(e)

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton:
            self._update_dir(e)

    def paintEvent(self, _):
        s = self.state
        with s.lock:
            nx_w, nz_w    = s.noark_x, s.noark_z
            has_noark     = s.has_noark
            dir_x, dir_z  = s.dir_x, s.dir_z
            force_mag     = s.force_mag
            meas_fx       = s.meas_fx
            meas_fz       = s.meas_fz
            meas_mag      = s.meas_mag

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        p.fillRect(0, 0, w, h, C_BG)

        # Grid
        pen = QPen(C_GRID, 1)
        p.setPen(pen)
        for xv in [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]:
            pt = self._w2c(xv, 0)
            p.drawLine(QPointF(pt.x(), 0), QPointF(pt.x(), h))
        for zv in [-1.0, -0.8, -0.6, -0.4, -0.2]:
            pt = self._w2c(0, zv)
            p.drawLine(QPointF(0, pt.y()), QPointF(w, pt.y()))

        # Top rail
        r0 = self._w2c(-0.6, -0.773)
        r1 = self._w2c( 0.6, -0.773)
        p.setPen(QPen(C_RAIL, 2))
        p.drawLine(r0, r1)

        # Solve tensions
        Fx = force_mag * dir_x
        Fz = force_mag * dir_z
        sol = solve_tensions(nx_w, nz_w, Fx, Fz)
        T1 = max(0.0, sol["T1"]) if sol else 0.0
        T3 = max(0.0, sol["T3"]) if sol else 0.0

        np_ = self._w2c(nx_w, nz_w)
        ml  = self._w2c(*ML)
        mr  = self._w2c(*MR)
        p1  = self._w2c(*P1)
        p3  = self._w2c(*P3)

        def cable_color(base: QColor, tension, max_t):
            alpha = int(64 + tension / max_t * 191) if max_t > 0 else 64
            c = QColor(base)
            c.setAlpha(alpha)
            return c

        def draw_cable(pts, color, width):
            pen = QPen(color, width)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            p.setPen(pen)
            path = QPainterPath()
            path.moveTo(pts[0])
            for pt in pts[1:]:
                path.lineTo(pt)
            p.drawPath(path)

        w1 = 1.0 + T1 / MAX_F * 3
        w3 = 1.0 + T3 / MAX_F * 3
        draw_cable([ml, p1, np_], cable_color(C_BLUE,  T1, MAX_F), w1)
        draw_cable([mr, p3, np_], cable_color(C_GREEN, T3, MAX_F), w3)

        font_small = QFont("Courier New", 8)
        p.setFont(font_small)

        # Motors
        for pt, lbl in [(ml, "ML"), (mr, "MR")]:
            rect = QRectF(pt.x() - 7, pt.y() - 7, 14, 14)
            p.setBrush(QBrush(QColor("#7D4A7A")))
            p.setPen(QPen(C_AMBER, 1.5))
            p.drawRect(rect)
            p.setPen(C_AMBER)
            p.drawText(QPointF(pt.x() - 8, pt.y() + 22), lbl)

        # Pulleys
        for pt, lbl, col in [(p1, "P1", C_BLUE), (p3, "P3", C_GREEN)]:
            p.setBrush(QBrush(C_BG))
            p.setPen(QPen(col, 1.5))
            p.drawEllipse(QPointF(pt.x(), pt.y()), 7, 7)
            p.setPen(col)
            p.drawText(QPointF(pt.x() - 6, pt.y() - 10), lbl)

        # NOARK dot
        ncol = C_RED if has_noark else QColor("#444")
        p.setBrush(QBrush(QColor("#220d0d") if has_noark else QColor("#1a1a1a")))
        p.setPen(QPen(ncol, 2))
        p.drawEllipse(QPointF(np_.x(), np_.y()), 10, 10)
        p.setPen(C_WHITE)
        p.setFont(QFont("Courier New", 8, QFont.Bold))
        p.drawText(QRectF(np_.x() - 8, np_.y() - 6, 16, 12),
                   Qt.AlignCenter, "N")
        p.setPen(ncol)
        p.setFont(font_small)
        p.drawText(QPointF(np_.x() - 30, np_.y() - 14),
                   f"({nx_w:.3f}, {nz_w:.3f})")

        # Commanded force arrow (red)
        if force_mag > 0.1 and has_noark:
            length = 35 + force_mag / MAX_F * 45
            ax = np_.x() + dir_x * length
            ay = np_.y() + dir_z * length
            pen = QPen(C_RED, 2)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.drawLine(QPointF(np_.x(), np_.y()), QPointF(ax, ay))
            self._draw_arrowhead(p, np_.x(), np_.y(), ax, ay, C_RED)
            p.setPen(C_RED)
            p.setFont(font_small)
            p.drawText(QPointF(ax + 6, ay - 4), f"{force_mag:.1f}N cmd")

        # Measured force arrow (green) from load cell
        if meas_mag > 0.1 and has_noark:
            meas_dx = meas_fx / meas_mag if meas_mag > 0 else 0
            meas_dz = meas_fz / meas_mag if meas_mag > 0 else 0
            length  = 35 + meas_mag / MAX_F * 45
            mx = np_.x() + meas_dx * length
            my = np_.y() + meas_dz * length
            pen = QPen(C_GREEN, 2)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.drawLine(QPointF(np_.x(), np_.y()), QPointF(mx, my))
            self._draw_arrowhead(p, np_.x(), np_.y(), mx, my, C_GREEN)
            p.setPen(C_GREEN)
            p.setFont(font_small)
            p.drawText(QPointF(mx + 6, my + 10), f"{meas_mag:.1f}N meas")

        p.end()

    @staticmethod
    def _draw_arrowhead(painter, x0, y0, x1, y1, color):
        angle  = math.atan2(y1 - y0, x1 - x0)
        size   = 10
        spread = math.radians(25)
        pts = QPolygonF([
            QPointF(x1, y1),
            QPointF(x1 - size * math.cos(angle - spread),
                    y1 - size * math.sin(angle - spread)),
            QPointF(x1 - size * math.cos(angle + spread),
                    y1 - size * math.sin(angle + spread)),
        ])
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.NoPen)
        painter.drawPolygon(pts)


# ─────────────────────────────────────────────────────────────────────────
# Camera feed widget
# ─────────────────────────────────────────────────────────────────────────
class CameraCanvas(QWidget):
    def __init__(self, state: State, parent=None):
        super().__init__(parent)
        self.state = state
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pixmap = None

    def update_frame(self):
        with self.state.lock:
            frame = self.state.video_frame
        if frame is None:
            return
        fh, fw = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, fw, fh, fw * 3, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(img)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(0, 0, self.width(), self.height(), C_BG)
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.width(), self.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            x = (self.width()  - scaled.width())  // 2
            y = (self.height() - scaled.height()) // 2
            p.drawPixmap(x, y, scaled)
        else:
            p.setPen(C_MUTED)
            p.setFont(QFont("Courier New", 10))
            p.drawText(self.rect(), Qt.AlignCenter, "no camera feed")
        p.end()


# ─────────────────────────────────────────────────────────────────────────
# Sidebar helpers
# ─────────────────────────────────────────────────────────────────────────
def _section_label(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setFont(QFont("Courier New", 8))
    lbl.setStyleSheet("color: #444; padding: 6px 10px 2px 10px;")
    return lbl


def _separator() -> QFrame:
    f = QFrame()
    f.setFrameShape(QFrame.HLine)
    f.setStyleSheet("color: #2a2a2a; margin: 0 10px;")
    return f


def _data_row(parent_layout: QVBoxLayout, label: str, color: str = "#ccc"):
    row = QWidget()
    row.setStyleSheet("background: transparent;")
    hl = QHBoxLayout(row)
    hl.setContentsMargins(10, 2, 10, 2)
    lbl_w = QLabel(label)
    lbl_w.setFont(QFont("Courier New", 12))
    lbl_w.setStyleSheet("color: #555;")
    val_w = QLabel("—")
    val_w.setFont(QFont("Courier New", 11, QFont.Bold))
    val_w.setStyleSheet(f"color: {color};")
    val_w.setAlignment(Qt.AlignRight)
    hl.addWidget(lbl_w)
    hl.addWidget(val_w)
    parent_layout.addWidget(row)
    return val_w


def _button(text: str, fg: str, bg: str, border: str) -> QPushButton:
    btn = QPushButton(text)
    btn.setFont(QFont("Courier New", 10))
    btn.setFixedHeight(38)
    btn.setCursor(Qt.PointingHandCursor)
    btn.setStyleSheet(f"""
        QPushButton {{
            background: {bg};
            color: {fg};
            border: 1px solid {border};
            border-radius: 2px;
            padding: 0 8px;
        }}
        QPushButton:hover {{ background: {border}22; }}
        QPushButton:pressed {{ background: {border}44; }}
    """)
    return btn


# ─────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────
class NOARKWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NOARK Force Control")
        self.resize(960, 650)
        self.setStyleSheet("QMainWindow, QWidget { background: #111; }")

        self.state    = STATE
        self.running  = True
        self._enc     = None
        self._sending = False   # toggle for continuous send

        self._build_ui()
        self._start_camera()
        self._start_teensy()
        self._start_loadcell()
        self._start_send_loop()

        # UI refresh at 20Hz
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(50)

    # ──────────────────────────────────────────────────────────────────
    # BUILD UI
    # ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_h = QHBoxLayout(central)
        root_h.setContentsMargins(8, 8, 8, 8)
        root_h.setSpacing(8)

        # ── left: stacked canvases ────────────────────────────────────
        left   = QWidget()
        left_v = QVBoxLayout(left)
        left_v.setContentsMargins(0, 0, 0, 0)
        left_v.setSpacing(6)

        self.workspace = WorkspaceCanvas(self.state)
        self.workspace.directionChanged.connect(self._on_direction)
        self.workspace.setStyleSheet(
            "border: 1px solid #2a2a2a; border-radius: 2px;")
        left_v.addWidget(self.workspace, 3)

        self.cam_view = CameraCanvas(self.state)
        self.cam_view.setStyleSheet(
            "border: 1px solid #2a2a2a; border-radius: 2px;")
        left_v.addWidget(self.cam_view, 2)

        root_h.addWidget(left, 3)

        # ── right: sidebar ────────────────────────────────────────────
        sidebar = QWidget()
        sidebar.setFixedWidth(230)
        sidebar.setStyleSheet("background: #141414; border-radius: 2px;")
        side_v = QVBoxLayout(sidebar)
        side_v.setContentsMargins(0, 4, 0, 8)
        side_v.setSpacing(0)

        # NOARK position
        side_v.addWidget(_section_label("NOARK position (m)"))
        side_v.addWidget(_separator())
        self.v_nx = _data_row(side_v, "x", "#50b4ff")
        self.v_nz = _data_row(side_v, "z", "#50b4ff")

        cam_row = QWidget(); cam_row.setStyleSheet("background:transparent;")
        cam_hl  = QHBoxLayout(cam_row); cam_hl.setContentsMargins(10,2,10,2)
        lbl_cam = QLabel("camera"); lbl_cam.setFont(QFont("Courier New",12))
        lbl_cam.setStyleSheet("color:#555;")
        self.v_cam = QLabel("searching…")
        self.v_cam.setFont(QFont("Courier New",12))
        self.v_cam.setStyleSheet("color:#f05050;")
        self.v_cam.setAlignment(Qt.AlignRight)
        cam_hl.addWidget(lbl_cam); cam_hl.addWidget(self.v_cam)
        side_v.addWidget(cam_row)

        # Encoders
        side_v.addWidget(_section_label("Encoders (°)"))
        side_v.addWidget(_separator())
        self.v_e1 = _data_row(side_v, "enc 1 (left)",  "#ffbb00")
        self.v_e2 = _data_row(side_v, "enc 2 (right)", "#ffbb00")

        btn_reset = _button("reset encoders", "#ffbb00", "#1a1400", "#ffbb00")
        btn_reset.clicked.connect(self._reset_encoders)
        w = QWidget(); w.setStyleSheet("background:transparent;")
        wl = QVBoxLayout(w); wl.setContentsMargins(10,4,10,2); wl.addWidget(btn_reset)
        side_v.addWidget(w)

        # Force slider
        side_v.addWidget(_section_label("Force"))
        side_v.addWidget(_separator())
        self.v_fmag = _data_row(side_v, "magnitude", "#f05050")
        self.v_fdir = _data_row(side_v, "direction", "#888")

        sl_wrap = QWidget(); sl_wrap.setStyleSheet("background:transparent;")
        sl_l = QVBoxLayout(sl_wrap); sl_l.setContentsMargins(10, 2, 10, 2)
        self.fmag_slider = QSlider(Qt.Horizontal)
        self.fmag_slider.setRange(0, int(MAX_F * 10))
        self.fmag_slider.setValue(120)
        self.fmag_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                background: #2a2a2a; height: 4px; border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #f05050; width: 14px; height: 14px;
                margin: -5px 0; border-radius: 7px;
            }
            QSlider::sub-page:horizontal { background: #f05050; border-radius:2px; }
        """)
        self.fmag_slider.valueChanged.connect(self._on_force_slider)
        sl_l.addWidget(self.fmag_slider)
        side_v.addWidget(sl_wrap)

        # Cable tensions
        side_v.addWidget(_section_label("Cable tensions"))
        side_v.addWidget(_separator())
        self.v_t1 = _data_row(side_v, "T₁ left",  "#50b4ff")
        self.v_t3 = _data_row(side_v, "T₃ right", "#44cc88")

        # Motor torques
        side_v.addWidget(_section_label("Motor torques (Nm)"))
        side_v.addWidget(_separator())
        self.v_tau1 = _data_row(side_v, "τ₁", "#50b4ff")
        self.v_tau2 = _data_row(side_v, "τ₂", "#44cc88")

        # ── Commanded force ───────────────────────────────────────────
        side_v.addWidget(_section_label("Commanded force"))
        side_v.addWidget(_separator())
        self.v_cmd_mag = _data_row(side_v, "magnitude", "#f05050")
        self.v_cmd_dir = _data_row(side_v, "direction", "#f05050")

        # ── Measured force (load cell) ────────────────────────────────
        side_v.addWidget(_section_label("Measured force"))
        side_v.addWidget(_separator())
        self.v_lc_status = _data_row(side_v, "status",    "#888")
        self.v_lc_mag    = _data_row(side_v, "magnitude", "#44cc88")
        self.v_lc_dir    = _data_row(side_v, "direction", "#44cc88")

        # ── Error ─────────────────────────────────────────────────────
        side_v.addWidget(_section_label("Error"))
        side_v.addWidget(_separator())
        self.v_err_mag = _data_row(side_v, "mag error", "#ffbb00")
        self.v_err_dir = _data_row(side_v, "dir error", "#ffbb00")

        side_v.addStretch()

        # ── Buttons ───────────────────────────────────────────────────
        self.btn_send = _button("▶  send to Teensy", "#50b4ff", "#0a1020", "#50b4ff")
        self.btn_send.clicked.connect(self._toggle_send)

        btn_estop = _button("■  E-STOP", "#f05050", "#1a0808", "#f05050")
        btn_estop.setFont(QFont("Courier New", 12, QFont.Bold))
        btn_estop.clicked.connect(self._estop)

        for btn in (self.btn_send, btn_estop):
            bw = QWidget(); bw.setStyleSheet("background:transparent;")
            bl = QVBoxLayout(bw); bl.setContentsMargins(10,2,10,2); bl.addWidget(btn)
            side_v.addWidget(bw)

        root_h.addWidget(sidebar, 0)

    # ──────────────────────────────────────────────────────────────────
    # REFRESH (main-thread timer ~20Hz)
    # ──────────────────────────────────────────────────────────────────
    def _refresh(self):
        s = self.state
        with s.lock:
            has_noark    = s.has_noark
            nx, nz       = s.noark_x, s.noark_z
            e1, e2       = s.enc1, s.enc2
            dir_x, dir_z = s.dir_x, s.dir_z
            force_mag    = s.force_mag
            lc_stale     = s.lc_stale
            lc_x         = s.lc_x
            lc_y         = s.lc_y

        # ── NOARK position ────────────────────────────────────────────
        if has_noark:
            self.v_nx.setText(f"{nx:.4f}")
            self.v_nz.setText(f"{nz:.4f}")
            self.v_cam.setText("tracking")
            self.v_cam.setStyleSheet("color:#44cc88;")
        else:
            self.v_nx.setText("—")
            self.v_nz.setText("—")
            self.v_cam.setText("searching…")
            self.v_cam.setStyleSheet("color:#f05050;")

        # ── Encoders ──────────────────────────────────────────────────
        self.v_e1.setText(f"{e1:.2f}")
        self.v_e2.setText(f"{e2:.2f}")

        # ── Commanded force ───────────────────────────────────────────
        cmd_fx  = force_mag * dir_x
        cmd_fz  = force_mag * dir_z
        cmd_mag = force_mag
        cmd_dir = math.degrees(math.atan2(cmd_fz, cmd_fx))

        ang = math.degrees(math.atan2(dir_z, dir_x))
        self.v_fdir.setText(f"{ang:.1f}°")
        self.v_fmag.setText(f"{force_mag:.1f} N")
        self.v_cmd_mag.setText(f"{cmd_mag:.2f} N")
        self.v_cmd_dir.setText(f"{cmd_dir:.1f}°")

        # ── Tensions / torques ────────────────────────────────────────
        sol = solve_tensions(nx, nz, cmd_fx, cmd_fz)
        if sol and has_noark:
            T1   = max(T_MIN, sol["T1"])
            T3   = max(T_MIN, sol["T3"])
            tau1 = -(T1 * R_SPOOL)
            tau2 =   T3 * R_SPOOL
            self.v_t1.setText(f"{T1:.2f} N")
            self.v_t3.setText(f"{T3:.2f} N")
            self.v_tau1.setText(f"{tau1:.3f}")
            self.v_tau2.setText(f"{tau2:.3f}")
        else:
            for w in (self.v_t1, self.v_t3, self.v_tau1, self.v_tau2):
                w.setText("—")

        # ── Load cell measured force ──────────────────────────────────
        # Axis mapping: load cell X → NOARK X, load cell Y → NOARK Z
        meas_fx  = lc_x * 9.81   # kg → N
        meas_fz  = lc_y * 9.81   # kg → N
        meas_mag = math.hypot(meas_fx, meas_fz)
        meas_dir = math.degrees(math.atan2(meas_fz, meas_fx))

        # Update state for canvas drawing
        with s.lock:
            s.cmd_fx   = cmd_fx
            s.cmd_fz   = cmd_fz
            s.cmd_mag  = cmd_mag
            s.cmd_dir  = cmd_dir
            s.meas_fx  = meas_fx
            s.meas_fz  = meas_fz
            s.meas_mag = meas_mag
            s.meas_dir = meas_dir

        # Load cell status
        if lc_stale:
            self.v_lc_status.setText("no data")
            self.v_lc_status.setStyleSheet("color:#f05050;")
            self.v_lc_mag.setText("—")
            self.v_lc_dir.setText("—")
            self.v_err_mag.setText("—")
            self.v_err_dir.setText("—")
        else:
            self.v_lc_status.setText("live")
            self.v_lc_status.setStyleSheet("color:#44cc88;")
            self.v_lc_mag.setText(f"{meas_mag:.2f} N")
            self.v_lc_dir.setText(f"{meas_dir:.1f}°")

            # ── Errors ────────────────────────────────────────────────
            err_mag = abs(cmd_mag - meas_mag)

            # Direction error with wrap-around handling
            err_dir = abs(cmd_dir - meas_dir)
            if err_dir > 180:
                err_dir = 360 - err_dir

            self.v_err_mag.setText(f"{err_mag:.2f} N")
            self.v_err_dir.setText(f"{err_dir:.1f}°")

            # Colour code errors
            mag_color = "#44cc88" if err_mag < 1.0 else "#ffbb00" if err_mag < 3.0 else "#f05050"
            dir_color = "#44cc88" if err_dir < 5.0 else "#ffbb00" if err_dir < 15.0 else "#f05050"
            self.v_err_mag.setStyleSheet(f"color:{mag_color};")
            self.v_err_dir.setStyleSheet(f"color:{dir_color};")

        # ── Repaint ───────────────────────────────────────────────────
        self.workspace.update()
        self.cam_view.update_frame()

    # ──────────────────────────────────────────────────────────────────
    # SLOTS
    # ──────────────────────────────────────────────────────────────────
    def _on_direction(self, dx: float, dz: float):
        with self.state.lock:
            self.state.dir_x = dx
            self.state.dir_z = dz

    def _on_force_slider(self, val: int):
        with self.state.lock:
            self.state.force_mag = val / 10.0

    def _reset_encoders(self):
        if self._enc:
            self._enc.encoder_reset()

    def _toggle_send(self):
        """Toggle continuous force sending on/off."""
        self._sending = not self._sending
        if self._sending:
            self.btn_send.setText("⏹  stop sending")
            self.btn_send.setStyleSheet(self.btn_send.styleSheet().replace("#0a1020", "#0a2010"))
            print("[send] Continuous sending STARTED at 60Hz")
        else:
            self.btn_send.setText("▶  send to Teensy")
            print("[send] Continuous sending STOPPED")
            if self._enc:
                self._enc.serialInst.write(b"0.000,0.000\n")

    def _send_force(self):
        s = self.state
        with s.lock:
            has_noark    = s.has_noark
            nx, nz       = s.noark_x, s.noark_z
            dir_x, dir_z = s.dir_x,   s.dir_z
            force_mag    = s.force_mag
            prev_tau1    = s.prev_tau1
            prev_tau2    = s.prev_tau2

        if not has_noark:
            return

        Fx, Fz = force_mag * dir_x, force_mag * dir_z
        sol = solve_tensions(nx, nz, Fx, Fz)
        if not sol:
            return

        # Fix 1 — minimum pretension (0.07 Nm → 2.12 N)
        T1 = max(T_MIN, sol["T1"])
        T3 = max(T_MIN, sol["T3"])

        tau1 = -(T1 * R_SPOOL)
        tau2 =   T3 * R_SPOOL

        # Fix 2 — rate limiting to smooth transitions
        tau1 = np.clip(tau1, prev_tau1 - MAX_TORQUE_RATE,
                             prev_tau1 + MAX_TORQUE_RATE)
        tau2 = np.clip(tau2, prev_tau2 - MAX_TORQUE_RATE,
                             prev_tau2 + MAX_TORQUE_RATE)

        # Fix 3 — hard clamp: never more negative than pretension torque
        TAU_MIN = -(T_MIN * R_SPOOL)
        tau1 = max(TAU_MIN, tau1)
        tau2 = max(0.0,     tau2)

        # Save for next rate-limit cycle
        with s.lock:
            s.prev_tau1 = tau1
            s.prev_tau2 = tau2

        if self._enc:
            line = f"{tau1:.3f},{tau2:.3f}\n"
            self._enc.serialInst.write(line.encode())

    def _estop(self):
        self._sending = False
        self.btn_send.setText("▶  send to Teensy")
        with self.state.lock:
            self.state.force_mag = 0.0
            self.state.prev_tau1 = 0.0
            self.state.prev_tau2 = 0.0
        self.fmag_slider.setValue(0)
        if self._enc:
            self._enc.serialInst.write(b"0.000,0.000\n")
        print("[E-STOP] torques zeroed, sending stopped")

    # ──────────────────────────────────────────────────────────────────
    # BACKGROUND THREADS
    # ──────────────────────────────────────────────────────────────────
    def _start_camera(self):
        cam = MainClass(CAM_TOML, TABLE_TOML)

        def loop():
            while self.running:
                try:
                    cam.process_frame()
                    pos = cam.noark_in_table_frame
                    with self.state.lock:
                        if pos is not None:
                            self.state.noark_x   = float(pos[0])
                            self.state.noark_z   = float(pos[2])
                            self.state.has_noark = True
                        else:
                            self.state.has_noark = False
                        self.state.video_frame = cam.video_frame
                except Exception as e:
                    print(f"[cam] {e}")
                    time.sleep(0.05)

        threading.Thread(target=loop, daemon=True).start()

    def _start_teensy(self):
        try:
            self._enc = TeensyPort()
            self._enc.start()
        except Exception as e:
            print(f"[teensy] {e} — encoder display disabled")
            self._enc = None

        def loop():
            while self.running:
                time.sleep(0.02)
                if self._enc:
                    with self.state.lock:
                        self.state.enc1 = self._enc.enc1
                        self.state.enc2 = self._enc.enc2

        threading.Thread(target=loop, daemon=True).start()

    def _start_loadcell(self):
        """Receive load cell data from laptop via UDP WiFi."""
        try:
            self._lc = LoadCellReceiver(port=LC_UDP_PORT)
        except Exception as e:
            print(f"[loadcell] Failed to start receiver: {e}")
            self._lc = None
            return

        def loop():
            while self.running:
                if self._lc:
                    stale = self._lc.is_stale()
                    with self.state.lock:
                        self.state.lc_x    = self._lc.lc_x
                        self.state.lc_y    = self._lc.lc_y
                        self.state.lc_z    = self._lc.lc_z
                        self.state.lc_stale = stale
                time.sleep(0.01)   # 100Hz

        threading.Thread(target=loop, daemon=True).start()

    def _start_send_loop(self):
        """Continuously send force to Teensy at 60Hz when _sending is True."""
        def loop():
            while self.running:
                if self._sending:
                    self._send_force()
                time.sleep(1 / 60)

        threading.Thread(target=loop, daemon=True).start()

    def closeEvent(self, event):
        self.running  = False
        self._sending = False
        self._timer.stop()
        if self._enc:
            self._enc.serialInst.write(b"0.000,0.000\n")
        if self._lc:
            self._lc.stop()
        event.accept()


# ─────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = NOARKWindow()
    win.show()
    sys.exit(app.exec())

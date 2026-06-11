
import sys, os, math, time, threading,serial
import csv
from datetime import datetime
import cv2, numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QFrame, QSizePolicy)
from PySide6.QtGui import (QPainter, QPen, QColor, QBrush, QFont, QPixmap,
    QImage, QPolygonF, QPainterPath, QShortcut, QKeySequence)
from PySide6.QtCore import Qt, QTimer, QPointF, Signal, QRectF
from data_logger import DataLogger
sys.path.insert(0, '/home/sujith/Documents/NOARK_backbone')
from camera_pose_fv import MainClass
from vaideesh.control_implementation_with_GUI.force_validation.teensy_seed_fv import SeeduinoPort, TeensyPort, find_seeed_port

CAM_TOML   = '/home/sujith/Documents/NOARK_backbone/notebooks/calibration/output/good.toml'
TABLE_TOML = '/home/sujith/Documents/NOARK_backbone/estimator/charuco_pose/charuco_pose_picam.toml'
# ── geometry (table-frame metres, x-z plane) ─────────────────────────────
P1 = (-0.495, -0.765)   # left pulley
P2 = (-0.495, -0.681)
P3 = ( 0.495, -0.765)   # right pulley
P4 = (0.495,-0.681)
ML = (-0.065, -0.707)   # left motor
MR = ( 0.065, -0.707)   # right motor
R_SPOOL = 0.033          # motor spool radius (m)
MAX_F   = 24.0           # N
T_MIN = 0.0


HOLD_TIME = 8.0          # seconds to hold for recording
ANGLE_STEPS = 7         # number of angle steps from 0 to 180 (inclusive) for recording

MAG_LIST = [0.0, 5.0, 10.0, 15.0, 24.0]
EDGE_MARGIN_DEG = 5.0
# ── world bounds ──────────────────────────────────────────────────────────
WX0, WX1 = -0.50,  0.50
WZ0, WZ1 = -0.80, 0
C_BG=QColor('#0d0d0d');
C_GRID=QColor('#1e1e1e'); 
C_RAIL=QColor('#3a3a3a')
C_MUTED=QColor('#555'); 
C_BLUE=QColor('#50b4ff'); 
C_GREEN=QColor('#44cc88')
C_AMBER=QColor('#ffbb00'); 
C_RED=QColor('#f05050'); 
C_WHITE=QColor('#ffffff')


# ─────────────────────────────────────────────────────────────────────────
# Kinematics helper
# ─────────────────────────────────────────────────────────────────────────

def solve_tensions(nx, nz, Fx, Fz):
    def unit(dx, dz):
        l = math.hypot(dx, dz)
        return (dx/l, dz/l) if l > 1e-9 else (0.0, 0.0)
    u1x, u1z = unit(P2[0] - nx, P2[1] - nz)
    u3x, u3z = unit(P4[0] - nx, P4[1] - nz)
    det = u1x*u3z - u3x*u1z
    if abs(det) < 1e-6:
        return None
    return {'T1': (Fx*u3z - Fz*u3x)/det, 'T3': (u1x*Fz - u1z*Fx)/det}
class AutoSweep:
    def __init__(self, parent, state, teensy, solve_tensions,
                 P2, P4, R_SPOOL, session_dir="csv_data"):
        self.state = state
        self.teensy = teensy
        self.solve_tensions = solve_tensions
        self.P2 = P2
        self.P4 = P4
        self.R_SPOOL = R_SPOOL
        self.session_dir = session_dir

        self._steps = []
        self._idx = -1
        self._running = False
        self._events = None
        self._events_fh = None

        self._timer = QTimer(parent)          # parent the timer to the window
        self._timer.timeout.connect(self._advance)
      
    def start(self):
        if self._running:
            print("[sweep] already running"); return
        if self.teensy is None:
            print("[sweep] no Teensy — cannot drive motors"); return
        with self.state.lock:
            has_noark = self.state.has_noark
            nx, nz = self.state.noark_x, self.state.noark_z
        if not has_noark:
            print("[sweep] NOARK not detected — cannot compute geometry"); return
        self._steps = self._build_grid(nx, nz)
        if not self._steps:
            print("[sweep] empty grid"); return
        self._open_events_log()
        dur = len(self._steps) * HOLD_TIME
        print(f"[sweep] {len(self._steps)} steps, ~{dur:.0f}s "
              f"({len(MAG_LIST)} mag x {ANGLE_STEPS} angle)")
        self._running = True
        self._idx = -1
        self._advance()
        self._timer.start(int(HOLD_TIME * 1000))

    def stop(self):
        if not self._running:
            return
        self._timer.stop()
        self._running = False
        self._zero_torque()
        if self._events_fh:
            self._events_fh.flush(); self._events_fh.close(); self._events_fh = None
        print("[sweep] stopped, torques zeroed")

    @property
    def running(self):
        return self._running

    def _build_grid(self, nx, nz):
        def unit(dx, dz):
            l = math.hypot(dx, dz)
            return (dx / l, dz / l) if l > 1e-9 else (0.0, 0.0)
        u1 = unit(self.P2[0] - nx, self.P2[1] - nz)
        u3 = unit(self.P4[0] - nx, self.P4[1] - nz)
        a1 = math.atan2(u1[1], u1[0])
        a3 = math.atan2(u3[1], u3[0])
        d = math.atan2(math.sin(a3 - a1), math.cos(a3 - a1))
        margin = math.radians(EDGE_MARGIN_DEG)
        a_start = a1 + math.copysign(margin, d)
        a_end   = a3 - math.copysign(margin, d)
        span = math.atan2(math.sin(a_end - a_start), math.cos(a_end - a_start))
        if ANGLE_STEPS == 1:
            angles = [a_start + span / 2.0]
        else:
            angles = [a_start + span * i / (ANGLE_STEPS - 1)
                      for i in range(ANGLE_STEPS)]
        return [(theta, m) for theta in angles for m in MAG_LIST]

    def _advance(self):
        self._idx += 1
        if self._idx >= len(self._steps):
            print("[sweep] grid complete"); self.stop(); return
        theta, mag = self._steps[self._idx]
        dx, dz = math.cos(theta), math.sin(theta)
        with self.state.lock:
            has_noark = self.state.has_noark
            nx, nz = self.state.noark_x, self.state.noark_z
        if not has_noark:
            print("[sweep] NOARK lost mid-sweep — aborting"); self.stop(); return
        Fx, Fz = mag * dx, mag * dz
        sol = self.solve_tensions(nx, nz, Fx, Fz)
        feasible = sol is not None
        T1 = T3 = tau1 = tau2 = 0.0
        if feasible:
            T1 = max(0.0, sol["T1"]); T3 = max(0.0, sol["T3"])
        with self.state.lock:
            self.state.dir_x = dx; self.state.dir_z = dz
            self.state.force_mag = mag if feasible else 0.0
        if feasible:
            tau1 = -(T1 * self.R_SPOOL); tau2 = T3 * self.R_SPOOL
            self._send_torque(tau1, tau2)
        else:
            self._zero_torque()
        ang_deg = math.degrees(theta)
        print(f"[sweep] {self._idx+1}/{len(self._steps)}  "
              f"ang={ang_deg:6.1f}  mag={mag:5.1f}N  {'OK' if feasible else 'SKIP'}")
        self._log_event(self._idx, ang_deg, mag, T1, T3, tau1, tau2, feasible)

    def _send_torque(self, tau1, tau2):
        try:
            self.teensy.serialInst.write(f"{tau1:.3f},{tau2:.3f}\n".encode())
        except Exception as e:
            print(f"[sweep] torque write failed: {e} — aborting"); self.stop()

    def _zero_torque(self):
        try:
            if self.teensy and self.teensy.serialInst.is_open:
                self.teensy.serialInst.write(b"0.000,0.000\n")
        except Exception:
            pass

    def _open_events_log(self):
        os.makedirs(self.session_dir, exist_ok=True)
        path = os.path.join(self.session_dir, "sweep_targets.csv")
        self._events_fh = open(path, "w", newline="")
        self._events = csv.writer(self._events_fh)
        self._events.writerow(["timestamp", "step", "target_angle_deg",
            "target_mag_N", "T1", "T3", "tau1", "tau2", "feasible"])
        print(f"[sweep] targets → {path}")

    def _log_event(self, step, ang, mag, T1, T3, tau1, tau2, feasible):
        if not self._events:
            return
        self._events.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"), step, f"{ang:.3f}",
            f"{mag:.3f}", f"{T1:.4f}", f"{T3:.4f}", f"{tau1:.4f}",
            f"{tau2:.4f}", int(feasible)])
        self._events_fh.flush()
# ─────────────────────────────────────────────────────────────────────────
# Shared state (thread-safe via lock)
# ─────────────────────────────────────────────────────────────────────────
class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.noark_x = 0.0; self.noark_z = 0.0; self.has_noark = False
        self.enc1 = 0.0; self.enc2 = 0.0
        self.dir_x = 0.0; self.dir_z = 0.0; self.force_mag = 0.0
        self.video_frame = None
        self.lc_x = 0.0; self.lc_y = 0.0; self.lc_z = 0.0
        self.lc_stale = True
        self.lc_offset_x = 0.0; self.lc_offset_y = 0.0
        self.meas_fx = 0.0; self.meas_fz = 0.0; self.meas_mag = 0.0
        self.cached_sol = None

STATE = State()

class SeeduinoReceiver:
    """Reads load cell X/Y force data from Seeduino over serial."""
    def __init__(self, port=None, baud=115200):
        if port is None:
            port = find_seeed_port()
        self.serialInst = serial.Serial()
        self.serialInst.port = port
        self.serialInst.baudrate = baud
        self._lock = threading.Lock()
        self._fx = 0.0
        self._fy = 0.0
        self._timestamp = 0.0
        self._running = True
        self._tare_event = threading.Event()   # same pattern as TeensyPort

    def send_tare(self):
        try:
            if self.serialInst.is_open:
                self._tare_event.clear()
                self.serialInst.write(b'T\n')
                print("[seeeduino] tare command sent")
        except Exception as e:
            print(f"[seeeduino tare] {e}")

    def _parse_line(self, line: str):
        """Parse: '0.123,-0.456'"""
        if line.startswith("TARE DONE"):
            print("[seeeduino] TARE DONE received")
            self._tare_event.set()
            return
        try:
            parts = line.split(",")
            if len(parts) == 2:
                fx, fy = float(parts[0]), float(parts[1])
                with self._lock:
                    self._fx = fx
                    self._fy = fy
                    self._timestamp = time.time()
        except Exception as e:
            print(f"[seeeduino parse] {e} | raw: {line}")

    def _loop(self):
        while self._running:
            try:
                waiting = self.serialInst.in_waiting
                if waiting > 0:
                    line = self.serialInst.readline().decode("utf-8", errors="ignore").strip()
                    if line:
                        self._parse_line(line)
                else:
                    time.sleep(0.0005)
            except Exception as e:
                if self._running:
                    print(f"[seeeduino] {e}")

    @property
    def lc_x(self):
        with self._lock: return self._fx

    @property
    def lc_y(self):
        with self._lock: return self._fy

    @property
    def tare_confirmed(self):
        return self._tare_event.is_set()

    @property
    def lc_z(self):
        return 0.0

    def is_stale(self):
        with self._lock:
            ts = self._timestamp
        return ts == 0.0 or (time.time() - ts) * 1000 > 100

    def start(self):
        try:
            self.serialInst.open()
            print(f"[seeeduino] Connected to {self.serialInst.port}")
            threading.Thread(target=self._loop, daemon=True).start()
        except serial.SerialException as e:
            print(f"[seeeduino] Could not open port: {e}")

    def stop(self):
        self._running = False
        if self.serialInst.is_open:
            self.serialInst.close()

# ─────────────────────────────────────────────────────────────────────────
# Workspace canvas — draws the 2-D table view
# ─────────────────────────────────────────────────────────────────────────
class WorkspaceCanvas(QWidget):
    directionChanged = Signal(float, float) # (dir_x, dir_z)

    def __init__(self, state, parent=None):
        super().__init__(parent); 
        self.state = state
        self.setMinimumHeight(240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

# ── coordinate mapping ──────────────────────────────────────────
    def _w2c(self, wx, wz):
        w, h = self.width(), self.height()
        return QPointF((wx-WX0)/(WX1-WX0)*w, (wz-WZ0)/(WZ1-WZ0)*h)

    def _upd(self, e):
        s = self.state
        with s.lock: 
            np_ = self._w2c(s.noark_x, s.noark_z)
        dx = e.position().x() - np_.x()
        dy = e.position().y() - np_.y()
        l = math.hypot(dx, dy)
        if l < 4:
            return
        self.directionChanged.emit(dx/l, dy/l)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton: self._upd(e)

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton: self._upd(e)

    def paintEvent(self, _):
        s = self.state
        with s.lock:
            nx_w, nz_w = s.noark_x, s.noark_z
            has_noark = s.has_noark
            dx, dz = s.dir_x, s.dir_z
            fm = s.force_mag
            mfx = s.meas_fx; mfz = s.meas_fz; mm = s.meas_mag

        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        # Background
        p.fillRect(0, 0, w, h, C_BG)
        # Grid
        p.setPen(QPen(C_GRID, 1))
        for xv in [-0.4, -0.2, 0, 0.2, 0.4]:
            pt = self._w2c(xv, 0)
            p.drawLine(QPointF(pt.x(), 0), QPointF(pt.x(), h))
        for zv in [-0.8, -0.6, -0.4, -0.2]:
            pt = self._w2c(0, zv)
            p.drawLine(QPointF(0, pt.y()), QPointF(w, pt.y()))

          # Top rail
        r0 = self._w2c(-0.6, -0.773)
        r1 = self._w2c( 0.6, -0.773)
        p.setPen(QPen(C_RAIL, 2))
        p.drawLine(r0,r1)

        # p.drawLine(self._w2c(-0.55, -0.773), self._w2c(0.55, -0.773))
        # Solve tensions
        Fx = fm*dx; Fz = fm*dz
        sol = solve_tensions(nx_w, nz_w, Fx, Fz)
        T1 = max(0.0, sol['T1']) if sol else 0.0
        T3 = max(0.0, sol['T3']) if sol else 0.0

        np_ = self._w2c(nx_w, nz_w)
        ml  = self._w2c(*ML)
        mr  = self._w2c(*MR)
        p1  = self._w2c(*P1)
        p3  = self._w2c(*P3)
        p2  = self._w2c(*P2)
        p4  = self._w2c(*P4)

        def cable(pts, col, t):
            alpha = int(64 + min(t, MAX_F)/MAX_F*191)
            c = QColor(col); c.setAlpha(alpha)
            pen = QPen(c, 1.0 + t/MAX_F*3)
            pen.setCapStyle(Qt.RoundCap); pen.setJoinStyle(Qt.RoundJoin)
            p.setPen(pen); path = QPainterPath(); path.moveTo(pts[0])
            for pt in pts[1:]: path.lineTo(pt)
            p.drawPath(path)

        cable([ml, p1,p2, np_], C_BLUE, T1)
        cable([mr, p3,p4, np_], C_GREEN, T3)
        f9 = QFont('Courier New', 8); p.setFont(f9)
        for pt, lbl in [(ml, 'ML'), (mr, 'MR')]:
            p.setBrush(QBrush(QColor('#332200'))); p.setPen(QPen(C_AMBER, 1.5))
            p.drawRect(QRectF(pt.x()-7, pt.y()-7, 14, 14)); p.setPen(C_AMBER)
        for pt, lbl in [(p1, 'P1'), (p3, 'P3')]:
            p.setBrush(QBrush(C_BG))
            p.setPen(QPen(C_RAIL, 1.2))
            p.drawEllipse(QPointF(pt.x(), pt.y()), 5, 5)
            p.setPen(C_MUTED)
            p.drawText(QPointF(pt.x()-6, pt.y()-10), lbl)
        for pt, lbl, col in [(p2, 'P2', C_BLUE), (p4, 'P4', C_GREEN)]:
            p.setBrush(QBrush(C_BG))
            p.setPen(QPen(col, 1.5))
            p.drawEllipse(QPointF(pt.x(), pt.y()), 7, 7)
            p.setPen(col)
            p.drawText(QPointF(pt.x()-6, pt.y()-10), lbl)
        ncol = C_RED if has_noark else QColor('#444')
        p.setBrush(QBrush(QColor('#220d0d') if has_noark else QColor('#1a1a1a')))
        p.setPen(QPen(ncol, 2))
        p.drawEllipse(QPointF(np_.x(), np_.y()), 10, 10)
        p.setPen(C_WHITE); p.setFont(QFont('Courier New', 8, QFont.Bold))
        p.drawText(QRectF(np_.x()-8, np_.y()-6, 16, 12), Qt.AlignCenter, 'N')
        p.setPen(ncol); p.setFont(f9)
        p.drawText(QPointF(np_.x()-30, np_.y()-14), f'({nx_w:.3f}, {nz_w:.3f})')
        if fm > 0.1 and has_noark:
            ln = 35 + fm/MAX_F*45; ax = np_.x()+dx*ln; ay = np_.y()+dz*ln
            p.setPen(QPen(C_RED, 2))
            p.drawLine(QPointF(np_.x(), np_.y()), QPointF(ax, ay))
            self._head(p, np_.x(), np_.y(), ax, ay, C_RED); p.setPen(C_RED)
            p.drawText(QPointF(ax+6, ay-4), f'{fm:.1f}N cmd')
        if mm > 0.1 and has_noark:
            mdx = mfx/mm; mdz = mfz/mm; ln = 35 + mm/MAX_F*45
            mx = np_.x()+mdx*ln; my = np_.y()+mdz*ln
            p.setPen(QPen(C_GREEN, 2))
            p.drawLine(QPointF(np_.x(), np_.y()), QPointF(mx, my))
            self._head(p, np_.x(), np_.y(), mx, my, C_GREEN); p.setPen(C_GREEN)
            p.drawText(QPointF(mx+6, my+10), f'{mm:.1f}N meas')
        p.end()

    @staticmethod
    def _head(painter, x0, y0, x1, y1, color):
        a = math.atan2(y1-y0, x1-x0); sz = 10; s = math.radians(25)
        pts = QPolygonF([QPointF(x1, y1),
            QPointF(x1 - sz*math.cos(a-s), y1 - sz*math.sin(a-s)),
            QPointF(x1 - sz*math.cos(a+s), y1 - sz*math.sin(a+s))])
        painter.setBrush(QBrush(color)); painter.setPen(Qt.NoPen)
        painter.drawPolygon(pts)


class CameraCanvas(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent); self.state = state
        self.setMinimumHeight(200)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pixmap = None

    def update_frame(self):
        frame = self.state.video_frame
        if frame is None: return
        fh, fw = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = QImage(rgb.data, fw, fh, fw*3, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(img); self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(0, 0, self.width(), self.height(), C_BG)
        if self._pixmap:
            s = self._pixmap.scaled(self.width(), self.height(),
                                    Qt.KeepAspectRatio, Qt.SmoothTransformation)
            p.drawPixmap((self.width()-s.width())//2,
                         (self.height()-s.height())//2, s)
        else:
            p.setPen(C_MUTED); p.setFont(QFont('Courier New', 10))
            p.drawText(self.rect(), Qt.AlignCenter, 'no camera feed')
        p.end()


def _sec(t):
    l = QLabel(t.upper()); l.setFont(QFont('Courier New', 8))
    l.setStyleSheet('color:#444;padding:6px 10px 2px 10px;'); return l

def _sep():
    f = QFrame(); f.setFrameShape(QFrame.HLine)
    f.setStyleSheet('color:#2a2a2a;margin:0 10px;'); return f

def _row(layout, label, color='#ccc'):
    w = QWidget(); w.setStyleSheet('background:transparent;')
    h = QHBoxLayout(w); h.setContentsMargins(10, 2, 10, 2)
    ll = QLabel(label); ll.setFont(QFont('Courier New', 12))
    ll.setStyleSheet('color:#555;')
    vl = QLabel('--'); vl.setFont(QFont('Courier New', 11, QFont.Bold))
    vl.setStyleSheet(f'color:{color};'); vl.setAlignment(Qt.AlignRight)
    h.addWidget(ll); h.addWidget(vl); layout.addWidget(w); return vl

def _btn(text, fg, bg, border):
    b = QPushButton(text); b.setFont(QFont('Courier New', 10))
    b.setFixedHeight(38); b.setCursor(Qt.PointingHandCursor)
    b.setStyleSheet(
        f'QPushButton{{background:{bg};color:{fg};border:1px solid {border};'
        f'border-radius:2px;padding:0 8px;}}'
        f'QPushButton:hover{{background:{border}22;}}'
        f'QPushButton:pressed{{background:{border}44;}}')
    return b

def _wrap(layout, widget):
    w = QWidget(); w.setStyleSheet('background:transparent;')
    v = QVBoxLayout(w); v.setContentsMargins(10, 2, 10, 2); v.addWidget(widget)
    layout.addWidget(w)


class NOARKWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('NOARK Force Control')
        self.resize(960, 660)
        self.setStyleSheet('QMainWindow,QWidget{background:#111;}')
        self.state = STATE; self.running = True
        self._enc = None; self._lc = None; self._sending = False
        self._threads = []  # FIX 7 - track threads for join on close
        self._logger = DataLogger(base_dir="csv_data")
        self._rec_state = "idle"
        self._arm_t0 = 0.0

        self._arm_timer = QTimer(self)
        self._arm_timer.timeout.connect(self._check_arm)

        QShortcut(QKeySequence("S"), self, activated=self._begin_arming)
        QShortcut(QKeySequence("X"), self, activated=self._stop_recording)
       

        self._build_ui()
        self._start_camera()
        self._start_teensy()
        self._sweep = AutoSweep(self, self.state, self._enc, solve_tensions,
                        P2, P4, R_SPOOL, session_dir=self._logger.session_dir)
        self._start_loadcell()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(33)       # ~30 Hz display — leaves main thread time for log timer

        self._log_timer = QTimer(self)
        self._log_timer.timeout.connect(self._log_tick)
        self._log_timer.start(5)    # 200 Hz target, independent of paint

        self._cam_timer = QTimer(self)
        self._cam_timer.timeout.connect(self.cam_view.update_frame)
        self._cam_timer.start(33)   # 30 Hz camera display
        # No continuous send timer — button press sends once

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8); root.setSpacing(8)
        left = QWidget(); lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0); lv.setSpacing(6)
        self.workspace = WorkspaceCanvas(self.state)
        self.workspace.directionChanged.connect(self._on_direction)
        self.workspace.setStyleSheet('border:1px solid #2a2a2a;border-radius:2px;')
        lv.addWidget(self.workspace, 3)
        self.cam_view = CameraCanvas(self.state)
        self.cam_view.setStyleSheet('border:1px solid #2a2a2a;border-radius:2px;')
        lv.addWidget(self.cam_view, 2); root.addWidget(left, 3)
        sb = QWidget(); sb.setFixedWidth(230)
        sb.setStyleSheet('background:#141414;border-radius:2px;')
        sv = QVBoxLayout(sb); sv.setContentsMargins(0, 4, 0, 8); sv.setSpacing(0)
        sv.addWidget(_sec('NOARK position (m)')); sv.addWidget(_sep())
        self.v_nx = _row(sv, 'x', '#50b4ff'); self.v_nz = _row(sv, 'z', '#50b4ff')
        cw = QWidget(); cw.setStyleSheet('background:transparent;')
        ch = QHBoxLayout(cw); ch.setContentsMargins(10, 2, 10, 2)
        lc = QLabel('camera'); lc.setFont(QFont('Courier New', 12))
        lc.setStyleSheet('color:#555;')
        self.v_cam = QLabel('searching')
        self.v_cam.setFont(QFont('Courier New', 12))
        self.v_cam.setStyleSheet('color:#f05050;')
        self.v_cam.setAlignment(Qt.AlignRight)
        ch.addWidget(lc); ch.addWidget(self.v_cam); sv.addWidget(cw)
        sv.addWidget(_sec('Encoders (deg)')); sv.addWidget(_sep())
        self.v_e1 = _row(sv, 'enc 1 (left)', '#ffbb00')
        self.v_e2 = _row(sv, 'enc 2 (right)', '#ffbb00')
        btn_reset = _btn('reset encoders', '#ffbb00', '#1a1400', '#ffbb00')
        btn_reset.clicked.connect(self._reset_encoders); _wrap(sv, btn_reset)
        sv.addWidget(_sec('Force')); sv.addWidget(_sep())
        self.v_fmag = _row(sv, 'magnitude', '#f05050')
        self.v_fdir = _row(sv, 'direction', '#888')
        slw = QWidget(); slw.setStyleSheet('background:transparent;')
        sll = QVBoxLayout(slw); sll.setContentsMargins(10, 2, 10, 2)
        self.fmag_slider = QSlider(Qt.Horizontal)
        self.fmag_slider.setRange(0, int(MAX_F*10))
        self.fmag_slider.setValue(0)
        self.fmag_slider.setStyleSheet(
            'QSlider::groove:horizontal{background:#2a2a2a;height:4px;border-radius:2px;}'
            'QSlider::handle:horizontal{background:#f05050;width:14px;height:14px;'
            'margin:-5px 0;border-radius:7px;}'
            'QSlider::sub-page:horizontal{background:#f05050;border-radius:2px;}')
        self.fmag_slider.valueChanged.connect(self._on_force_slider)
        sll.addWidget(self.fmag_slider); sv.addWidget(slw)
        sv.addWidget(_sec('Cable tensions')); sv.addWidget(_sep())
        self.v_t1 = _row(sv, 'T1 left', '#50b4ff')
        self.v_t3 = _row(sv, 'T3 right', '#44cc88')
        sv.addWidget(_sec('Motor torques (Nm)')); sv.addWidget(_sep())
        self.v_tau1 = _row(sv, 'tau1', '#50b4ff')
        self.v_tau2 = _row(sv, 'tau2', '#44cc88')
        sv.addWidget(_sec('Commanded force')); sv.addWidget(_sep())
        self.v_cmd_mag = _row(sv, 'magnitude', '#f05050')
        self.v_cmd_dir = _row(sv, 'direction', '#f05050')
        sv.addWidget(_sec('Measured force')); sv.addWidget(_sep())
        self.v_lc_status = _row(sv, 'status', '#888')
        self.v_lc_mag = _row(sv, 'magnitude', '#44cc88')
        self.v_lc_dir = _row(sv, 'direction', '#44cc88')
        btn_tare = _btn('tare load cell', '#44cc88', '#0a1a10', '#44cc88')
        btn_tare.clicked.connect(self._tare_loadcell); _wrap(sv, btn_tare)
        sv.addWidget(_sec('Error')); sv.addWidget(_sep())
        self.v_err_mag = _row(sv, 'mag error', '#ffbb00')
        self.v_err_dir = _row(sv, 'dir error', '#ffbb00')
        self.v_err_mag_per = _row(sv,'mag_error_pcnt', '#ffbb00')
        self.v_err_dir_per = _row(sv,'dir_error_pcnt', '#ffbb00')   
        sv.addStretch()
        self.btn_send = _btn('send to Teensy', '#50b4ff', '#0a1020', '#50b4ff')
        self.btn_send.clicked.connect(self._toggle_send); _wrap(sv, self.btn_send)
        btn_estop = _btn('E-STOP', '#f05050', '#1a0808', '#f05050')
        btn_estop.setFont(QFont('Courier New', 12, QFont.Bold))
        btn_estop.clicked.connect(self._estop); _wrap(sv, btn_estop)
        root.addWidget(sb, 0)

    def _refresh(self):
        s = self.state
        with s.lock:
            has_noark = s.has_noark; nx, nz = s.noark_x, s.noark_z
            e1, e2 = s.enc1, s.enc2
            dir_x, dir_z = s.dir_x, s.dir_z
            fm = s.force_mag; lc_stale = s.lc_stale
            lc_x, lc_y = s.lc_x, s.lc_y
            ox, oy = s.lc_offset_x, s.lc_offset_y
        if has_noark:
            self.v_nx.setText(f'{nx:.4f}'); self.v_nz.setText(f'{nz:.4f}')
            self.v_cam.setText('tracking')
            self.v_cam.setStyleSheet('color:#44cc88;')
        else:
            self.v_nx.setText('--'); self.v_nz.setText('--')
            self.v_cam.setText('searching')
            self.v_cam.setStyleSheet('color:#f05050;')
        self.v_e1.setText(f'{e1:.2f}'); self.v_e2.setText(f'{e2:.2f}')
        cmd_fx = fm*dir_x; cmd_fz = fm*dir_z
        cmd_dir = math.degrees(math.atan2(cmd_fz, cmd_fx))
        self.v_fmag.setText(f'{fm:.1f} N')
        self.v_fdir.setText(f'{cmd_dir:.1f} deg')
        self.v_cmd_mag.setText(f'{fm:.2f} N')
        self.v_cmd_dir.setText(f'{cmd_dir:.1f} deg')
        # FIX 4 - solve once; cache result for _send_force to reuse
        sol = solve_tensions(nx, nz, cmd_fx, cmd_fz)
        with s.lock:
            s.cached_sol = sol
        if sol and has_noark:
            T1 = max(T_MIN, sol['T1']); T3 = max(T_MIN, sol['T3'])
            tau1 = -(T1*R_SPOOL); tau2 = T3*R_SPOOL
            self.v_t1.setText(f'{T1:.2f} N'); self.v_t3.setText(f'{T3:.2f} N')
            self.v_tau1.setText(f'{tau1:.3f}'); self.v_tau2.setText(f'{tau2:.3f}')
        else:
            for w in (self.v_t1, self.v_t3, self.v_tau1, self.v_tau2):
                w.setText('--')
        mfx = lc_x; mfz = lc_y 
        mm = math.hypot(mfx, mfz)
        mdir = math.degrees(math.atan2(mfz, mfx))
        with s.lock:
            s.meas_fx = mfx; s.meas_fz = mfz; s.meas_mag = mm
        if lc_stale:
            self.v_lc_status.setText('no data')
            self.v_lc_status.setStyleSheet('color:#f05050;')
            for w in (self.v_lc_mag, self.v_lc_dir, self.v_err_mag, self.v_err_dir):
                w.setText('--')
        else:
            self.v_lc_status.setText('live')
            self.v_lc_status.setStyleSheet('color:#44cc88;')
            self.v_lc_mag.setText(f'{mm:.2f} N')
            self.v_lc_dir.setText(f'{mdir:.1f} deg')
            # err_mag = abs((mm - fm) / fm) if fm > 1e-6 else 0.0
            err_mag = abs(mm - fm)
            err_dir = abs(cmd_dir - mdir)
            err_mag_pcnt = (err_mag /fm)* 100 if fm > 1e-6 else 0.0
            if err_dir > 180:
                err_dir -= 360
            elif err_dir < -180:
                err_dir += 360
            err_dir_pcnt = (err_dir / 360.0) * 100
            self.v_err_mag.setText(f'{err_mag:.2f} N')
            self.v_err_dir.setText(f'{err_dir:.1f} deg')
            self.v_err_mag_per.setText(f'{err_mag_pcnt:.1f} %')
            self.v_err_dir_per.setText(f'{err_dir_pcnt:.1f} %') 
            mc = '#44cc88' if err_mag < 1 else '#ffbb00' if err_mag < 3 else '#f05050'
            dc = '#44cc88' if err_dir < 5 else '#ffbb00' if err_dir < 15 else '#f05050'
            self.v_err_mag.setStyleSheet(f'color:{mc};')
            self.v_err_dir.setStyleSheet(f'color:{dc};')
        self.workspace.update()
        # self.cam_view.update_frame()

    def _log_tick(self):
        s = self.state
        with s.lock:
            has_noark = s.has_noark
            nx, nz = s.noark_x, s.noark_z
            dir_x, dir_z = s.dir_x, s.dir_z
            fm = s.force_mag
        sol = solve_tensions(nx, nz, fm * dir_x, fm * dir_z)
        if sol and has_noark:
            T1 = max(T_MIN, sol['T1']); T3 = max(T_MIN, sol['T3'])
            tau1 = -(T1 * R_SPOOL); tau2 = T3 * R_SPOOL
        else:
            T1 = T3 = tau1 = tau2 = 0.0
        self._logger.log_gui(fm, math.degrees(math.atan2(fm * dir_z, fm * dir_x)),
                             T1, T3, tau1, tau2)

    def _on_direction(self, dx, dz):
        with self.state.lock:
            self.state.dir_x = dx; self.state.dir_z = dz

    def _on_force_slider(self, val):
        with self.state.lock:
            self.state.force_mag = val / 10.0

    def _reset_encoders(self):
        if self._enc: self._enc.encoder_reset()

    def _tare_loadcell(self):
        # Reset Pi-side offsets to zero first
        with self.state.lock:
            self.state.lc_offset_x = 0.0
            self.state.lc_offset_y = 0.0
        # Send tare command to Seeduino firmware
        if self._lc:
            self._lc.send_tare()
    def _begin_arming(self):
        if self._rec_state != "idle":
            print(f"[arm] ignored — already {self._rec_state}")
            return
        if not self._enc or not self._lc:
            print("[arm] devices not connected — cannot arm")
            return

        print("[arm] taring encoders + load cell…")
        # zero Pi-side load-cell offsets too
        with self.state.lock:
            self.state.lc_offset_x = 0.0
            self.state.lc_offset_y = 0.0

        self._enc.encoder_reset()   # sends 'R\n', resets its confirm flag
        self._lc.send_tare()        # sends 'T\n', resets its confirm flag

        self._rec_state = "arming"
        self._arm_t0 = time.time()
        self._arm_timer.start(10)   # poll at 100 Hz

    def _check_arm(self):
        if self._rec_state != "arming":
            self._arm_timer.stop()
            return

        enc_ok = bool(getattr(self._enc, "tare_confirmed", False))
        lc_ok  = self._lc is not None and self._lc.tare_confirmed

        if enc_ok and lc_ok:
            self._arm_timer.stop()
            self._logger.start()
            self._rec_state = "recording"
            print("[arm] both confirmed → RECORDING, starting sweep")
            self._sweep.start()
        elif time.time() - self._arm_t0 > 5.0:
            self._arm_timer.stop()
            self._rec_state = "idle"
            print(f"[arm] TIMEOUT — enc_ok={enc_ok} lc_ok={lc_ok}. Not recording.")

    def _stop_recording(self):
        if self._rec_state == "recording":
            self._logger._active = False   # stop logging, keep files open
            self._rec_state = "idle"
            print("[arm] recording stopped")

    def _toggle_send(self):
        s = self.state
        with s.lock:
            has_noark = s.has_noark
            sol = s.cached_sol
            nx, nz = s.noark_x, s.noark_z
            dir_x, dir_z = s.dir_x, s.dir_z
            fm = s.force_mag
            lc_x = s.lc_x; lc_y = s.lc_y
            ox = s.lc_offset_x; oy = s.lc_offset_y

        if not has_noark or not sol:
            print("[send] No NOARK detected — not sending")
            return
        if abs(dir_x) < 1e-6 and abs(dir_z) < 1e-6:
            print("[send] No direction set — click on canvas first")
            return

        T1 = max(T_MIN, sol['T1']); T3 = max(T_MIN, sol['T3'])
        tau1 = -(T1 * R_SPOOL); tau2 = T3 * R_SPOOL
        Fx = fm * dir_x; Fz = fm * dir_z

        if self._enc:
            cmd = f'{tau1:.3f},{tau2:.3f}\n'
            self._enc.serialInst.write(cmd.encode())
            print(f'[send] sent: {cmd.strip()}')
        else:
            print('[send] no Teensy connected')
        # Measured force from load cell (subtract tare offset)
        mfx = lc_x
        mfy = lc_y
        lc_mag = math.hypot(mfx, mfy)

    def _estop(self):
        if getattr(self, "_sweep", None):
            self._sweep.stop()
        self._sending = False
        self.btn_send.setText('send to Teensy')
        with self.state.lock:
            self.state.force_mag = 0.0
        self.fmag_slider.setValue(0)
        if self._enc:
            # FIX 1 - \n is a proper escape sequence
            self._enc.serialInst.write(b'0.000,0.000\n')
        print('[E-STOP] torques zeroed')

    def _start_camera(self):
        cam = MainClass(CAM_TOML, TABLE_TOML)
        def loop():
            while self.running:
                try:
                    cam.process_frame()
                    pos = cam.noark_in_table_frame
                    frame = cam.video_frame
                    with self.state.lock:
                        self.state.has_noark = pos is not None
                        if pos is not None:
                            self.state.noark_x = float(pos[0])
                            self.state.noark_z = float(pos[2])
                            self._logger.log_camera(float(pos[0]), float(pos[2]))
                        self.state.video_frame = frame
                        
                except Exception as e:
                    print(f'[cam] {e}')
                    time.sleep(0.05)
        t = threading.Thread(target=loop, daemon=True)
        t.start(); self._threads.append(t)

    def _start_teensy(self):
        try:
            self._enc = TeensyPort()
            self._enc.start()
            self._enc.on_update = lambda e1, e2: self._logger.log_encoder(e1, e2)
            # Auto-tare after connection settles
            time.sleep(2.0)   # wait for serial to stabilise
            self._enc.encoder_reset()
            print("[teensy] auto-tared on startup")
        except Exception as e:
            print(f'[teensy] {e}'); self._enc = None
        def loop():
            while self.running:
                time.sleep(0.001)
                if self._enc:
                    with self.state.lock:
                        self.state.enc1 = self._enc.enc1
                        self.state.enc2 = self._enc.enc2
        t = threading.Thread(target=loop, daemon=True)
        t.start(); self._threads.append(t)

    def _start_loadcell(self):
        try:
            self._lc = SeeduinoReceiver()
            self._lc.start()
            original_parse = self._lc._parse_line
            def _patched_parse(line):
                original_parse(line)
                self._logger.log_loadcell(self._lc.lc_x, self._lc.lc_y)
            self._lc._parse_line = _patched_parse
        except Exception as e:
            print(f'[loadcell] {e}'); self._lc = None; return
        def loop():
            while self.running:
                if self._lc:
                    with self.state.lock:
                        self.state.lc_x = self._lc.lc_x
                        self.state.lc_y = self._lc.lc_y
                        self.state.lc_z = self._lc.lc_z
                        self.state.lc_stale = self._lc.is_stale()
                time.sleep(0.001)
        t = threading.Thread(target=loop, daemon=True)
        t.start(); self._threads.append(t)

    def closeEvent(self, event):
        self.running = False
        self._timer.stop()
        self._log_timer.stop()
        if self._enc:
            try: self._enc.serialInst.write(b'0.000,0.000\n')  # FIX 1
            except Exception: pass
        if self._lc:
            self._lc.stop()
        for t in self._threads:
            t.join(timeout=1.0)
        self._logger.stop()
        if getattr(self, "_sweep", None):
            self._sweep.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv); app.setStyle('Fusion')
    win = NOARKWindow(); win.show(); sys.exit(app.exec())
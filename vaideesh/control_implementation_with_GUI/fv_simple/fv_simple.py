
import sys, os, math, time, threading, serial, csv, queue
from datetime import datetime
import cv2, numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QFrame,
    QSizePolicy, QInputDialog)
from PySide6.QtGui import (QPainter, QPen, QColor, QBrush, QFont,
    QPolygonF, QPainterPath, QShortcut, QKeySequence)
from PySide6.QtCore import Qt, QTimer, QPointF, Signal, QRectF

sys.path.insert(0, '/home/sujith/Documents/NOARK_backbone')
sys.path.insert(0, '/home/sujith/Documents/NOARK_backbone/vaideesh/control_implementation_with_GUI/force_validation')
from camera_pose_fv import MainClass
from teensy_seed_fv import SeeduinoPort, TeensyPort, find_seeed_port

CAM_TOML   = '/home/sujith/Documents/NOARK_backbone/notebooks/calibration/output/good.toml'
TABLE_TOML = '/home/sujith/Documents/NOARK_backbone/estimator/charuco_pose/charuco_pose_picam.toml'

P2 = (-0.495, -0.681)
P4 = ( 0.495, -0.681)
ML = (-0.065, -0.707)
MR = ( 0.065, -0.707)
R_SPOOL     = 0.033
MAX_F       = 24.0
T_MIN       = 0.0
HOLD_TIME   = 2.0
ANGLE_STEPS = 7
MAG_LIST    = [5.0, 10.0, 15.0, 24.0]
EDGE_MARGIN_DEG = 5.0
WX0, WX1 = -0.50,  0.50
WZ0, WZ1 = -0.80,  0.0

C_BG    = QColor('#0d0d0d')
C_GRID  = QColor('#1e1e1e')
C_RAIL  = QColor('#3a3a3a')
C_MUTED = QColor('#555')
C_BLUE  = QColor('#50b4ff')
C_GREEN = QColor('#44cc88')
C_AMBER = QColor('#ffbb00')
C_RED   = QColor('#f05050')
C_WHITE = QColor('#ffffff')


# ── minimal non-blocking CSV writer ──────────────────────────────────────────
class _StreamWriter:
    def __init__(self, filepath, header):
        self._q  = queue.Queue()
        self._fh = open(filepath, 'w', newline='', buffering=131072)
        self._w  = csv.writer(self._fh)
        self._w.writerow(header)
        self._running = True
        threading.Thread(target=self._drain, daemon=False).start()

    def log(self, row):
        self._q.put(row)

    def _drain(self):
        last_flush = time.monotonic()
        while self._running or not self._q.empty():
            try:
                self._w.writerow(self._q.get(timeout=0.05))
                now = time.monotonic()
                if now - last_flush >= 0.1:
                    self._fh.flush(); last_flush = now
            except queue.Empty:
                pass

    def stop(self):
        self._running = False
        # drain thread joins itself; flush and close after
        while not self._q.empty():
            time.sleep(0.01)
        self._fh.flush(); self._fh.close()


class DataLogger:
    """Two-stream logger: loadcell + gui only."""
    def __init__(self, base_dir='csv_data', session_name='session'):
        d = os.path.join(base_dir, session_name)
        os.makedirs(d, exist_ok=True)
        self.session_dir = d
        self._lc  = _StreamWriter(os.path.join(d, 'loadcell.csv'),
                                  ['timestamp', 'Fx', 'Fy'])
        self._gui = _StreamWriter(os.path.join(d, 'gui.csv'),
                                  ['timestamp', 'magnitude', 'direction',
                                   'Fx', 'Fz', 'T1_left', 'T3_right', 'tau1', 'tau2'])
        self._active = False
        print(f'[logger] session → {d}/')

    @staticmethod
    def _ts():
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

    def start(self):
        self._active = True
        print('[logger] recording started')

    def stop(self):
        self._active = False
        self._lc.stop(); self._gui.stop()
        print('[logger] stopped, files closed')

    def log_loadcell(self, fx, fy):
        if not self._active: return
        self._lc.log([self._ts(), f'{fx:.6f}', f'{fy:.6f}'])

    def log_gui(self, magnitude, direction, Fx, Fz, T1, T3, tau1, tau2):
        if not self._active: return
        self._gui.log([self._ts(),
                       f'{magnitude:.6f}', f'{direction:.6f}',
                       f'{Fx:.6f}', f'{Fz:.6f}',
                       f'{T1:.6f}', f'{T3:.6f}',
                       f'{tau1:.6f}', f'{tau2:.6f}'])


# ── kinematics ────────────────────────────────────────────────────────────────
def solve_tensions(nx, nz, Fx, Fz):
    def unit(dx, dz):
        l = math.hypot(dx, dz)
        return (dx/l, dz/l) if l > 1e-9 else (0.0, 0.0)
    u1x, u1z = unit(P2[0]-nx, P2[1]-nz)
    u3x, u3z = unit(P4[0]-nx, P4[1]-nz)
    det = u1x*u3z - u3x*u1z
    if abs(det) < 1e-6: return None
    return {'T1': (Fx*u3z - Fz*u3x)/det,
            'T3': (u1x*Fz - u1z*Fx)/det}


# ── shared state ──────────────────────────────────────────────────────────────
class State:
    def __init__(self):
        self.lock       = threading.Lock()
        self.noark_x    = 0.0;  self.noark_z  = 0.0
        self.has_noark  = False
        self.dir_x      = 0.0;  self.dir_z    = 0.0
        self.force_mag  = 0.0
        self.lc_x       = 0.0;  self.lc_y     = 0.0
        self.lc_stale   = True
        self.meas_fx    = 0.0;  self.meas_fz  = 0.0;  self.meas_mag = 0.0
        self.cached_sol = None
        self.cam_status = 'searching'   # 'searching' | 'captured' | 'timeout'

STATE = State()


# ── auto sweep ────────────────────────────────────────────────────────────────
class AutoSweep:
    def __init__(self, parent, state, teensy, session_dir='csv_data'):
        self.state       = state
        self.teensy      = teensy
        self.session_dir = session_dir
        self._steps      = []
        self._idx        = -1
        self._running    = False
        self._events     = None
        self._events_fh  = None
        self._sweep_t0   = 0.0
        self._timer      = QTimer(parent)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._advance)

    def start(self):
        if self._running: return
        if self.teensy is None:
            print('[sweep] no Teensy'); return
        with self.state.lock:
            has_noark = self.state.has_noark
            nx, nz   = self.state.noark_x, self.state.noark_z
        if not has_noark:
            print('[sweep] position not captured'); return
        self._steps = self._build_grid(nx, nz)
        if not self._steps: return
        self._open_events_log()
        print(f'[sweep] {len(self._steps)} steps, ~{len(self._steps)*HOLD_TIME:.0f}s')
        self._running  = True
        self._idx      = -1
        self._sweep_t0 = time.perf_counter()
        self._advance()

    def stop(self):
        if not self._running: return
        self._timer.stop(); self._running = False
        self._zero_torque()
        if self._events_fh:
            self._events_fh.flush(); self._events_fh.close(); self._events_fh = None
        print('[sweep] stopped')

    @property
    def running(self): return self._running

    def _build_grid(self, nx, nz):
        def unit(dx, dz):
            l = math.hypot(dx, dz)
            return (dx/l, dz/l) if l > 1e-9 else (0.0, 0.0)
        u1  = unit(P2[0]-nx, P2[1]-nz)
        u3  = unit(P4[0]-nx, P4[1]-nz)
        a1  = math.atan2(u1[1], u1[0])
        a3  = math.atan2(u3[1], u3[0])
        d   = math.atan2(math.sin(a3-a1), math.cos(a3-a1))
        m   = math.radians(EDGE_MARGIN_DEG)
        a_s = a1 + math.copysign(m, d)
        a_e = a3 - math.copysign(m, d)
        sp  = math.atan2(math.sin(a_e-a_s), math.cos(a_e-a_s))
        angles = ([a_s + sp/2] if ANGLE_STEPS == 1
                  else [a_s + sp*i/(ANGLE_STEPS-1) for i in range(ANGLE_STEPS)])
        result = []
        for i, theta in enumerate(angles):
            mags = MAG_LIST if i % 2 == 0 else list(reversed(MAG_LIST))
            for m in mags:
                result.append((theta, m))
        return result

    def _arm_timer(self):
        next_deadline = self._sweep_t0 + (self._idx + 1) * HOLD_TIME
        delay_ms = max(1, int((next_deadline - time.perf_counter()) * 1000))
        self._timer.start(delay_ms)

    def _advance(self):
        self._idx += 1
        if self._idx >= len(self._steps):
            print('[sweep] complete'); self.stop(); return
        theta, mag = self._steps[self._idx]
        dx, dz = math.cos(theta), math.sin(theta)
        with self.state.lock:
            nx, nz = self.state.noark_x, self.state.noark_z
        Fx, Fz = mag*dx, mag*dz
        sol = solve_tensions(nx, nz, Fx, Fz)
        T1 = T3 = tau1 = tau2 = 0.0
        if sol:
            T1 = max(T_MIN, sol['T1']); T3 = max(T_MIN, sol['T3'])
            tau1 = -(T1*R_SPOOL);      tau2 = T3*R_SPOOL
            self._send_torque(tau1, tau2)
        else:
            self._zero_torque()
        with self.state.lock:
            self.state.dir_x = dx; self.state.dir_z = dz
            self.state.force_mag = mag if sol else 0.0
        ang_deg = math.degrees(theta)
        print(f'[sweep] {self._idx+1}/{len(self._steps)}  ang={ang_deg:.1f}  mag={mag:.1f}N')
        self._log_event(self._idx, ang_deg, mag, T1, T3, tau1, tau2, sol is not None)
        self._arm_timer()

    def _send_torque(self, tau1, tau2):
        try:
            self.teensy.serialInst.write(f'{tau1:.3f},{tau2:.3f}\n'.encode())
        except Exception as e:
            print(f'[sweep] write failed: {e}'); self.stop()

    def _zero_torque(self):
        try:
            if self.teensy and self.teensy.serialInst.is_open:
                self.teensy.serialInst.write(b'0.000,0.000\n')
        except Exception: pass

    def _open_events_log(self):
        os.makedirs(self.session_dir, exist_ok=True)
        path = os.path.join(self.session_dir, 'sweep_targets.csv')
        self._events_fh = open(path, 'w', newline='')
        self._events    = csv.writer(self._events_fh)
        self._events.writerow(['timestamp','step','target_angle_deg',
                                'target_mag_N','T1','T3','tau1','tau2','feasible'])

    def _log_event(self, step, ang, mag, T1, T3, tau1, tau2, feasible):
        if not self._events: return
        self._events.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                                step, f'{ang:.3f}', f'{mag:.3f}',
                                f'{T1:.4f}', f'{T3:.4f}',
                                f'{tau1:.4f}', f'{tau2:.4f}', int(feasible)])
        self._events_fh.flush()


# ── load cell serial reader ───────────────────────────────────────────────────
class SeeduinoReceiver:
    def __init__(self, port=None, baud=115200):
        if port is None: port = find_seeed_port()
        self.serialInst = serial.Serial()
        self.serialInst.port = port
        self.serialInst.baudrate = baud
        self._lock        = threading.Lock()
        self._fx = self._fy = self._timestamp = 0.0
        self._running     = True
        self._tare_event  = threading.Event()

    def send_tare(self):
        try:
            if self.serialInst.is_open:
                self._tare_event.clear()
                self.serialInst.write(b'T\n')
        except Exception as e:
            print(f'[seeeduino tare] {e}')

    def _parse_line(self, line):
        if line.startswith('TARE DONE'):
            print('[seeeduino] TARE DONE'); self._tare_event.set(); return
        try:
            parts = line.split(',')
            if len(parts) == 2:
                fx, fy = float(parts[0]), float(parts[1])
                with self._lock:
                    self._fx = fx; self._fy = fy
                    self._timestamp = time.time()
        except Exception: pass

    def _loop(self):
        while self._running:
            try:
                line = self.serialInst.readline().decode('utf-8', errors='ignore').strip()
                if line: self._parse_line(line)
            except Exception as e:
                if self._running: print(f'[seeeduino] {e}')

    @property
    def lc_x(self):
        with self._lock: return self._fx
    @property
    def lc_y(self):
        with self._lock: return self._fy
    @property
    def tare_confirmed(self): return self._tare_event.is_set()

    def is_stale(self):
        with self._lock: ts = self._timestamp
        return ts == 0.0 or (time.time()-ts)*1000 > 100

    def start(self):
        try:
            self.serialInst.timeout = None
            self.serialInst.open()
            print(f'[seeeduino] connected to {self.serialInst.port}')
            threading.Thread(target=self._loop, daemon=True).start()
        except serial.SerialException as e:
            print(f'[seeeduino] {e}')

    def stop(self):
        self._running = False
        if self.serialInst.is_open: self.serialInst.close()


# ── workspace canvas ──────────────────────────────────────────────────────────
class WorkspaceCanvas(QWidget):
    directionChanged = Signal(float, float)

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.setMinimumHeight(300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

    def _w2c(self, wx, wz):
        w, h = self.width(), self.height()
        return QPointF((wx-WX0)/(WX1-WX0)*w, (wz-WZ0)/(WZ1-WZ0)*h)

    def _upd(self, e):
        s = self.state
        with s.lock: np_ = self._w2c(s.noark_x, s.noark_z)
        dx = e.position().x()-np_.x(); dy = e.position().y()-np_.y()
        l  = math.hypot(dx, dy)
        if l < 4: return
        self.directionChanged.emit(dx/l, dy/l)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton: self._upd(e)
    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton: self._upd(e)

    def paintEvent(self, _):
        s = self.state
        with s.lock:
            nx_w, nz_w  = s.noark_x, s.noark_z
            has_noark   = s.has_noark
            dx, dz      = s.dir_x, s.dir_z
            fm          = s.force_mag
            meas_fx     = s.meas_fx; meas_fz = s.meas_fz; meas_mag = s.meas_mag
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, C_BG)
        p.setPen(QPen(C_GRID, 1))
        for xv in [-0.4,-0.2,0,0.2,0.4]:
            pt = self._w2c(xv, 0); p.drawLine(QPointF(pt.x(),0), QPointF(pt.x(),h))
        for zv in [-0.8,-0.6,-0.4,-0.2]:
            pt = self._w2c(0, zv); p.drawLine(QPointF(0,pt.y()), QPointF(w,pt.y()))
        p.setPen(QPen(C_RAIL, 2))
        p.drawLine(self._w2c(-0.6,-0.773), self._w2c(0.6,-0.773))

        Fx = fm*dx; Fz = fm*dz
        sol = solve_tensions(nx_w, nz_w, Fx, Fz)
        T1 = max(0.0, sol['T1']) if sol else 0.0
        T3 = max(0.0, sol['T3']) if sol else 0.0

        np_ = self._w2c(nx_w, nz_w)
        ml  = self._w2c(*ML); mr = self._w2c(*MR)
        p2  = self._w2c(*P2); p4 = self._w2c(*P4)

        def cable(pts, col, t):
            alpha = int(64 + min(t,MAX_F)/MAX_F*191)
            c = QColor(col); c.setAlpha(alpha)
            pen = QPen(c, 1.0+t/MAX_F*3)
            pen.setCapStyle(Qt.RoundCap); pen.setJoinStyle(Qt.RoundJoin)
            p.setPen(pen); path = QPainterPath(); path.moveTo(pts[0])
            for pt in pts[1:]: path.lineTo(pt)
            p.drawPath(path)

        cable([ml, p2, np_], C_BLUE,  T1)
        cable([mr, p4, np_], C_GREEN, T3)

        for pt in (ml, mr):
            p.setBrush(QBrush(QColor('#332200'))); p.setPen(QPen(C_AMBER, 1.5))
            p.drawRect(QRectF(pt.x()-7, pt.y()-7, 14, 14))
        for pt, col in [(p2, C_BLUE), (p4, C_GREEN)]:
            p.setBrush(QBrush(C_BG)); p.setPen(QPen(col, 1.5))
            p.drawEllipse(QPointF(pt.x(), pt.y()), 7, 7)

        ncol = C_RED if has_noark else QColor('#444')
        p.setBrush(QBrush(QColor('#220d0d') if has_noark else QColor('#1a1a1a')))
        p.setPen(QPen(ncol, 2))
        p.drawEllipse(QPointF(np_.x(), np_.y()), 10, 10)
        p.setPen(C_WHITE); p.setFont(QFont('Courier New', 8, QFont.Bold))
        p.drawText(QRectF(np_.x()-8, np_.y()-6, 16, 12), Qt.AlignCenter, 'N')
        p.setPen(ncol); p.setFont(QFont('Courier New', 8))
        p.drawText(QPointF(np_.x()-30, np_.y()-14), f'({nx_w:.3f}, {nz_w:.3f})')

        if fm > 0.1 and has_noark:
            ln = 35+fm/MAX_F*45; ax = np_.x()+dx*ln; ay = np_.y()+dz*ln
            p.setPen(QPen(C_RED, 2)); p.drawLine(QPointF(np_.x(),np_.y()), QPointF(ax,ay))
            self._head(p, np_.x(), np_.y(), ax, ay, C_RED)
        if meas_mag > 0.1 and has_noark:
            mdx = meas_fx/meas_mag; mdz = meas_fz/meas_mag
            ln = 35+meas_mag/MAX_F*45
            mx = np_.x()+mdx*ln; my = np_.y()+mdz*ln
            p.setPen(QPen(C_GREEN, 2)); p.drawLine(QPointF(np_.x(),np_.y()), QPointF(mx,my))
            self._head(p, np_.x(), np_.y(), mx, my, C_GREEN)
        p.end()

    @staticmethod
    def _head(painter, x0, y0, x1, y1, color):
        a = math.atan2(y1-y0, x1-x0); sz = 10; s = math.radians(25)
        pts = QPolygonF([QPointF(x1,y1),
                         QPointF(x1-sz*math.cos(a-s), y1-sz*math.sin(a-s)),
                         QPointF(x1-sz*math.cos(a+s), y1-sz*math.sin(a+s))])
        painter.setBrush(QBrush(color)); painter.setPen(Qt.NoPen)
        painter.drawPolygon(pts)


# ── helpers ───────────────────────────────────────────────────────────────────
def _sec(t):
    l = QLabel(t.upper()); l.setFont(QFont('Courier New', 8))
    l.setStyleSheet('color:#444;padding:6px 10px 2px 10px;'); return l

def _sep():
    f = QFrame(); f.setFrameShape(QFrame.HLine)
    f.setStyleSheet('color:#2a2a2a;margin:0 10px;'); return f

def _row(layout, label, color='#ccc'):
    w = QWidget(); w.setStyleSheet('background:transparent;')
    h = QHBoxLayout(w); h.setContentsMargins(10,2,10,2)
    ll = QLabel(label); ll.setFont(QFont('Courier New', 12)); ll.setStyleSheet('color:#555;')
    vl = QLabel('--');  vl.setFont(QFont('Courier New', 11, QFont.Bold))
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
    v = QVBoxLayout(w); v.setContentsMargins(10,2,10,2); v.addWidget(widget)
    layout.addWidget(w)


# ── main window ───────────────────────────────────────────────────────────────
class NOARKWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('NOARK Force Validation (simple)')
        self.resize(860, 560)
        self.setStyleSheet('QMainWindow,QWidget{background:#111;}')
        self.state   = STATE
        self.running = True
        self._enc    = None
        self._lc     = None
        self._threads = []
        self._display_tick = 0

        dlg = QInputDialog(self)
        dlg.setWindowTitle('Session Name')
        dlg.setLabelText('Enter session name:')
        dlg.setStyleSheet(
            'QDialog,QWidget{background:#f0f0f0;color:#111;}'
            'QLineEdit{background:#fff;color:#111;border:1px solid #aaa;padding:4px;}'
            'QPushButton{background:#ddd;color:#111;border:1px solid #aaa;padding:4px 12px;}'
            'QPushButton:hover{background:#bbb;}')
        ok   = dlg.exec()
        name = dlg.textValue()
        if not ok or not name.strip(): name = 'session'
        self._logger   = DataLogger(base_dir='csv_data', session_name=name.strip())
        self._rec_state = 'idle'
        self._arm_t0    = 0.0

        self._arm_timer = QTimer(self)
        self._arm_timer.timeout.connect(self._check_arm)
        QShortcut(QKeySequence('S'), self, activated=self._begin_arming)
        QShortcut(QKeySequence('X'), self, activated=self._stop_recording)

        self._build_ui()

        # capture position once in background — GUI shows 'searching' until done
        self._capture_position_once()

        self._start_teensy()
        self._sweep = AutoSweep(self, self.state, self._enc,
                                session_dir=self._logger.session_dir)
        self._start_loadcell()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(5)

        t = threading.Thread(target=self._log_loop, daemon=True)
        t.start(); self._threads.append(t)

    # ── camera: 100-sample averaged capture ──────────────────────────────────
    def _capture_position_once(self):
        cam = MainClass(CAM_TOML, TABLE_TOML)
        N_SAMPLES = 100

        def _search():
            samples_x, samples_z = [], []
            t0 = time.time()
            while time.time() - t0 < 30.0 and len(samples_x) < N_SAMPLES:
                try:
                    cam.process_frame()
                    pos = cam.noark_in_table_frame
                    if pos is not None:
                        samples_x.append(float(pos[0]))
                        samples_z.append(float(pos[2]))
                        with self.state.lock:
                            self.state.noark_x   = float(pos[0])
                            self.state.noark_z   = float(pos[2])
                            self.state.has_noark = True
                            self.state.cam_status = f'collecting {len(samples_x)}/{N_SAMPLES}'
                except Exception as e:
                    print(f'[cam] {e}'); time.sleep(0.05)

            if len(samples_x) == 0:
                with self.state.lock:
                    self.state.cam_status = 'timeout'
                print('[cam] NOARK not found within 30 s')
                return

            nx = float(np.mean(samples_x))
            nz = float(np.mean(samples_z))
            with self.state.lock:
                self.state.noark_x   = nx
                self.state.noark_z   = nz
                self.state.has_noark = True
                self.state.cam_status = 'captured'
            print(f'[cam] position captured (n={len(samples_x)}): x={nx:.4f}  z={nz:.4f}'
                  f'  std_x={np.std(samples_x):.4f}  std_z={np.std(samples_z):.4f}')

            import csv, datetime
            pos_path = os.path.join(self._logger.session_dir, 'position.csv')
            with open(pos_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['x', 'z', 'std_x', 'std_z', 'n_samples', 'timestamp'])
                w.writerow([nx, nz, np.std(samples_x), np.std(samples_z),
                            len(samples_x), datetime.datetime.now().isoformat()])
            print(f'[cam] position saved → {pos_path}')

        t = threading.Thread(target=_search, daemon=True)
        t.start(); self._threads.append(t)

    # ── UI build ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8,8,8,8); root.setSpacing(8)

        self.workspace = WorkspaceCanvas(self.state)
        self.workspace.directionChanged.connect(self._on_direction)
        self.workspace.setStyleSheet('border:1px solid #2a2a2a;border-radius:2px;')
        root.addWidget(self.workspace, 3)

        sb = QWidget(); sb.setFixedWidth(230)
        sb.setStyleSheet('background:#141414;border-radius:2px;')
        sv = QVBoxLayout(sb); sv.setContentsMargins(0,4,0,8); sv.setSpacing(0)

        sv.addWidget(_sec('NOARK position (m)')); sv.addWidget(_sep())
        self.v_nx  = _row(sv, 'x', '#50b4ff')
        self.v_nz  = _row(sv, 'z', '#50b4ff')

        # camera status row
        cw = QWidget(); cw.setStyleSheet('background:transparent;')
        ch = QHBoxLayout(cw); ch.setContentsMargins(10,2,10,2)
        ch.addWidget(QLabel('camera') )
        self.v_cam = QLabel('searching')
        self.v_cam.setFont(QFont('Courier New', 12))
        self.v_cam.setStyleSheet('color:#f05050;')
        self.v_cam.setAlignment(Qt.AlignRight)
        ch.addWidget(self.v_cam); sv.addWidget(cw)

        sv.addWidget(_sec('Force')); sv.addWidget(_sep())
        self.v_fmag = _row(sv, 'magnitude', '#f05050')
        self.v_fdir = _row(sv, 'direction', '#888')
        slw = QWidget(); slw.setStyleSheet('background:transparent;')
        sll = QVBoxLayout(slw); sll.setContentsMargins(10,2,10,2)
        self.fmag_slider = QSlider(Qt.Horizontal)
        self.fmag_slider.setRange(0, int(MAX_F*10)); self.fmag_slider.setValue(0)
        self.fmag_slider.setStyleSheet(
            'QSlider::groove:horizontal{background:#2a2a2a;height:4px;border-radius:2px;}'
            'QSlider::handle:horizontal{background:#f05050;width:14px;height:14px;'
            'margin:-5px 0;border-radius:7px;}'
            'QSlider::sub-page:horizontal{background:#f05050;border-radius:2px;}')
        self.fmag_slider.valueChanged.connect(self._on_force_slider)
        sll.addWidget(self.fmag_slider); sv.addWidget(slw)

        sv.addWidget(_sec('Cable tensions')); sv.addWidget(_sep())
        self.v_t1   = _row(sv, 'T1 left',  '#50b4ff')
        self.v_t3   = _row(sv, 'T3 right', '#44cc88')
        sv.addWidget(_sec('Motor torques (Nm)')); sv.addWidget(_sep())
        self.v_tau1 = _row(sv, 'tau1', '#50b4ff')
        self.v_tau2 = _row(sv, 'tau2', '#44cc88')
        sv.addWidget(_sec('Measured force')); sv.addWidget(_sep())
        self.v_lc_status = _row(sv, 'status',    '#888')
        self.v_lc_mag    = _row(sv, 'magnitude', '#44cc88')
        self.v_lc_dir    = _row(sv, 'direction', '#44cc88')
        btn_tare = _btn('tare load cell', '#44cc88', '#0a1a10', '#44cc88')
        btn_tare.clicked.connect(self._tare_loadcell); _wrap(sv, btn_tare)
        sv.addWidget(_sec('Error')); sv.addWidget(_sep())
        self.v_err_mag = _row(sv, 'mag error', '#ffbb00')
        self.v_err_dir = _row(sv, 'dir error', '#ffbb00')
        sv.addStretch()

        btn_estop = _btn('E-STOP', '#f05050', '#1a0808', '#f05050')
        btn_estop.setFont(QFont('Courier New', 12, QFont.Bold))
        btn_estop.clicked.connect(self._estop); _wrap(sv, btn_estop)
        root.addWidget(sb, 0)

    # ── refresh (display only, ~50 Hz) ────────────────────────────────────────
    def _refresh(self):
        s = self.state
        with s.lock:
            has_noark = s.has_noark
            nx, nz    = s.noark_x, s.noark_z
            dir_x, dir_z = s.dir_x, s.dir_z
            fm        = s.force_mag
            lc_stale  = s.lc_stale
            sol       = s.cached_sol
            cam_status = s.cam_status

        meas_fx = s.lc_x; meas_fz = s.lc_y
        meas_mag = math.hypot(meas_fx, meas_fz)
        with s.lock:
            s.meas_fx = meas_fx; s.meas_fz = meas_fz; s.meas_mag = meas_mag

        self._display_tick += 1
        if self._display_tick % 4 != 0:
            return

        # camera status label
        if cam_status == 'captured':
            self.v_cam.setText('captured')
            self.v_cam.setStyleSheet('color:#44cc88;')
        elif cam_status == 'timeout':
            self.v_cam.setText('timeout')
            self.v_cam.setStyleSheet('color:#ffbb00;')

        self.v_nx.setText(f'{nx:.4f}' if has_noark else '--')
        self.v_nz.setText(f'{nz:.4f}' if has_noark else '--')

        cmd_dir = math.degrees(math.atan2(dir_z, dir_x))
        self.v_fmag.setText(f'{fm:.1f} N')
        self.v_fdir.setText(f'{cmd_dir:.1f} deg')

        if sol and has_noark:
            T1 = max(T_MIN, sol['T1']); T3 = max(T_MIN, sol['T3'])
            self.v_t1.setText(f'{T1:.2f} N');    self.v_t3.setText(f'{T3:.2f} N')
            self.v_tau1.setText(f'{-(T1*R_SPOOL):.3f}')
            self.v_tau2.setText(f'{T3*R_SPOOL:.3f}')
        else:
            for w in (self.v_t1, self.v_t3, self.v_tau1, self.v_tau2): w.setText('--')

        meas_dir = math.degrees(math.atan2(meas_fz, meas_fx))
        if lc_stale:
            self.v_lc_status.setText('no data')
            self.v_lc_status.setStyleSheet('color:#f05050;')
            for w in (self.v_lc_mag, self.v_lc_dir, self.v_err_mag, self.v_err_dir):
                w.setText('--')
        else:
            self.v_lc_status.setText('live')
            self.v_lc_status.setStyleSheet('color:#44cc88;')
            self.v_lc_mag.setText(f'{meas_mag:.2f} N')
            self.v_lc_dir.setText(f'{meas_dir:.1f} deg')
            err_mag = abs(meas_mag - fm)
            err_dir = abs(cmd_dir - meas_dir)
            if err_dir > 180: err_dir -= 360
            self.v_err_mag.setText(f'{err_mag:.2f} N')
            self.v_err_dir.setText(f'{err_dir:.1f} deg')

        self.workspace.update()

    # ── log loop (runs at max speed) ──────────────────────────────────────────
    def _log_loop(self):
        while self.running:
            s = self.state
            with s.lock:
                has_noark = s.has_noark
                nx, nz    = s.noark_x, s.noark_z
                dir_x, dir_z = s.dir_x, s.dir_z
                fm        = s.force_mag
            if not has_noark:
                time.sleep(0.005); continue
            sol = solve_tensions(nx, nz, fm*dir_x, fm*dir_z)
            with s.lock:
                s.cached_sol = sol
            T1 = T3 = tau1 = tau2 = 0.0
            if sol:
                T1 = max(T_MIN, sol['T1']); T3 = max(T_MIN, sol['T3'])
                tau1 = -(T1*R_SPOOL);       tau2 = T3*R_SPOOL
            self._logger.log_gui(fm, math.degrees(math.atan2(fm*dir_z, fm*dir_x)),
                                 fm*dir_x, fm*dir_z, T1, T3, tau1, tau2)
            time.sleep(0.005)

    # ── device setup ──────────────────────────────────────────────────────────
    def _start_teensy(self):
        try:
            self._enc = TeensyPort()
            self._enc.start()
            time.sleep(2.0)
            self._enc.encoder_reset()
            print('[teensy] connected and tared')
        except Exception as e:
            print(f'[teensy] {e}'); self._enc = None

    def _start_loadcell(self):
        try:
            self._lc = SeeduinoReceiver()
            self._lc.start()
            original_parse = self._lc._parse_line
            def _patched_parse(line):
                if line.startswith('TARE DONE'):
                    original_parse(line); return
                original_parse(line)
                self._logger.log_loadcell(self._lc.lc_x, self._lc.lc_y)
            self._lc._parse_line = _patched_parse
        except Exception as e:
            print(f'[loadcell] {e}'); self._lc = None; return

        def loop():
            while self.running:
                if self._lc:
                    with self.state.lock:
                        self.state.lc_x    = self._lc.lc_x
                        self.state.lc_y    = self._lc.lc_y
                        self.state.lc_stale = self._lc.is_stale()
                time.sleep(0.001)
        t = threading.Thread(target=loop, daemon=True)
        t.start(); self._threads.append(t)

    # ── controls ──────────────────────────────────────────────────────────────
    def _on_direction(self, dx, dz):
        with self.state.lock:
            self.state.dir_x = dx; self.state.dir_z = dz

    def _on_force_slider(self, val):
        with self.state.lock:
            self.state.force_mag = val / 10.0

    def _tare_loadcell(self):
        if self._lc: self._lc.send_tare()

    def _begin_arming(self):
        if self._rec_state != 'idle': return
        if not self._lc:
            print('[arm] load cell not connected'); return

        # zero motors first so cables are slack during tare
        if self._enc:
            try:
                self._enc.serialInst.write(b'0.000,0.000\n')
                print('[arm] motors zeroed')
            except Exception as e:
                print(f'[arm] motor zero failed: {e}')

        # wait 500 ms for cables to go slack before taring
        print('[arm] waiting 500 ms for cables to go slack…')
        QTimer.singleShot(500, self._do_tare)

    def _do_tare(self):
        print('[arm] taring load cell…')
        self._lc.send_tare()
        self._rec_state = 'arming'
        self._arm_t0    = time.time()
        self._arm_timer.start(10)

    def _check_arm(self):
        if self._rec_state != 'arming':
            self._arm_timer.stop(); return
        lc_ok = self._lc is not None and self._lc.tare_confirmed
        if lc_ok:
            self._arm_timer.stop()
            self._logger.start()
            self._rec_state = 'recording'
            print('[arm] tare confirmed → RECORDING, starting sweep')
            self._sweep.start()
        elif time.time() - self._arm_t0 > 5.0:
            self._arm_timer.stop()
            self._rec_state = 'idle'
            print('[arm] TIMEOUT — load cell tare not confirmed')

    def _stop_recording(self):
        if self._rec_state == 'recording':
            self._logger._active = False
            self._rec_state = 'idle'
            print('[arm] recording stopped')

    def _estop(self):
        if getattr(self, '_sweep', None): self._sweep.stop()
        with self.state.lock: self.state.force_mag = 0.0
        self.fmag_slider.setValue(0)
        if self._enc:
            try: self._enc.serialInst.write(b'0.000,0.000\n')
            except Exception: pass
        print('[E-STOP] torques zeroed')

    def closeEvent(self, event):
        self.running = False
        self._timer.stop()
        if self._enc:
            try: self._enc.serialInst.write(b'0.000,0.000\n')
            except Exception: pass
        if self._lc: self._lc.stop()
        for t in self._threads:
            t.join(timeout=3.0)
        self._logger.stop()
        if getattr(self, '_sweep', None): self._sweep.stop()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv); app.setStyle('Fusion')
    win = NOARKWindow(); win.show(); sys.exit(app.exec())

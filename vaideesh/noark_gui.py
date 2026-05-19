"""
noark_gui.py
------------
Run this on the Pi. It imports your existing camera_pose and pyteensy directly.

    python noark_gui.py

Controls:
    Click canvas  → set force direction
    Slider        → force magnitude (0–24 N)
    Reset button  → zero encoders
    Send button   → write torques to Teensy (uncomment serial line below)
    E-STOP        → zero force immediately
"""

import tkinter as tk
import threading, time, math, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camera_pose import MainClass
from pyteensy   import TeensyPort

# ── calibration paths ─────────────────────────────────────────────────────
CAM_TOML   = "/home/sujith/Documents/NOARK_backbone/notebooks/calibration/output/good.toml"
TABLE_TOML = "/home/sujith/Documents/NOARK_backbone/estimator/charuco_pose/charuco_pose.toml"

# ── geometry (table-frame metres, x-z plane) ──────────────────────────────
P1 = (-0.495, -0.773)   # left pulley
P3 = ( 0.495, -0.773)   # right pulley
ML = (-0.065, -0.707)   # left motor
MR = ( 0.065, -0.707)   # right motor
R_SPOOL = 0.033         # motor spool radius (m)
MAX_F   = 24.0          # N

# ── world bounds for canvas mapping ──────────────────────────────────────
WX0, WX1 = -0.65,  0.65
WZ0, WZ1 = -1.05, -0.05


class NOARKGui:
    def __init__(self, root):
        self.root = root
        root.title("NOARK Force Control")
        root.configure(bg="#111")
        root.resizable(True, True)

        # ── shared state ──────────────────────────────────────────────
        self.lock     = threading.Lock()
        self.noark_x  = 0.0
        self.noark_z  = -0.4
        self.has_noark = False
        self.enc1     = 0.0
        self.enc2     = 0.0
        self.dir_x    = 0.0      # unit force direction
        self.dir_z    = -1.0
        self.force_mag = 12.0
        self.running  = True

        self._build_ui()
        self._start_camera()
        self._start_teensy()
        self._redraw_loop()

    # ──────────────────────────────────────────────────────────────────
    # UI LAYOUT
    # ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        BG   = "#111"
        BG2  = "#1a1a1a"
        FG   = "#ddd"
        MUTED = "#666"
        FONT  = ("Courier New", 11)
        FONT_S = ("Courier New", 10)
        FONT_B = ("Courier New", 12, "bold")

        # ── top frame (canvas) ────────────────────────────────────────
        self.canvas = tk.Canvas(self.root, bg="#0d0d0d",
                                highlightthickness=1,
                                highlightbackground="#333",
                                width=560, height=380)
        self.canvas.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.canvas.bind("<Button-1>",    self._on_canvas_click)
        self.canvas.bind("<B1-Motion>",   self._on_canvas_click)

        # ── right sidebar ─────────────────────────────────────────────
        side = tk.Frame(self.root, bg=BG2, width=230)
        side.grid(row=0, column=1, padx=(0,10), pady=10, sticky="nsew")
        side.grid_propagate(False)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        def section(parent, title):
            tk.Label(parent, text=title.upper(), bg=BG2, fg=MUTED,
                     font=("Courier New", 9), anchor="w"
                     ).pack(fill="x", padx=10, pady=(10,2))
            tk.Frame(parent, bg="#2a2a2a", height=1).pack(fill="x", padx=10)

        def row(parent, label, var, color="#fff"):
            f = tk.Frame(parent, bg=BG2)
            f.pack(fill="x", padx=10, pady=2)
            tk.Label(f, text=label, bg=BG2, fg=MUTED, font=FONT_S,
                     anchor="w", width=14).pack(side="left")
            lbl = tk.Label(f, textvariable=var, bg=BG2, fg=color,
                           font=FONT_B, anchor="e")
            lbl.pack(side="right")
            return lbl

        # ── NOARK position ────────────────────────────────────────────
        self.v_nx  = tk.StringVar(value="—")
        self.v_nz  = tk.StringVar(value="—")
        self.v_cam = tk.StringVar(value="searching…")

        section(side, "noark position (m)")
        row(side, "x", self.v_nx, "#5bf")
        row(side, "z", self.v_nz, "#5bf")
        f = tk.Frame(side, bg=BG2); f.pack(fill="x", padx=10, pady=2)
        tk.Label(f, text="camera", bg=BG2, fg=MUTED, font=FONT_S).pack(side="left")
        self.cam_lbl = tk.Label(f, textvariable=self.v_cam, bg=BG2,
                                fg="#f55", font=FONT_S)
        self.cam_lbl.pack(side="right")

        # ── encoders ──────────────────────────────────────────────────
        self.v_e1 = tk.StringVar(value="0.00")
        self.v_e2 = tk.StringVar(value="0.00")

        section(side, "encoders (°)")
        row(side, "enc 1 (left)",  self.v_e1, "#fb0")
        row(side, "enc 2 (right)", self.v_e2, "#fb0")
        tk.Button(side, text="reset encoders",
                  command=self._reset_encoders,
                  bg="#1a1a1a", fg="#fb0", activebackground="#222",
                  relief="flat", bd=1, highlightthickness=1,
                  highlightbackground="#444",
                  font=FONT_S, cursor="hand2"
                  ).pack(fill="x", padx=10, pady=(4,0))

        # ── force ─────────────────────────────────────────────────────
        self.v_fmag = tk.StringVar(value="12.0 N")
        self.v_fdir = tk.StringVar(value="—")

        section(side, "force")
        row(side, "magnitude", self.v_fmag, "#f55")
        self.fmag_slider = tk.Scale(
            side, from_=0, to=MAX_F, resolution=0.5,
            orient="horizontal", bg=BG2, fg=FG,
            highlightthickness=0, troughcolor="#333",
            activebackground="#f55", font=FONT_S,
            command=self._on_force_slider
        )
        self.fmag_slider.set(12.0)
        self.fmag_slider.pack(fill="x", padx=10)
        row(side, "direction", self.v_fdir, "#aaa")

        # ── cable tensions ────────────────────────────────────────────
        self.v_t1   = tk.StringVar(value="— N")
        self.v_t2   = tk.StringVar(value="— N")
        self.v_tau1 = tk.StringVar(value="—")
        self.v_tau2 = tk.StringVar(value="—")

        section(side, "cable tensions")
        row(side, "T₁ left",  self.v_t1, "#5bf")
        row(side, "T₃ right", self.v_t2, "#4c8")

        # ── motor torques ─────────────────────────────────────────────
        section(side, "motor torques (Nm)")
        row(side, "Tou_1", self.v_tau1, "#5bf")
        row(side, "Tou_2", self.v_tau2, "#4c8")

        # ── buttons ───────────────────────────────────────────────────
        tk.Frame(side, bg=BG2).pack(expand=True)   # spacer

        tk.Button(side, text="send to Teensy",
                  command=self._send_force,
                  bg="#10182a", fg="#5bf", activebackground="#1a2a40",
                  relief="flat", highlightthickness=1,
                  highlightbackground="#5bf",
                  font=FONT_S, cursor="hand2", pady=7
                  ).pack(fill="x", padx=10, pady=(0,6))

        tk.Button(side, text="■  E-STOP",
                  command=self._estop,
                  bg="#2a1010", fg="#f55", activebackground="#3a1515",
                  relief="flat", highlightthickness=1,
                  highlightbackground="#f55",
                  font=("Courier New", 12, "bold"),
                  cursor="hand2", pady=7
                  ).pack(fill="x", padx=10, pady=(0,10))

    # ──────────────────────────────────────────────────────────────────
    # CANVAS DRAWING
    # ──────────────────────────────────────────────────────────────────
    def _w2c(self, wx, wz):
        """World (x,z) → canvas (cx,cy)."""
        cw = self.canvas.winfo_width()  or 560
        ch = self.canvas.winfo_height() or 380
        cx = (wx - WX0) / (WX1 - WX0) * cw
        cy = (wz - WZ0) / (WZ1 - WZ0) * ch
        return cx, cy

    def _c2w(self, cx, cy):
        cw = self.canvas.winfo_width()  or 560
        ch = self.canvas.winfo_height() or 380
        return (WX0 + cx / cw * (WX1 - WX0),
                WZ0 + cy / ch * (WZ1 - WZ0))

    def _draw(self):
        c = self.canvas
        c.delete("all")
        cw = c.winfo_width()  or 560
        ch = c.winfo_height() or 380

        # Background
        c.create_rectangle(0, 0, cw, ch, fill="#0d0d0d", outline="")

        # Grid
        for x in [-0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6]:
            cx, _ = self._w2c(x, 0)
            c.create_line(cx, 0, cx, ch, fill="#1a1a1a")
        for z in [-1.0, -0.8, -0.6, -0.4, -0.2]:
            _, cy = self._w2c(0, z)
            c.create_line(0, cy, cw, cy, fill="#1a1a1a")

        # Top rail
        rx0, ry = self._w2c(-0.6, -0.773)
        rx1, _  = self._w2c( 0.6, -0.773)
        c.create_line(rx0, ry, rx1, ry, fill="#444", width=2)

        # Solve tensions
        Fx = self.force_mag * self.dir_x
        Fz = self.force_mag * self.dir_z
        sol = self._solve(self.noark_x, self.noark_z, Fx, Fz)
        T1 = max(0.0, sol["T1"]) if sol else 0.0
        T3 = max(0.0, sol["T3"]) if sol else 0.0

        # Cable routes: ML→P1→NOARK  and  MR→P3→NOARK
        nx, ny   = self._w2c(self.noark_x, self.noark_z)
        mlx, mly = self._w2c(*ML)
        mrx, mry = self._w2c(*MR)
        p1x, p1y = self._w2c(*P1)
        p3x, p3y = self._w2c(*P3)

        alpha1 = int(64 + T1 / MAX_F * 191)
        alpha3 = int(64 + T3 / MAX_F * 191)
        w1 = 1 + T1 / MAX_F * 3
        w3 = 1 + T3 / MAX_F * 3

        c.create_line(mlx, mly, p1x, p1y, nx, ny,
                      fill=self._hex_alpha("#50b4ff", alpha1), width=w1)
        c.create_line(mrx, mry, p3x, p3y, nx, ny,
                      fill=self._hex_alpha("#44cc88", alpha3), width=w3)

        # ML, MR — amber squares
        for (px, py), lbl in [((mlx, mly), "ML"), ((mrx, mry), "MR")]:
            c.create_rectangle(px-7, py-7, px+7, py+7,
                               fill="#332200", outline="#fb0", width=1)
            c.create_text(px, py+16, text=lbl, fill="#fb0",
                          font=("Courier New", 9))

        # P1, P3 — circles
        for (px, py), lbl, col in [
            ((p1x, p1y), "P1", "#5bf"),
            ((p3x, p3y), "P3", "#4c8"),
        ]:
            c.create_oval(px-7, py-7, px+7, py+7,
                          fill="#0d0d0d", outline=col, width=1.5)
            c.create_text(px, py-14, text=lbl, fill=col,
                          font=("Courier New", 9))

        # NOARK dot
        ncol = "#f55" if self.has_noark else "#444"
        c.create_oval(nx-10, ny-10, nx+10, ny+10,
                      fill="#220d0d" if self.has_noark else "#1a1a1a",
                      outline=ncol, width=2)
        c.create_text(nx, ny, text="N", fill="#fff",
                      font=("Courier New", 9, "bold"))
        c.create_text(nx, ny-20,
                      text=f"({self.noark_x:.3f}, {self.noark_z:.3f})",
                      fill=ncol, font=("Courier New", 8))

        # Force arrow
        if self.force_mag > 0.1 and self.has_noark:
            length = 35 + self.force_mag / MAX_F * 45
            ax = nx + self.dir_x * length
            ay = ny + self.dir_z * length
            c.create_line(nx, ny, ax, ay, fill="#f55", width=2,
                          arrow="last", arrowshape=(10, 12, 4))
            c.create_text(ax + 6, ay - 8,
                          text=f"{self.force_mag:.1f}N",
                          fill="#f55", font=("Courier New", 9), anchor="w")

    @staticmethod
    def _hex_alpha(hex_col, alpha):
        """Return tkinter-compatible solid approximation (tk doesn't do alpha)."""
        # Blend with #0d0d0d background
        r = int(hex_col[1:3], 16)
        g = int(hex_col[3:5], 16)
        b = int(hex_col[5:7], 16)
        a = alpha / 255
        br, bg, bb = 13, 13, 13   # background #0d0d0d
        rr = int(r * a + br * (1 - a))
        rg = int(g * a + bg * (1 - a))
        rb = int(b * a + bb * (1 - a))
        return f"#{rr:02x}{rg:02x}{rb:02x}"

    # ──────────────────────────────────────────────────────────────────
    # KINEMATICS
    # ──────────────────────────────────────────────────────────────────
    @staticmethod
    def _solve(nx, nz, Fx, Fz):
        def unit(dx, dz):
            l = math.hypot(dx, dz)
            return (dx / l, dz / l) if l > 1e-9 else (0, 0)

        u1x, u1z = unit(P1[0] - nx, P1[1] - nz)
        u3x, u3z = unit(P3[0] - nx, P3[1] - nz)
        det = u1x * u3z - u3x * u1z
        if abs(det) < 1e-6:
            return None
        return {
            "T1": (Fx * u3z - Fz * u3x) / det,
            "T3": (u1x * Fz - u1z * Fx) / det,
        }

    # ──────────────────────────────────────────────────────────────────
    # SIDEBAR UPDATE
    # ──────────────────────────────────────────────────────────────────
    def _update_sidebar(self):
        # NOARK
        if self.has_noark:
            self.v_nx.set(f"{self.noark_x:.4f}")
            self.v_nz.set(f"{self.noark_z:.4f}")
            self.v_cam.set("tracking")
            self.cam_lbl.config(fg="#4c8")
        else:
            self.v_nx.set("—")
            self.v_nz.set("—")
            self.v_cam.set("searching…")
            self.cam_lbl.config(fg="#f55")

        # Encoders
        self.v_e1.set(f"{self.enc1:.2f}")
        self.v_e2.set(f"{self.enc2:.2f}")

        # Force direction angle
        ang = math.degrees(math.atan2(self.dir_z, self.dir_x))
        self.v_fdir.set(f"{ang:.1f}°")
        self.v_fmag.set(f"{self.force_mag:.1f} N")

        # Tensions & torques
        Fx = self.force_mag * self.dir_x
        Fz = self.force_mag * self.dir_z
        sol = self._solve(self.noark_x, self.noark_z, Fx, Fz)
        if sol and self.has_noark:
            T1   = max(0.0, sol["T1"])
            T3   = max(0.0, sol["T3"])
            tau1 = -(T1 * R_SPOOL)
            tau2 =   T3 * R_SPOOL
            self.v_t1.set(f"{T1:.2f} N")
            self.v_t2.set(f"{T3:.2f} N")
            self.v_tau1.set(f"{tau1:.3f}")
            self.v_tau2.set(f"{tau2:.3f}")
            return tau1, tau2
        else:
            self.v_t1.set("— N")
            self.v_t2.set("— N")
            self.v_tau1.set("—")
            self.v_tau2.set("—")
            return None

    # ──────────────────────────────────────────────────────────────────
    # REDRAW LOOP  (runs on main thread via after())
    # ──────────────────────────────────────────────────────────────────
    def _redraw_loop(self):
        with self.lock:
            self._draw()
            self._update_sidebar()
        self.root.after(50, self._redraw_loop)   # ~20 Hz

    # ──────────────────────────────────────────────────────────────────
    # CANVAS INTERACTION
    # ──────────────────────────────────────────────────────────────────
    def _on_canvas_click(self, event):
        nx, ny = self._w2c(self.noark_x, self.noark_z)
        dx, dy = event.x - nx, event.y - ny
        length = math.hypot(dx, dy)
        if length < 4:
            return
        with self.lock:
            self.dir_x = dx / length
            self.dir_z = dy / length

    def _on_force_slider(self, val):
        with self.lock:
            self.force_mag = float(val)

    # ──────────────────────────────────────────────────────────────────
    # BUTTONS
    # ──────────────────────────────────────────────────────────────────
    def _reset_encoders(self):
        self.enc.encoder_reset()

    def _send_force(self):
        res = self._update_sidebar()
        if res is None:
            return
        tau1, tau2 = res
        # ── uncomment once firmware serial-receive is enabled ──────────
        # line = f"{tau1:.3f},{tau2:.3f}\n"
        # self.enc.serialInst.write(line.encode())
        print(f"[send] Tou_1={tau1:.3f}  Tou_2={tau2:.3f}")

    def _estop(self):
        with self.lock:
            self.force_mag = 0.0
        self.fmag_slider.set(0.0)
        # self.enc.serialInst.write(b"0.000,0.000\n")
        print("[E-STOP] torques zeroed")

    # ──────────────────────────────────────────────────────────────────
    # BACKGROUND THREADS
    # ──────────────────────────────────────────────────────────────────
    def _start_camera(self):
        self.cam = MainClass(CAM_TOML, TABLE_TOML)

        def loop():
            while self.running:
                try:
                    self.cam.process_frame()
                    pos = self.cam.noark_in_table_frame
                    with self.lock:
                        if pos is not None:
                            self.noark_x  = float(pos[0])
                            self.noark_z  = float(pos[2])
                            self.has_noark = True
                        else:
                            self.has_noark = False
                except Exception as e:
                    print(f"[cam] {e}")
                    time.sleep(0.05)

        threading.Thread(target=loop, daemon=True).start()

    def _start_teensy(self):
        try:
            self.enc = TeensyPort()
            self.enc.start()
        except Exception as e:
            print(f"[teensy] {e} — encoder display disabled")
            self.enc = None

        def loop():
            while self.running:
                time.sleep(0.02)
                if self.enc:
                    with self.lock:
                        self.enc1 = self.enc.enc1
                        self.enc2 = self.enc.enc2

        threading.Thread(target=loop, daemon=True).start()

    def on_close(self):
        self.running = False
        self.root.destroy()


# ── entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app  = NOARKGui(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

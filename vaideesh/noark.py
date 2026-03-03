import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

R_M = 4.0
R_P = 2.0
O = np.array([20.0, -4.0])   # motor center
A = np.array([2.0, -2.0])    # pulley A
B = np.array([2.0, -10.0])   # pulley B
box = np.array([15.0, -30.0])
y_target = box[1]

theta = 0.0
R_wound = 0.0

def external_tangents(O1, r1, O2, r2):
    dx, dy = O2 - O1
    d = np.hypot(dx, dy)
    if d <= abs(r1 - r2):  
        return (O1, O2), (O1, O2)
    ang = np.arctan2(dy, dx)
    alpha = np.arccos((r1 - r2) / d)
    a1 = ang + alpha
    a2 = ang - alpha
    T1a = O1 + r1 * np.array([np.cos(a1), np.sin(a1)])
    T2a = O2 + r2 * np.array([np.cos(a1), np.sin(a1)])
    T1b = O1 + r1 * np.array([np.cos(a2), np.sin(a2)])
    T2b = O2 + r2 * np.array([np.cos(a2), np.sin(a2)])
    return (T1a, T2a), (T1b, T2b)

def circle_point_tangents(C, r, P):
    v = P - C
    d = np.linalg.norm(v)
    if d <= r:
        return (C, C)
    phi = np.arctan2(v[1], v[0])
    alpha = np.arccos(r / d)
    t1 = phi + alpha
    t2 = phi - alpha
    T1 = C + r * np.array([np.cos(t1), np.sin(t1)])
    T2 = C + r * np.array([np.cos(t2), np.sin(t2)])
    if T1[0] < C[0]:
        return T1, T2
    else:
        return T2, T1

def arc_points(center, r, ang1, ang2, n=30):
    d = ang2 - ang1
    if d > np.pi:
        d -= 2*np.pi
    if d < -np.pi:
        d += 2*np.pi
    angs = np.linspace(ang1, ang1 + d, n)
    x = center[0] + r*np.cos(angs)
    y = center[1] + r*np.sin(angs)
    return np.column_stack([x, y])


fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(0,50)
ax.set_ylim(-50,0)
ax.set_aspect('equal')
ax.set_title("Motor–Pulley System (50×50 cm Table)")
ax.set_xlabel("X (cm)")
ax.set_ylabel("Y (cm)")
ax.add_patch(Rectangle((0,-50),50,50,fill=False,ls='--',color='gray'))

motor_circ = plt.Circle(O,R_M,fill=False,color='red',lw=2)
A_circ = plt.Circle(A,R_P,fill=False,color='blue',lw=2)
B_circ = plt.Circle(B,R_P,fill=False,color='blue',lw=2)
ax.add_patch(motor_circ); ax.add_patch(A_circ); ax.add_patch(B_circ)

rope_line, = ax.plot([],[],'k-',lw=2)
box_marker, = ax.plot([],[],'s',color='green',markersize=10)


def update(frame):
    global box, y_target, R_wound, theta

    if abs(box[1]-y_target)>0.1:
        dy = np.sign(y_target-box[1])*0.2
        box[1]+=dy
        R_wound += dy
        theta = R_wound / R_M  # radians

    # --- Motor → A 
   
    (Tma1, Ta1), (Tma2, Ta2) = external_tangents(O, R_M, A, R_P)

  
    candidates = [(Tma1, Ta1), (Tma2, Ta2)]
    TmA, TaM = min(candidates, key=lambda p: (p[0][1] > O[1], p[0][0]))  


   # --- A → B (outer / left side) ---
    (TaB1, TbB1), (TaB2, TbB2) = external_tangents(A, R_P, B, R_P)

   
    if TaB1[0] < A[0] and TbB1[0] < B[0]:
        TaB, TbA = TaB1, TbB1
    else:
        TaB, TbA = TaB2, TbB2

    # --- B → Box  
    Tb_left,_=circle_point_tangents(B,R_P,box)
    Tb = Tb_left

  
    
    ang_start = np.arctan2(TmA[1] - O[1], TmA[0] - O[0])
    ang_end   = ang_start - np.deg2rad(0)   
    arcM = arc_points(O, R_M, ang_start, ang_end)


    ang_A1=np.arctan2(TaM[1]-A[1],TaM[0]-A[0])
    ang_A2=np.arctan2(TaB[1]-A[1],TaB[0]-A[0])
    if ang_A2 < ang_A1:
        ang_A2 = ( np.pi )
    arcA = arc_points(A,R_P,ang_A1,ang_A2)

    ang_B1=np.arctan2(TbA[1]-B[1],TbA[0]-B[0])
    ang_B2=np.arctan2(Tb[1]-B[1],Tb[0]-B[0])
    arcB = arc_points(B,R_P,ang_B1,ang_B2)

    pts = np.vstack([arcM,[TaM],arcA,[TaB],arcB,[box]])
    rope_line.set_data(pts[:,0],pts[:,1])
    box_marker.set_data([box[0]],[box[1]])

    motor_circ.set_transform(
        Affine2D().rotate_around(O[0],O[1],-theta)+ax.transData
    )
    return rope_line, box_marker, motor_circ


def on_click(ev):
    global y_target
    if ev.inaxes!=ax: return
    y_click=ev.ydata
    y_target=np.clip(y_click,-45,-5)
    print(f"Clicked y={y_click:.1f} → new box y={y_target:.1f}")

fig.canvas.mpl_connect('button_press_event',on_click)

anim=FuncAnimation(fig,update,interval=50,blit=True)
plt.show()

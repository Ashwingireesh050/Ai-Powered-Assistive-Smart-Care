# ============================================================
#   HANDS-FREE CONTROLLER · 
#   Flow:
#     Mouth×2 -> YES/NO -> (blink=yes / mouth=no)
#     Menu: mouth = next · blink = select
#     After action: "ANY OTHER OPERATION?" (blink=yes / mouth=no)
# ============================================================

import cv2
import math
import time
import os
import queue
import threading

import numpy as np
import mediapipe as mp
from gesture_ai import GestureAI

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import serial as pyserial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# ============================================================
# CONFIG  (your original working values)
# ============================================================
CAMERA_INDEX  = 1
CAMERA_WIDTH  = 1280
CAMERA_HEIGHT = 720
WINDOW_NAME   = "HANDS-FREE CONTROLLER"

SERIAL_ENABLED = True
SERIAL_PORT    = "COM4"        # <-- your ESP32 HUB port
SERIAL_BAUD    = 115200

# ---------------- MOUTH (original) ----------------
MOUTH_OPEN_THRESHOLD  = 0.42
MOUTH_CLOSE_THRESHOLD = 0.32
MIN_OPEN_TIME   = 0.12
MAX_OPEN_TIME   = 0.90
MIN_BETWEEN_CYCLES = 0.12
MAX_GESTURE_TIME   = 2.2
REQUIRED_CYCLES  = 2
MENU_COOLDOWN    = 2.0

# ---------------- BLINK (original) ----------------
EYE_CLOSED_THRESHOLD = 0.20
EYE_OPEN_THRESHOLD   = 0.25
MIN_BLINK_TIME = 0.05
MAX_BLINK_TIME = 0.60

# ---------------- HEAD (original) ----------------
MAX_HEAD_MOVEMENT = 0.18

# ---------------- DIALOGS ----------------
STATE_GRACE      = 0.5     # ignore gestures right after a state change
YESNO_TIMEOUT    = 5.0     # "open menu?" auto-cancel
ASK_MORE_TIMEOUT = 6.0     # "any other operation?" auto-exit

MENU_ITEMS = ["FAN", "LIGHT", "SOS", "CALL"]

# ============================================================
# THEME (BGR)
# ============================================================
def _bgr(rgb): return (int(rgb[2]), int(rgb[1]), int(rgb[0]))

ACCENT = _bgr((0, 210, 255))
GREEN  = _bgr((70, 235, 120))
AMBER  = _bgr((60, 190, 255))
RED    = _bgr((90, 80, 255))
WHITE  = (245, 246, 248)
GRAY   = (158, 162, 170)
DGRAY  = (92, 96, 104)
PANEL  = (17, 19, 25)

STATE_NORMAL, STATE_YESNO, STATE_MENU, STATE_ASK_MORE = \
    "NORMAL", "YES/NO", "MENU", "AGAIN?"

STATE_COLORS = {"NORMAL": ACCENT, "YES/NO": AMBER, "MENU": GREEN, "AGAIN?": GREEN}
TOPBAR_LABEL = {"NORMAL": "TRACKING", "YES/NO": "CONFIRM", "MENU": "MENU", "AGAIN?": "AGAIN?"}

_FONT_PATHS = {
    True:  ["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf"],
    False: ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf"],
}
_fonts = {}
TEXTQ = []

def get_font(size, bold=False):
    if not PIL_AVAILABLE:
        return None
    key = (size, bold)
    if key not in _fonts:
        path = next((p for p in _FONT_PATHS[bold] if os.path.exists(p)), None)
        try:
            _fonts[key] = ImageFont.truetype(path, size) if path else ImageFont.load_default()
        except Exception:
            _fonts[key] = ImageFont.load_default()
    return _fonts[key]

def txt(x, y, s, size=16, color=WHITE, bold=False, anchor="la"):
    TEXTQ.append((x, y, str(s), size, tuple(int(c) for c in color), bold, anchor))

def flush_text(frame):
    if not TEXTQ:
        return frame
    if PIL_AVAILABLE:
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for x, y, s, size, color, bold, anchor in TEXTQ:
            rgb = (color[2], color[1], color[0])
            try:
                d.text((x, y), s, font=get_font(size, bold), fill=rgb, anchor=anchor)
            except TypeError:
                d.text((x, y), s, font=get_font(size, bold), fill=rgb)
        frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    else:
        for x, y, s, size, color, bold, anchor in TEXTQ:
            ax, ay = x, y
            if "r" in anchor:   ax -= int(len(s) * size * 0.42)
            elif "m" in anchor: ax -= int(len(s) * size * 0.21)
            if len(anchor) > 1 and anchor[1] == "m": ay += int(size * 0.35)
            cv2.putText(frame, s, (int(ax), int(ay)), cv2.FONT_HERSHEY_SIMPLEX,
                        size / 24.0, color, 2 if size >= 26 else 1, cv2.LINE_AA)
    TEXTQ.clear()
    return frame

# ============================================================
# DRAW PRIMITIVES
# ============================================================
def clamp(v, a, b): return max(a, min(b, v))
def lerp(a, b, k):  return a + (b - a) * k
def pulse(t, period=1.6): return 0.5 + 0.5 * math.sin(t * 6.28318 / period)

def mix(c1, c2, k):
    return (int(lerp(c1[0], c2[0], k)), int(lerp(c1[1], c2[1], k)), int(lerp(c1[2], c2[2], k)))

def faded(color, a, bg=(15, 17, 22)): return mix(bg, color, clamp(a, 0, 1))

SMOOTH = {}
def smooth(key, target, k=0.25):
    cur = SMOOTH.get(key, target)
    cur = lerp(cur, target, k)
    SMOOTH[key] = cur
    return cur

def rrect(img, p1, p2, color, radius, thickness=1, lt=cv2.LINE_AA):
    x1, y1 = int(p1[0]), int(p1[1]); x2, y2 = int(p2[0]), int(p2[1])
    if x2 - x1 < 2 or y2 - y1 < 2: return
    r = int(min(radius, (x2 - x1) // 2, (y2 - y1) // 2))
    if r < 1:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, lt); return
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, lt)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1, lt)
        for cx, cy in [(x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)]:
            cv2.circle(img, (cx, cy), r, color, -1, lt)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, lt)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, lt)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, lt)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, lt)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 0, 180, 270, color, thickness, lt)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 0, 270, 360, color, thickness, lt)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 0, 90, 180, color, thickness, lt)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, lt)

def glass(img, x, y, w, h, r=16, tint=PANEL, alpha=0.62, border=None):
    x, y, w, h = int(x), int(y), int(w), int(h)
    x0, y0 = clamp(x, 0, W), clamp(y, 0, H)
    x1, y1 = clamp(x + w, 0, W), clamp(y + h, 0, H)
    if x1 - x0 < 6 or y1 - y0 < 6: return
    roi = img[y0:y1, x0:x1]
    small = cv2.resize(roi, (max(1, (x1 - x0) // 6), max(1, (y1 - y0) // 6)))
    small = cv2.GaussianBlur(small, (0, 0), 3)
    blur = cv2.resize(small, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    img[y0:y1, x0:x1] = cv2.addWeighted(
        blur, 1 - alpha, np.full(blur.shape, tint, dtype=np.uint8), alpha, 0)
    if border is not None:
        rrect(img, (x, y), (x + w - 1, y + h - 1), border, r, 1)
        cl = 12
        for cx, cy, dx, dy in [(x, y, 1, 1), (x + w - 1, y, -1, 1),
                               (x, y + h - 1, 1, -1), (x + w - 1, y + h - 1, -1, -1)]:
            cv2.line(img, (cx, cy), (cx + dx * cl, cy), border, 2)
            cv2.line(img, (cx, cy), (cx, cy + dy * cl), border, 2)

def meter(img, x, y, w, h, frac, color, markers=()):
    frac = clamp(frac, 0, 1)
    rrect(img, (x, y), (x + w, y + h), (36, 40, 48), h // 2, -1)
    if frac > 0.03:
        rrect(img, (x, y), (x + int(w * frac), y + h), color, h // 2, -1)
    for m in markers:
        mx = x + int(w * clamp(m, 0, 1))
        cv2.line(img, (mx, y - 3), (mx, y + h + 3), (205, 210, 220), 1, cv2.LINE_AA)

def arc(img, center, radius, frac, color, thickness=5, bg_col=(52, 56, 64)):
    c = (int(center[0]), int(center[1]))
    cv2.ellipse(img, c, (radius, radius), 0, 0, 360, bg_col, max(1, thickness - 3), cv2.LINE_AA)
    frac = clamp(frac, 0, 1)
    if frac > 0.004:
        a2 = -90 + 360 * frac
        cv2.ellipse(img, c, (radius, radius), 0, -90, a2, color, thickness, cv2.LINE_AA)
        ang = math.radians(a2)
        cv2.circle(img, (c[0] + int(radius * math.cos(ang)), c[1] + int(radius * math.sin(ang))),
                   thickness // 2 + 1, color, -1, cv2.LINE_AA)

def dashline(img, p1, p2, color, thickness=1, dash=5, gap=4):
    x1, y1 = p1; x2, y2 = p2
    d = math.hypot(x2 - x1, y2 - y1)
    if d <= 0: return
    ux, uy = (x2 - x1) / d, (y2 - y1) / d
    s = 0.0
    while s < d:
        e = min(d, s + dash)
        cv2.line(img, (int(x1 + ux * s), int(y1 + uy * s)),
                     (int(x1 + ux * e), int(y1 + uy * e)), color, thickness, cv2.LINE_AA)
        s += dash + gap

# ---------------- icons ----------------
def icon_fan(img, cx, cy, s, color, t, spinning):
    ang = t * 300 if spinning else 12
    for i in range(3):
        a = math.radians(ang + i * 120)
        ex = cx + math.cos(a) * s * 0.55
        ey = cy + math.sin(a) * s * 0.55
        cv2.ellipse(img, (int(ex), int(ey)), (int(s * 0.52), int(s * 0.20)),
                    math.degrees(a), 0, 360, color, -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), max(2, int(s * 0.16)), color, -1, cv2.LINE_AA)

def icon_bulb(img, cx, cy, s, color, on):
    cv2.circle(img, (cx, cy - int(s * 0.15)), int(s * 0.48), color, 2, cv2.LINE_AA)
    rrect(img, (cx - int(s * 0.18), cy + int(s * 0.34)),
              (cx + int(s * 0.18), cy + int(s * 0.58)), color, 2, 2)
    if on:
        for a in (-60, -30, 0, 30, 60):
            r1, r2 = s * 0.66, s * 0.95
            ar = math.radians(a - 90)
            cv2.line(img, (int(cx + math.cos(ar) * r1), int(cy - s * 0.15 + math.sin(ar) * r1)),
                         (int(cx + math.cos(ar) * r2), int(cy - s * 0.15 + math.sin(ar) * r2)),
                         color, 2, cv2.LINE_AA)

def icon_sos(img, cx, cy, s, color):
    pts = np.array([[cx, cy - int(s * 0.6)], [cx + int(s * 0.62), cy + int(s * 0.45)],
                    [cx - int(s * 0.62), cy + int(s * 0.45)]], np.int32)
    cv2.polylines(img, [pts], True, color, 2, cv2.LINE_AA)
    cv2.line(img, (cx, cy - int(s * 0.28)), (cx, cy + int(s * 0.08)), color, 3, cv2.LINE_AA)
    cv2.circle(img, (cx, cy + int(s * 0.28)), 2, color, -1, cv2.LINE_AA)

def icon_phone(img, cx, cy, s, color):
    rrect(img, (cx - int(s * 0.34), cy - int(s * 0.6)),
              (cx + int(s * 0.34), cy + int(s * 0.6)), color, int(s * 0.18), 2)
    cv2.line(img, (cx - int(s * 0.12), cy - int(s * 0.42)),
                 (cx + int(s * 0.12), cy - int(s * 0.42)), color, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy + int(s * 0.4)), 2, color, -1, cv2.LINE_AA)

def icon_eye(img, cx, cy, s, color):
    cv2.ellipse(img, (cx, cy), (int(s), int(s * 0.55)), 0, 0, 360, color, 2, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), int(s * 0.28), color, -1, cv2.LINE_AA)

def icon_mouth(img, cx, cy, s, color):
    cv2.ellipse(img, (cx, cy), (int(s * 0.95), int(s * 0.6)), 0, 0, 360, color, 2, cv2.LINE_AA)
    cv2.line(img, (cx - int(s * 0.9), cy), (cx + int(s * 0.9), cy), color, 1, cv2.LINE_AA)

# ============================================================
# SERIAL LINK (background thread, auto-reconnect)
# ============================================================
class SerialLink(threading.Thread):
    def __init__(self, port, baud):
        super().__init__(daemon=True)
        self.port, self.baud = port, baud
        self.q = queue.Queue()
        self.status = "OFFLINE"
        self.on_line = lambda s: None
        self.alive = True

    def send(self, cmd):
        self.q.put(cmd)

    def run(self):
        while self.alive:
            try:
                while not self.q.empty():
                    self.q.get_nowait()
                with pyserial.Serial(self.port, self.baud, timeout=0.1) as ser:
                    ser.setDTR(False)          # same trick as the passing test
                    ser.setRTS(False)
                    self.status = "BOOT"
                    self.on_line("OK:HUB BOOTING…")

                    time.sleep(2.5)            # ESP32 auto-resets on port open
                    ser.reset_input_buffer()   # drop boot ROM garbage

                    # wait for hub to finish booting (max 8s)
                    t0 = time.time()
                    while time.time() - t0 < 8.0 and self.alive:
                        raw = ser.readline()
                        if raw:
                            line = raw.decode(errors="ignore").strip()
                            if line:
                                self.on_line(line)
                            if "HUB_READY" in line:
                                break

                    ser.reset_input_buffer()
                    ser.write(b"PING\n"); ser.flush()   # -> hub LED flash + PONG
                    self.status = "ONLINE"
                    self.on_line("OK:HUB LINKED")

                    while self.alive:
                        raw = ser.readline()
                        if raw:
                            line = raw.decode(errors="ignore").strip()
                            if line:
                                self.on_line(line)
                        try:
                            cmd = self.q.get_nowait()
                            ser.write((cmd + "\n").encode())
                        except queue.Empty:
                            pass
            except Exception:
                self.status = "OFFLINE"
                time.sleep(2.0)

LINK = None
hub_ack, hub_ack_time = "", 0.0

def handle_hub_line(line):
    global hub_ack, hub_ack_time
    hub_ack, hub_ack_time = line, time.time()
    print("[HUB]", line)
    if line.startswith("ERR"):
        set_status(line)

def send_cmd(cmd):
    if LINK and LINK.status == "ONLINE":
        LINK.send(cmd)
    else:
        print("[SIM] →", cmd)

# ============================================================
# MEDIAPIPE HELPERS (original logic)
# ============================================================
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
UPPER_LIP, LOWER_LIP = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
NOSE = 1

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

def eye_aspect_ratio(lm, idx):
    v1 = distance(lm[idx[1]], lm[idx[5]])
    v2 = distance(lm[idx[2]], lm[idx[4]])
    h  = distance(lm[idx[0]], lm[idx[3]])
    return 0.0 if h == 0 else (v1 + v2) / (2.0 * h)

def mouth_ratio(lm):
    v = distance(lm[UPPER_LIP], lm[LOWER_LIP])
    h = distance(lm[MOUTH_LEFT], lm[MOUTH_RIGHT])
    return 0.0 if h == 0 else v / h

# ============================================================
# APP STATE
# ============================================================
app_state = STATE_NORMAL
state_start = time.time()
state_grace_until = 0.0

mouth_open, mouth_open_start = False, None
mouth_cycles, gesture_start_time, last_cycle_time = 0, None, 0

blink_start = None
blink_count, last_blink_time = 0, 0

prev_nose_x, prev_nose_y = None, None
head_movement = 0.0

selected_index = 0
last_menu_trigger = 0

fan_on = light_on = False
alert_until = 0.0
last_action = "—"

status_message, status_message_time = "SYSTEM READY", time.time()

left_ear = right_ear = average_ear = mouth_value = 0.0
both_closed = False
mouth_hist = []
fps = 0.0
RT = {"x": 400, "y": 200, "x2": 640, "y2": 460}
VIG = None
W, H = CAMERA_WIDTH, CAMERA_HEIGHT

forced_blink = False   # debug key 'b'
forced_cycle = False   # debug key 'c'

# ---------------- AI GESTURE LAYER ----------------
AI = GestureAI(mode="verify")      # becomes "off" automatically until a model exists
AI_REC_KEYS = {ord("1"): "rest", ord("2"): "blink",
               ord("3"): "mouth_cycle", ord("4"): "noise"}
AI_GESTURE_WINDOW = 3.5            # seconds allowed for 2 mouth cycles in PRIMARY mode


def set_status(message):
    global status_message, status_message_time
    status_message, status_message_time = message, time.time()
    print("[STATUS]", message)


def menu_labels():
    return [
        ("TURN OFF FAN" if fan_on else "TURN ON FAN", "fan"),
        ("TURN OFF LIGHT" if light_on else "TURN ON LIGHT", "light"),
        ("SEND SOS ALERT", "sos"),
        ("CALL FAMILY", "call"),
    ]


def change_state(new_state):
    """Single entry point for every state change — resets all detectors."""
    global app_state, state_start, state_grace_until
    global mouth_open, mouth_open_start, mouth_cycles, gesture_start_time
    global blink_start
    app_state = new_state
    state_start = time.time()
    state_grace_until = state_start + STATE_GRACE
    mouth_open = False
    mouth_open_start = None
    mouth_cycles = 0
    gesture_start_time = None
    blink_start = None


def next_menu_item():
    global selected_index
    selected_index = (selected_index + 1) % len(MENU_ITEMS)
    set_status("SELECTED: " + menu_labels()[selected_index][0])


def execute_selected_item():
    global fan_on, light_on, alert_until, last_action
    kind = MENU_ITEMS[selected_index]
    print("=" * 46, "\nEXECUTING:", kind, "\n" + "=" * 46)

    if kind == "FAN":
        fan_on = not fan_on
        send_cmd("FAN_ON" if fan_on else "FAN_OFF")
        last_action = "FAN → " + ("ON" if fan_on else "OFF")
    elif kind == "LIGHT":
        light_on = not light_on
        send_cmd("LIGHT_ON" if light_on else "LIGHT_OFF")
        last_action = "LIGHT → " + ("ON" if light_on else "OFF")
    elif kind == "SOS":
        send_cmd("SOS")
        alert_until = time.time() + 4.5
        last_action = "SOS ALERT SENT"
    elif kind == "CALL":
        send_cmd("CALL")
        last_action = "CALLING FAMILY…"

    set_status("DONE: " + last_action)

# ============================================================
# HUD COMPONENTS
# ============================================================
def draw_topbar(frame, t):
    glass(frame, 0, 0, W, 62, r=0, tint=(9, 11, 15), alpha=0.55)
    cv2.line(frame, (0, 62), (W, 62), faded(ACCENT, 0.35), 1, cv2.LINE_AA)
    txt(30, 31, "HANDS-FREE CONTROLLER", 20, WHITE, True, "lm")

    sc = STATE_COLORS.get(app_state, ACCENT)
    cw, ch = 190, 34
    sx, sy = W - 24 - cw, 14
    rrect(frame, (sx, sy), (sx + cw, sy + ch), faded(sc, 0.18, PANEL), ch // 2, -1)
    rrect(frame, (sx, sy), (sx + cw, sy + ch), faded(sc, 0.75), ch // 2, 1)
    cv2.circle(frame, (sx + 16, sy + ch // 2), 4,
               faded(sc, 0.45 + 0.55 * pulse(t, 1.2)), -1, cv2.LINE_AA)
    txt(sx + 30, sy + ch // 2, TOPBAR_LABEL[app_state], 14, sc, True, "lm")

    txt(sx - 24, sy + ch // 2, time.strftime("%H:%M:%S"), 15, GRAY, False, "rm")

    if LINK:
        hub_col = GREEN if LINK.status == "ONLINE" else RED
        hub_txt = "HUB · USB" if LINK.status == "ONLINE" else "HUB · OFFLINE"
    else:
        hub_col, hub_txt = DGRAY, "HUB · SIM"
    hw = 150
    hx = sx - 24 - hw
    rrect(frame, (hx, sy), (hx + hw, sy + ch), faded(hub_col, 0.14, PANEL), ch // 2, -1)
    rrect(frame, (hx, sy), (hx + hw, sy + ch), faded(hub_col, 0.6), ch // 2, 1)
    cv2.circle(frame, (hx + 16, sy + ch // 2), 4,
               faded(hub_col, 0.45 + 0.55 * pulse(t, 1.6)), -1, cv2.LINE_AA)
    txt(hx + 30, sy + ch // 2, hub_txt, 13, hub_col, True, "lm")


def draw_telemetry(frame, t):
    x, y, w, h = 20, 82, 330, 380
    glass(frame, x, y, w, h, tint=(12, 14, 19), alpha=0.68, border=faded(ACCENT, 0.28))
    txt(x + 20, y + 28, "TELEMETRY", 14, faded(WHITE, 0.85), True, "lm")
    cv2.circle(frame, (x + w - 26, y + 28), 4,
               faded(GREEN, 0.4 + 0.6 * pulse(t, 1.4)), -1, cv2.LINE_AA)
    cv2.line(frame, (x + 16, y + 46), (x + w - 16, y + 46), (58, 62, 70), 1, cv2.LINE_AA)

    def row(yy, label, val, vmax, color, markers=()):
        txt(x + 20, yy, label, 12, GRAY, False, "lm")
        meter(frame, x + 104, yy - 6, 128, 12, val / vmax, color, markers)
        txt(x + w - 18, yy, f"{val:.3f}", 12, WHITE, False, "rm")

    row(y + 72,  "EAR · L", smooth("eL", left_ear),    0.5, ACCENT)
    row(y + 102, "EAR · R", smooth("eR", right_ear),   0.5, ACCENT)
    row(y + 132, "EYE AVG", smooth("eA", average_ear), 0.5,
        GREEN if not both_closed else RED,
        markers=[EYE_OPEN_THRESHOLD / 0.5, EYE_CLOSED_THRESHOLD / 0.5])
    row(y + 162, "MOUTH",   smooth("mo", mouth_value), 0.6, AMBER,
        markers=[MOUTH_CLOSE_THRESHOLD / 0.6, MOUTH_OPEN_THRESHOLD / 0.6])
    row(y + 192, "HEAD",    smooth("hd", head_movement, 0.4), 0.35,
        GREEN if head_movement <= MAX_HEAD_MOVEMENT else RED,
        markers=[MAX_HEAD_MOVEMENT / 0.35])

    ec = GREEN if not both_closed else RED
    etxt = "EYES OPEN" if not both_closed else "EYES CLOSED"
    rrect(frame, (x + 20, y + 216), (x + 138, y + 244), faded(ec, 0.14, PANEL), 14, -1)
    rrect(frame, (x + 20, y + 216), (x + 138, y + 244), faded(ec, 0.55), 14, 1)
    txt(x + 30, y + 230, etxt, 12, ec, True, "lm")
    rrect(frame, (x + 150, y + 216), (x + 262, y + 244), PANEL, 14, -1)
    rrect(frame, (x + 150, y + 216), (x + 262, y + 244), (60, 64, 72), 14, 1)
    txt(x + 160, y + 230, f"BLINKS {blink_count}", 12, WHITE, False, "lm")

    sx, sy, sw, sh = x + 20, y + 260, w - 40, 52
    rrect(frame, (sx, sy), (sx + sw, sy + sh), (14, 16, 22), 10, -1)
    rrect(frame, (sx, sy), (sx + sw, sy + sh), (52, 56, 64), 10, 1)
    txt(sx + 10, sy + 12, "MOUTH TRACE", 10, DGRAY, False, "lm")
    ty = sy + sh - int((MOUTH_OPEN_THRESHOLD / 0.6) * (sh - 8)) - 4
    dashline(frame, (sx + 4, ty), (sx + sw - 4, ty), faded(AMBER, 0.5), 1, 4, 4)
    if len(mouth_hist) > 1:
        pts = []
        for i, v in enumerate(mouth_hist):
            px = sx + 4 + int(i / (len(mouth_hist) - 1) * (sw - 8))
            py = sy + sh - 4 - int(clamp(v / 0.6, 0, 1) * (sh - 12))
            pts.append((px, py))
        cv2.polylines(frame, [np.array(pts, np.int32)], False, AMBER, 1, cv2.LINE_AA)

    rc = (x + 46, y + h - 40)
    frac = mouth_cycles / REQUIRED_CYCLES
    col = AMBER if frac < 1 else GREEN
    arc(frame, rc, 20, frac, faded(col, 0.6 + 0.4 * pulse(t, 0.9)), 4)
    txt(rc[0], rc[1], f"{mouth_cycles}", 15, col, True, "mm")
    txt(x + 80, y + h - 48, "MENU CYCLES", 11, GRAY, False, "lm")
    txt(x + 80, y + h - 28, "OPEN–CLOSE MOUTH ×2", 10, DGRAY, False, "lm")

    if LINK:
        hc = GREEN if LINK.status == "ONLINE" else RED
        hstat = "HUB ONLINE" if LINK.status == "ONLINE" else "HUB OFFLINE"
    else:
        hc, hstat = DGRAY, "HUB SIM MODE"
    ack_a = clamp(1 - (t - hub_ack_time) / 5.0, 0, 1) if hub_ack else 0
    txt(x + w - 18, y + h - 48, hstat, 11, hc, True, "rm")
    if ack_a > 0:
        txt(x + w - 18, y + h - 28, hub_ack[:26], 10, faded(GRAY, ack_a), False, "rm")


def draw_reticle(frame, t, color, face_box):
    x0, y0, x1, y1 = face_box
    RT["x"]  = lerp(RT["x"],  x0, 0.2); RT["y"]  = lerp(RT["y"],  y0, 0.2)
    RT["x2"] = lerp(RT["x2"], x1, 0.2); RT["y2"] = lerp(RT["y2"], y1, 0.2)
    rx0, ry0, rx1, ry1 = int(RT["x"]), int(RT["y"]), int(RT["x2"]), int(RT["y2"])
    L = int(min(44, (rx1 - rx0) / 3, (ry1 - ry0) / 3))
    a = 0.65 + 0.35 * pulse(t, 1.8)
    c_soft, c_full = faded(color, a * 0.35), faded(color, a)
    corners = [((rx0, ry0), (1, 1)), ((rx1, ry0), (-1, 1)),
               ((rx0, ry1), (1, -1)), ((rx1, ry1), (-1, -1))]
    for (cx, cy), (dx, dy) in corners:
        cv2.line(frame, (cx, cy), (cx + dx * L, cy), c_soft, 5, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx + dx * L, cy), c_full, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * L), c_soft, 5, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * L), c_full, 2, cv2.LINE_AA)
    txt(rx0, ry0 - 12, "USER · LOCKED", 11, faded(color, a), True, "lb")

    if app_state == STATE_NORMAL and mouth_value > 0:
        mx = (rx0 + rx1) // 2
        ok_window = mouth_open and MIN_OPEN_TIME <= t - (mouth_open_start or t) <= MAX_OPEN_TIME
        mc = GREEN if ok_window else ACCENT
        meter(frame, mx - 55, ry1 + 16, 110, 8, mouth_value / 0.6, mc,
              markers=[MOUTH_OPEN_THRESHOLD / 0.6])


def draw_decision(frame, t, title, subtitle, timeout, color, extra=None):
    """Shared YES/NO card for 'open menu?' and 'any other operation?'."""
    frame[:] = cv2.addWeighted(frame, 0.40, np.full(frame.shape, (8, 10, 14), np.uint8), 0.60, 0)
    cw, ch = 680, 330
    cx, cy = (W - cw) // 2, (H - ch) // 2
    bc = faded(color, 0.55 + 0.45 * pulse(t, 1.1))
    glass(frame, cx, cy, cw, ch, tint=(13, 15, 20), alpha=0.82, border=bc)

    remaining = clamp(1 - (t - state_start) / timeout, 0, 1)
    ring_c = (cx + 92, cy + 112)
    arc(frame, ring_c, 52, remaining, color, 6)
    icon_eye(frame, ring_c[0], ring_c[1], 24, WHITE)
    secs = max(0, int(timeout - (t - state_start)))
    txt(ring_c[0], ring_c[1] + 78, f"{secs}s", 15, faded(GRAY, 0.9), True, "mm")

    tx = cx + 195
    txt(tx, cy + 66, title, 28, WHITE, True, "lm")
    cv2.line(frame, (tx, cy + 92), (tx + int(250 * pulse(t, 2.4)), cy + 92),
             faded(color, 0.85), 2, cv2.LINE_AA)
    txt(tx, cy + 122, subtitle, 15, GRAY, False, "lm")

    y1 = cy + 152
    rrect(frame, (tx, y1), (tx + 390, y1 + 44), faded(GREEN, 0.13, PANEL), 14, -1)
    rrect(frame, (tx, y1), (tx + 390, y1 + 44), faded(GREEN, 0.75), 14, 1)
    icon_eye(frame, tx + 30, y1 + 22, 13, GREEN)
    txt(tx + 56, y1 + 22, "BLINK ONCE", 16, GREEN, True, "lm")
    txt(tx + 362, y1 + 22, "=  YES", 16, WHITE, True, "rm")

    y2 = y1 + 56
    rrect(frame, (tx, y2), (tx + 390, y2 + 44), faded(RED, 0.13, PANEL), 14, -1)
    rrect(frame, (tx, y2), (tx + 390, y2 + 44), faded(RED, 0.75), 14, 1)
    icon_mouth(frame, tx + 30, y2 + 22, 13, RED)
    txt(tx + 56, y2 + 22, "MOUTH OPEN–CLOSE", 16, RED, True, "lm")
    txt(tx + 362, y2 + 22, "=  NO", 16, WHITE, True, "rm")

    if extra:
        txt(tx, cy + ch - 32, "LAST ACTION:  " + extra, 14, faded(AMBER, 0.95), True, "lm")


sel_y = 0.0
def draw_menu(frame, t):
    global sel_y
    frame[:] = cv2.addWeighted(frame, 0.34, np.full(frame.shape, (8, 10, 14), np.uint8), 0.66, 0)
    labels = menu_labels()
    rows_h, header_h, footer_h = 76, 96, 56
    cw = 600
    ch = header_h + rows_h * len(labels) + footer_h
    cx, cy = (W - cw) // 2, (H - ch) // 2 - 8
    glass(frame, cx, cy, cw, ch, tint=(12, 14, 19), alpha=0.85, border=faded(ACCENT, 0.55))

    txt(cx + 34, cy + 44, "CONTROL MENU", 27, WHITE, True, "lm")
    txt(cx + cw - 34, cy + 44, "GESTURE NAVIGATION", 12, GRAY, False, "rm")
    uw = int((cw - 68) * (0.5 + 0.5 * pulse(t, 2.6)))
    cv2.line(frame, (cx + 34, cy + 68), (cx + 34 + uw, cy + 68), faded(ACCENT, 0.85), 2, cv2.LINE_AA)

    target_y = cy + header_h + selected_index * rows_h + rows_h // 2
    sel_y = lerp(sel_y, target_y, 0.22)

    for i, (label, kind) in enumerate(labels):
        ry = cy + header_h + i * rows_h + rows_h // 2
        if i != len(labels) - 1:
            cv2.line(frame, (cx + 24, ry + rows_h // 2), (cx + cw - 24, ry + rows_h // 2),
                     (46, 50, 58), 1, cv2.LINE_AA)
        selected = (i == selected_index)
        icol = {"fan": ACCENT, "light": AMBER, "sos": RED, "call": GREEN}[kind]
        icx = cx + 66
        if kind == "fan":     icon_fan(frame, icx, ry, 17, faded(icol, 1 if selected else 0.45), t, fan_on)
        elif kind == "light": icon_bulb(frame, icx, ry, 17, faded(icol, 1 if selected else 0.45), light_on)
        elif kind == "sos":   icon_sos(frame, icx, ry, 16, faded(icol, 1 if selected else 0.45))
        else:                 icon_phone(frame, icx, ry, 16, faded(icol, 1 if selected else 0.45))

        lcol = WHITE if selected else faded(WHITE, 0.45)
        txt(cx + 104, ry, label, 20, lcol, selected, "lm")

        if kind in ("fan", "light"):
            on = fan_on if kind == "fan" else light_on
            s_txt, s_col = ("ON", GREEN) if on else ("OFF", DGRAY)
        elif kind == "sos":
            s_txt, s_col = "ALERT", RED
        else:
            s_txt, s_col = "DIAL", ACCENT
        scx, scw = cx + cw - 118, 84
        rrect(frame, (scx, ry - 13), (scx + scw, ry + 13), faded(s_col, 0.14, PANEL), 13, -1)
        txt(scx + scw // 2, ry, s_txt, 12, faded(s_col, 1 if selected else 0.5), True, "mm")

    if 0 <= sel_y - cy - header_h <= rows_h * len(labels):
        rrect(frame, (cx + 14, int(sel_y) - 30), (cx + cw - 14, int(sel_y) + 30),
              faded(ACCENT, 0.10, PANEL), 16, -1)
        rrect(frame, (cx + 14, int(sel_y) - 30), (cx + cw - 14, int(sel_y) + 30),
              faded(ACCENT, 0.8), 16, 1)
        cv2.line(frame, (cx + 14, int(sel_y) - 22), (cx + 14, int(sel_y) + 22), ACCENT, 4, cv2.LINE_AA)

    fy = cy + ch - footer_h // 2
    cv2.line(frame, (cx + 24, fy - footer_h // 2 + 6), (cx + cw - 24, fy - footer_h // 2 + 6),
             (46, 50, 58), 1, cv2.LINE_AA)
    txt(cx + 34, fy, "MOUTH OPEN–CLOSE  =  NEXT      BLINK  =  SELECT", 14, GRAY, False, "lm")
    txt(cx + cw - 34, fy, f"{selected_index + 1} / {len(labels)}", 13, faded(ACCENT, 0.9), True, "rm")


def draw_toast(frame, t):
    if t - status_message_time > 3.0:
        return
    age = t - status_message_time
    a = clamp(age / 0.15, 0, 1) * clamp((3.0 - age) / 0.4, 0, 1)
    msg = status_message
    col = RED if "SOS" in msg else (GREEN if "DONE" in msg or "SENT" in msg else ACCENT)
    cw = 560
    cx, cy = (W - cw) // 2, H - 96
    glass(frame, cx, cy, cw, 46, r=23, tint=(12, 14, 19), alpha=0.72)
    rrect(frame, (cx, cy), (cx + cw - 1, cy + 45), faded(col, a * 0.85), 23, 1)
    cv2.circle(frame, (cx + 26, cy + 23), 5,
               faded(col, a * (0.5 + 0.5 * pulse(t, 0.8))), -1, cv2.LINE_AA)
    txt(cx + cw // 2, cy + 23, msg[:44], 16, faded(WHITE, max(a, 0.2)), True, "mm")


def draw_alert(frame, t):
    if t > alert_until:
        return
    k = pulse(t, 0.55)
    col = faded(RED, 0.35 + 0.65 * k)
    cv2.rectangle(frame, (0, 0), (W - 1, 6), col, -1)
    cv2.rectangle(frame, (0, H - 6), (W - 1, H - 1), col, -1)
    cv2.rectangle(frame, (0, 0), (5, H - 1), col, -1)
    cv2.rectangle(frame, (W - 5, 0), (W - 1, H - 1), col, -1)
    bw, bh = 430, 54
    bx, by = (W - bw) // 2, 78
    rrect(frame, (bx, by), (bx + bw, by + bh), faded(RED, 0.85), 14, -1)
    rrect(frame, (bx, by), (bx + bw, by + bh), RED, 14, 2)
    txt(bx + bw // 2, by + bh // 2 - 8, "EMERGENCY  SOS", 22, WHITE, True, "mm")
    txt(bx + bw // 2, by + bh - 12, "ALERT SIGNAL TRANSMITTED", 10, faded(WHITE, 0.8), False, "mm")


def draw_no_face(frame, t):
    frame[:] = cv2.addWeighted(frame, 0.55, np.full(frame.shape, (8, 10, 14), np.uint8), 0.45, 0)
    for k in range(3):
        ph = (t * 0.55 + k / 3) % 1.0
        r = int(24 + ph * 60)
        cv2.circle(frame, (W // 2, H // 2 - 30), r, faded(RED, (1 - ph) * 0.6), 2, cv2.LINE_AA)
    cv2.circle(frame, (W // 2, H // 2 - 30), 5, faded(RED, 0.9), -1, cv2.LINE_AA)
    txt(W // 2, H // 2 + 70, "SEARCHING FOR USER", 26, WHITE, True, "mm")
    txt(W // 2, H // 2 + 104, "Please stay centered in the frame", 15, GRAY, False, "mm")


def make_vignette(w, h):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2)
    m = np.clip(1 - 0.5 * np.clip(d - 0.55, 0, 1) ** 1.4, 0.5, 1)
    return m[..., None]

# ============================================================
# MAIN
# ============================================================
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("ERROR: Camera could not be opened."); exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, CAMERA_WIDTH, CAMERA_HEIGHT)

if SERIAL_ENABLED and SERIAL_AVAILABLE:
    LINK = SerialLink(SERIAL_PORT, SERIAL_BAUD)
    LINK.on_line = handle_hub_line
    LINK.start()
else:
    print("[SIM MODE] Commands will print to console only.")

t_prev = time.time()

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                           min_detection_confidence=0.6,
                           min_tracking_confidence=0.6) as face_mesh:

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Camera frame failed."); break

        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]
        if VIG is None or VIG.shape[0] != H:
            VIG = make_vignette(W, H)
        frame = (frame.astype(np.float32) * VIG).astype(np.uint8)

        t_now = time.time()
        dt = t_now - t_prev
        t_prev = t_now
        fps = lerp(fps, 1.0 / max(dt, 1e-4), 0.08)
        current_time = t_now
        state_color = STATE_COLORS.get(app_state, ACCENT)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            mouth_hist.clear()
            prev_nose_x, prev_nose_y = None, None
            AI.clear()
            draw_no_face(frame, current_time)
        else:
            face = result.multi_face_landmarks[0]
            landmarks = face.landmark

            mp_drawing.draw_landmarks(frame, face, mp_face_mesh.FACEMESH_CONTOURS, None,
                mp_drawing.DrawingSpec(color=(90, 190, 255), thickness=1, circle_radius=0))
            mp_drawing.draw_landmarks(frame, face, mp_face_mesh.FACEMESH_LIPS, None,
                mp_drawing.DrawingSpec(color=(110, 255, 170), thickness=1, circle_radius=0))
            mp_drawing.draw_landmarks(frame, face, mp_face_mesh.FACEMESH_LEFT_EYE, None,
                mp_drawing.DrawingSpec(color=(120, 220, 255), thickness=1, circle_radius=0))
            mp_drawing.draw_landmarks(frame, face, mp_face_mesh.FACEMESH_RIGHT_EYE, None,
                mp_drawing.DrawingSpec(color=(120, 220, 255), thickness=1, circle_radius=0))

            lm_np = np.array([(l.x, l.y) for l in landmarks])
            x0 = clamp(int(lm_np[:, 0].min() * W) - 34, 0, W - 1)
            x1 = clamp(int(lm_np[:, 0].max() * W) + 34, 0, W - 1)
            y0 = clamp(int(lm_np[:, 1].min() * H) - 40, 0, H - 1)
            y1 = clamp(int(lm_np[:, 1].max() * H) + 30, 0, H - 1)

            mouth_value = mouth_ratio(landmarks)
            left_ear  = eye_aspect_ratio(landmarks, LEFT_EYE)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE)
            average_ear = (left_ear + right_ear) / 2.0

            # ---------- HEAD MOVEMENT (original logic) ----------
            nose = landmarks[NOSE]
            if prev_nose_x is not None:
                head_movement = abs(nose.x - prev_nose_x) + abs(nose.y - prev_nose_y)
            prev_nose_x, prev_nose_y = nose.x, nose.y

            both_closed = (left_ear < EYE_CLOSED_THRESHOLD and
                           right_ear < EYE_CLOSED_THRESHOLD)
            both_open   = (left_ear > EYE_OPEN_THRESHOLD and
                           right_ear > EYE_OPEN_THRESHOLD)

            mouth_hist.append(mouth_value)
            if len(mouth_hist) > 90:
                mouth_hist.pop(0)

            draw_reticle(frame, current_time, state_color, (x0, y0, x1, y1))

            _mc_before, _gs_before = mouth_cycles, gesture_start_time

            # ============ BLINK DETECTION (your original logic) ============
            new_blink = False
            if both_closed and blink_start is None:
                blink_start = current_time
            if blink_start is not None and both_open:
                blink_duration = current_time - blink_start
                if MIN_BLINK_TIME <= blink_duration <= MAX_BLINK_TIME:
                    new_blink = True
                    blink_count += 1
                    last_blink_time = current_time
                    print("BLINK DETECTED:", round(blink_duration, 3), "s")
                blink_start = None

            # ============ MOUTH CYCLE DETECTION (your original logic) ============
            mouth_cycle_event = False
            if not mouth_open:
                if mouth_value >= MOUTH_OPEN_THRESHOLD:
                    mouth_open = True
                    mouth_open_start = current_time
                    if gesture_start_time is None:
                        gesture_start_time = current_time
                        mouth_cycles = 0
            else:
                open_duration = current_time - mouth_open_start
                if mouth_value <= MOUTH_CLOSE_THRESHOLD:
                    if MIN_OPEN_TIME <= open_duration <= MAX_OPEN_TIME:
                        if mouth_cycles == 0 or \
                           (current_time - last_cycle_time) >= MIN_BETWEEN_CYCLES:
                            mouth_cycle_event = True
                            mouth_cycles += 1
                            last_cycle_time = current_time
                            print("Mouth cycle:", mouth_cycles)
                    mouth_open = False
                    mouth_open_start = None

            # gesture timeout (only relevant in NORMAL)
            if gesture_start_time is not None and \
               current_time - gesture_start_time > MAX_GESTURE_TIME:
                mouth_cycles = 0
                gesture_start_time = None

            # ============ AI GESTURE LAYER (gesture_ai.py) ============
            ai_event = AI.update(current_time, left_ear, right_ear,
                                 mouth_value, head_movement)
            if AI.rec_label:
                # data-collection mode: never open menus / send hub commands
                mouth_cycles = 0
                if app_state != STATE_NORMAL:
                    change_state(STATE_NORMAL)
            elif AI.mode == "verify":
                # thresholds detect, AI confirms or vetoes
                if new_blink and not AI.confirms("blink"):
                    new_blink = False
                    set_status("BLINK REJECTED BY AI")
                if mouth_cycle_event and not AI.confirms("mouth_cycle"):
                    mouth_cycle_event = False
                    mouth_cycles = max(0, mouth_cycles - 1)
                    set_status("MOUTH GESTURE REJECTED BY AI")
            elif AI.mode == "primary":
                # AI detects the gestures; threshold results are ignored
                mouth_cycles, gesture_start_time = _mc_before, _gs_before
                new_blink = (ai_event == "blink")
                mouth_cycle_event = (ai_event == "mouth_cycle")
                if mouth_cycle_event:
                    if gesture_start_time is None:
                        gesture_start_time = current_time
                        mouth_cycles = 0
                    mouth_cycles += 1
                if gesture_start_time is not None and \
                   current_time - gesture_start_time > AI_GESTURE_WINDOW:
                    mouth_cycles = 0
                    gesture_start_time = None

            # debug key injection
            if forced_blink:
                new_blink = True
                forced_blink = False
            if forced_cycle:
                mouth_cycle_event = True
                forced_cycle = False

            grace_ok = current_time >= state_grace_until

            # =================================================
            # STATE MACHINE  —  your exact flow
            # =================================================
            if app_state == STATE_NORMAL:
                # 2 mouth cycles -> ask YES/NO to open menu
                if mouth_cycles >= REQUIRED_CYCLES:
                    if head_movement > MAX_HEAD_MOVEMENT:
                        set_status("GESTURE REJECTED — HEAD MOVED")
                    elif current_time - last_menu_trigger >= MENU_COOLDOWN:
                        last_menu_trigger = current_time
                        change_state(STATE_YESNO)
                        set_status("OPEN MENU? — BLINK = YES")
                    else:
                        set_status("PLEASE WAIT…")
                    mouth_cycles = 0
                    gesture_start_time = None

            elif app_state == STATE_YESNO:
                if grace_ok:
                    if new_blink:                       # YES -> open menu
                        selected_index = 0
                        change_state(STATE_MENU)
                        set_status("MENU OPENED — MOUTH = NEXT")
                    elif mouth_cycle_event:             # NO -> cancel
                        change_state(STATE_NORMAL)
                        set_status("CANCELLED — MENU NOT OPENED")
                if current_time - state_start > YESNO_TIMEOUT:
                    change_state(STATE_NORMAL)
                    set_status("TIMED OUT — STILL TRACKING")

            elif app_state == STATE_MENU:
                if grace_ok:
                    if new_blink:                       # SELECT item
                        execute_selected_item()
                        change_state(STATE_ASK_MORE)
                        set_status("ANY OTHER OPERATION?")
                    elif mouth_cycle_event:             # NEXT item
                        next_menu_item()

            elif app_state == STATE_ASK_MORE:
                if grace_ok:
                    if new_blink:                       # YES -> back to menu
                        change_state(STATE_MENU)
                        set_status("CONTINUE — PICK NEXT ACTION")
                    elif mouth_cycle_event:             # NO -> exit
                        change_state(STATE_NORMAL)
                        set_status("MENU CLOSED")
                if current_time - state_start > ASK_MORE_TIMEOUT:
                    change_state(STATE_NORMAL)
                    set_status("MENU CLOSED — TIMEOUT")

        # ============ HUD LAYERS ============
        if app_state == STATE_YESNO:
            draw_decision(frame, current_time,
                          "OPEN CONTROL MENU?",
                          "You performed the menu gesture — confirm:",
                          YESNO_TIMEOUT, AMBER)
        elif app_state == STATE_MENU:
            draw_menu(frame, current_time)
        elif app_state == STATE_ASK_MORE:
            draw_decision(frame, current_time,
                          "ANY OTHER OPERATION?",
                          "Action completed — continue or finish:",
                          ASK_MORE_TIMEOUT, GREEN, extra=last_action)

        draw_alert(frame, current_time)
        draw_topbar(frame, current_time)
        draw_telemetry(frame, current_time)
        draw_toast(frame, current_time)

        if AI.rec_label:
            txt(20, 486, f"REC · {AI.rec_label.upper()} · {AI.rows_logged} windows",
                14, RED, True, "lm")
        else:
            _lab = f"{AI.label} {AI.probs.get(AI.label, 0):.2f}" if AI.label else "-"
            txt(20, 486, f"AI · {AI.mode.upper()} · {_lab}", 14,
                GREEN if AI.mode != "off" else DGRAY, True, "lm")
        txt(20, H - 46, "1 REST   2 BLINK   3 MOUTH   4 NOISE  =  REC (press again to stop)"
                        "    0 STOP    A AI MODE", 12, DGRAY, False, "lm")
        txt(20, H - 24, "Q · QUIT    M · MENU    B · SIM BLINK    C · SIM CYCLE    T · TEST SOS",
            12, DGRAY, False, "lm")
        txt(W - 20, H - 24, f"{fps:.1f} FPS", 12, DGRAY, False, "rm")

        frame = flush_text(frame)
        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break
        elif key == ord("m"):
            if app_state == STATE_MENU:
                change_state(STATE_NORMAL); set_status("MENU CLOSED")
            else:
                selected_index = 0
                change_state(STATE_MENU); set_status("MENU OPENED (KEY)")
        elif key == ord("b"):
            forced_blink = True
        elif key == ord("c"):
            forced_cycle = True
        elif key in AI_REC_KEYS:
            _lab = AI_REC_KEYS[key]
            if AI.rec_label == _lab:
                AI.stop_recording(); set_status("REC STOPPED")
            else:
                AI.start_recording(_lab); set_status("REC: " + _lab.upper())
        elif key == ord("0"):
            AI.stop_recording(); set_status("REC STOPPED")
        elif key == ord("a"):
            set_status("AI MODE: " + AI.cycle_mode().upper())
        elif key == ord("t"):
            send_cmd("SOS")
            alert_until = time.time() + 4.5
            set_status("!!! SOS TEST SENT !!!")

AI.stop_recording()
cap.release()
cv2.destroyAllWindows()
print("\n==========================================\nCONTROLLER STOPPED\n==========================================")

"""
gesture_ai.py  -  ML gesture classifier for the hands-free controller.

Camera features (EAR, mouth ratio, head movement) -> 1.2 s sliding window
-> Random Forest -> {rest, blink, mouth_cycle, noise}

Modes
  off      : AI disabled (pure threshold logic, your original behaviour)
  verify   : thresholds detect, AI confirms/vetoes (removes false triggers)
  primary  : AI detects the gestures itself (thresholds only drive telemetry)

Also records labelled training data to gesture_data.csv.
"""
import csv
import os
import time
from collections import deque

import numpy as np

LABELS = ["rest", "blink", "mouth_cycle", "noise"]
GESTURES = ("blink", "mouth_cycle")

SIGNALS = ["lear", "rear", "avg_ear", "mouth", "head"]
STATS = ["min", "max", "mean", "std", "first", "last", "tv"]
FEATURE_NAMES = [f"{s}_{k}" for s in SIGNALS for k in STATS]

WINDOW_SEC = 1.2      # length of the sliding window
N_STEPS = 24          # window is resampled to this many points (FPS independent)
REARM_SEC = {"blink": 1.5, "mouth_cycle": 1.5}   # cooldown after a gesture fires (one gesture = one event)
LOG_EVERY = 0.1       # seconds between logged training rows

MODEL_PATH = "gesture_model.pkl"
DATA_PATH = "gesture_data.csv"


def fidx(signal, stat):
    """Index of a feature in the feature vector."""
    return SIGNALS.index(signal) * len(STATS) + STATS.index(stat)


def window_features(win):
    """win: (N_STEPS, 5) array -> flat feature list."""
    f = []
    for k in range(win.shape[1]):
        s = win[:, k]
        f += [s.min(), s.max(), s.mean(), s.std(), s[0], s[-1],
              float(np.abs(np.diff(s)).sum())]
    return f


class GestureAI:
    MODES = ["off", "verify", "primary"]

    def __init__(self, mode="verify", model_path=MODEL_PATH,
                 data_path=DATA_PATH, fire_prob=0.6):
        self.buf = deque()
        self.hist = deque()
        self.clf = None
        self.classes = []
        self.fire_prob = fire_prob
        self.last_fire = -1e9
        self.last_pred_t = 0.0
        self.label = None
        self.probs = {}

        self.data_path = data_path
        self.rec_label = None
        self.session = 0
        self.rows_logged = 0
        self.last_log = 0.0
        self._fh = None
        self._w = None
        self._last_toggle = 0.0

        self.load(model_path)
        self.mode = mode if self.clf is not None else "off"

    # ------------------------------------------------------------ model
    def load(self, path):
        try:
            import joblib
            d = joblib.load(path)
            self.clf, self.classes = d["clf"], list(d["classes"])
            print(f"[AI] model loaded: {path}  classes={self.classes}")
            return True
        except Exception as e:
            self.clf = None
            print(f"[AI] no model loaded ({e}) - collect data and run train_model.py")
            return False

    def cycle_mode(self):
        """off -> verify -> primary -> off ... (skips AI modes if no model)."""
        if self.clf is None:
            self.mode = "off"
            return self.mode
        i = self.MODES.index(self.mode)
        self.mode = self.MODES[(i + 1) % len(self.MODES)]
        return self.mode

    # ------------------------------------------------------------ data in
    def clear(self):
        """Call when the face is lost so stale samples never enter a window."""
        self.buf.clear()
        self.hist.clear()
        self.label, self.probs = None, {}

    def push(self, t, lear, rear, mouth, head):
        self.buf.append((t, lear, rear, (lear + rear) / 2.0, mouth, head))
        while self.buf and t - self.buf[0][0] > WINDOW_SEC + 0.3:
            self.buf.popleft()

    def _window(self, t):
        if len(self.buf) < 5 or self.buf[-1][0] - self.buf[0][0] < WINDOW_SEC * 0.9:
            return None
        a = np.asarray(self.buf, dtype=np.float64)
        grid = np.linspace(t - WINDOW_SEC, t, N_STEPS)
        return np.stack([np.interp(grid, a[:, 0], a[:, k]) for k in range(1, 6)], axis=1)

    # ------------------------------------------------------------ main step
    def update(self, t, lear, rear, mouth, head):
        """Feed one frame. Returns 'blink' / 'mouth_cycle' / None (primary mode only)."""
        self.push(t, lear, rear, mouth, head)
        win = self._window(t)
        if win is None:
            return None
        feats = window_features(win)

        if self.rec_label and t - self.last_log >= LOG_EVERY:
            self._log(t, feats)

        if self.clf is None or self.mode == "off":
            return None

        event = None
        if t - self.last_pred_t >= 0.05:          # ~20 predictions / second
            p = self.clf.predict_proba([feats])[0]
            self.probs = dict(zip(self.classes, p))
            self.label = max(self.probs, key=self.probs.get)
            self.last_pred_t = t
            self.hist.append((t, self.probs))
            while self.hist and t - self.hist[0][0] > 1.0:
                self.hist.popleft()

            # one event per gesture: fire, then cool down while the gesture leaves the window
            if (self.label in GESTURES and self.probs[self.label] >= self.fire_prob
                    and t - self.last_fire >= REARM_SEC[self.label]):
                event = self.label
                self.last_fire = t

        return event if self.mode == "primary" else None

    def confirms(self, label, min_prob=0.35, horizon=0.6):
        """verify mode: did the model see `label` recently? (True if no model)"""
        if self.clf is None or not self.hist:
            return True
        t_now = self.hist[-1][0]
        return any(t_now - t <= horizon and p.get(label, 0.0) >= min_prob
                   for t, p in self.hist)

    # ------------------------------------------------------------ recording
    def start_recording(self, label):
        if time.time() - self._last_toggle < 1.0:     # ignore key-repeat / double press
            return False
        self._close()
        new = not os.path.exists(self.data_path) or os.path.getsize(self.data_path) == 0
        self._fh = open(self.data_path, "a", newline="")
        self._w = csv.writer(self._fh)
        if new:
            self._w.writerow(["session", "label"] + FEATURE_NAMES)
        self.session = int(time.time() * 1000)
        self.rec_label = label
        self._last_toggle = time.time()
        return True

    def stop_recording(self):
        if self.rec_label is not None and time.time() - self._last_toggle < 1.0:
            return False
        self._close()
        self._last_toggle = time.time()
        return True

    def _close(self):
        if self._fh:
            self._fh.close()
        self._fh = self._w = None
        self.rec_label = None

    def _log(self, t, feats):
        self._w.writerow([self.session, self.rec_label] +
                         [round(float(x), 5) for x in feats])
        self.last_log = t
        self.rows_logged += 1
        if self.rows_logged % 50 == 0:
            self._fh.flush()

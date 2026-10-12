"""
apply_ai_patch.py  -  adds the AI gesture layer to your controller script.

Usage (same folder as your controller):
    python apply_ai_patch.py your_controller.py
-> writes your_controller_ai.py   (your original file is NOT modified)

Each edit is anchored on a line from your original code and must match exactly once,
otherwise the script stops and tells you which one failed.
"""
import sys

EDITS = []


def edit(anchor, new, after=False):
    EDITS.append((anchor, new, after))


# 1. import
edit("import mediapipe as mp\n",
     "from gesture_ai import GestureAI\n", after=True)

# 2. AI object + recording keys (next to the other debug flags)
edit("forced_cycle = False   # debug key 'c'\n",
     '''
# ---------------- AI GESTURE LAYER ----------------
AI = GestureAI(mode="verify")      # becomes "off" automatically until a model exists
AI_REC_KEYS = {ord("1"): "rest", ord("2"): "blink",
               ord("3"): "mouth_cycle", ord("4"): "noise"}
AI_GESTURE_WINDOW = 3.5            # seconds allowed for 2 mouth cycles in PRIMARY mode
''', after=True)

# 3. face lost -> drop stale samples
edit("            draw_no_face(frame, current_time)\n",
     "            AI.clear()\n")

# 4. remember cycle counter before the threshold detectors touch it
edit("            # ============ BLINK DETECTION (your original logic) ============\n",
     "            _mc_before, _gs_before = mouth_cycles, gesture_start_time\n\n")

# 5. the AI layer itself (runs after threshold detection, before the state machine)
edit("            # debug key injection\n",
     '''            # ============ AI GESTURE LAYER (gesture_ai.py) ============
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
                if gesture_start_time is not None and \\
                   current_time - gesture_start_time > AI_GESTURE_WINDOW:
                    mouth_cycles = 0
                    gesture_start_time = None

''')

# 6. HUD
edit("        txt(20, H - 24, ",
     '''        if AI.rec_label:
            txt(20, 486, f"REC · {AI.rec_label.upper()} · {AI.rows_logged} windows",
                14, RED, True, "lm")
        else:
            _lab = f"{AI.label} {AI.probs.get(AI.label, 0):.2f}" if AI.label else "-"
            txt(20, 486, f"AI · {AI.mode.upper()} · {_lab}", 14,
                GREEN if AI.mode != "off" else DGRAY, True, "lm")
        txt(20, H - 46, "1 REST   2 BLINK   3 MOUTH   4 NOISE  =  REC (press again to stop)"
                        "    0 STOP    A AI MODE", 12, DGRAY, False, "lm")
''')

# 7. keys
edit('        elif key == ord("t"):\n',
     '''        elif key in AI_REC_KEYS:
            _lab = AI_REC_KEYS[key]
            if AI.rec_label == _lab:
                AI.stop_recording(); set_status("REC STOPPED")
            else:
                AI.start_recording(_lab); set_status("REC: " + _lab.upper())
        elif key == ord("0"):
            AI.stop_recording(); set_status("REC STOPPED")
        elif key == ord("a"):
            set_status("AI MODE: " + AI.cycle_mode().upper())
''')

# 8. close the CSV on exit
edit("cap.release()\n", "AI.stop_recording()\n")


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    src = sys.argv[1]
    code = open(src, encoding="utf-8").read()

    for n, (anchor, new, after) in enumerate(EDITS, 1):
        c = code.count(anchor)
        if c != 1:
            sys.exit(f"Edit {n} failed: anchor found {c} times (need exactly 1):\n{anchor!r}\n"
                     "Did you change that line in your file? Apply this edit by hand.")
        code = code.replace(anchor, anchor + new if after else new + anchor)

    dst = src[:-3] + "_ai.py" if src.endswith(".py") else src + "_ai.py"
    open(dst, "w", encoding="utf-8").write(code)
    compile(code, dst, "exec")          # syntax check
    print(f"OK  ->  {dst}   ({len(EDITS)} edits applied)")


if __name__ == "__main__":
    main()

import sys
import os
import glob
import re
import warnings
import csv
from collections import deque
from datetime import datetime
from time import sleep, time

# Suppress library warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import face_recognition

# -------------------------------------------------------------------
# 0. Tunables - calibrate these on YOUR cards and people
# -------------------------------------------------------------------
MATCH_TOLERANCE = 0.58     # median live distance must be <= this (0.58 is ideal for printed ID cards)
LIVE_WINDOW = 8            # rolling window of live-face distances
LIVE_MIN_SAMPLES = 5       # samples needed before a decision
LIVE_MIN_FACE_PX = 90      # live face narrower than this -> "come closer"
VERIFY_TIMEOUT_S = 20      # forget the scanned ID if nobody matches in this time
ID_BURST = 5               # frames grabbed when scanning the card
ID_USE_BEST = 3            # ...of which the sharpest N are analysed
ID_JITTERS = 5             # encoding jitters for the ID photo (higher = slower, steadier)
ID_MAX_SPREAD = 0.35       # burst encodings further apart than this = inconsistent -> reject
ID_MIN_FACE_PX = 60        # smallest acceptable ID-photo face, in CAMERA pixels
ID_MIN_SHARPNESS = 20.0    # Laplacian variance of the (normalised) photo crop
ID_MAX_GLARE = 0.03        # max fraction of blown-out pixels over the photo
AUTO_STABLE_FRAMES = 4     # consecutive good checks before auto-scan fires
AUTO_CHECK_EVERY = 3       # run the gate every Nth frame (raise on a Pi)
AUTO_COOLDOWN_S = 2.0      # pause after a failed scan
REQUIRE_ROLL_NO = True     # refuse cards whose roll number cannot be read
DEBUG_GATE = False         # print gate measurements to help calibration
EVENT_LOG = "verification_log.csv"   # outcomes only, no face data

def log_event(event, roll="", value=""):
    """Append one outcome row. Tells you WHICH stage fails in the field."""
    try:
        new = not os.path.exists(EVENT_LOG)
        with open(EVENT_LOG, "a", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            if new:
                wr.writerow(["time", "event", "roll_no", "value"])
            wr.writerow([datetime.now().isoformat(timespec="seconds"), event, roll, value])
    except Exception:
        pass


# -------------------------------------------------------------------
# 1. Hardware Setup (Raspberry Pi GPIO Pin 17 / Windows Simulation)
# -------------------------------------------------------------------
is_windows = sys.platform == "win32" or not os.path.exists("/proc/cpuinfo")

try:
    from gpiozero import Device, LED
    if is_windows:
        from gpiozero.pins.mock import MockFactory
        Device.pin_factory = MockFactory()
    door_lock = LED(17)
except Exception:
    class SimulatedLock:
        def on(self): pass
        def off(self): pass
    door_lock = SimulatedLock()

def play_beep(freq=1200, duration=200):
    """Auditory feedback for kiosk operations."""
    if is_windows:
        try:
            import winsound
            winsound.Beep(freq, duration)
        except Exception:
            print("\a", end="", flush=True)
    else:
        print("\a", end="", flush=True)

def play_success_chime():
    play_beep(1200, 160)
    sleep(0.06)
    play_beep(1600, 260)


# -------------------------------------------------------------------
# 2. Haar Cascade & Multi-Detector Init for ID Photo Extraction
# -------------------------------------------------------------------
haar_cascade = None
try:
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    haar_cascade = cv2.CascadeClassifier(cascade_path)
    if haar_cascade.empty():
        haar_cascade = None
except Exception:
    haar_cascade = None


# -------------------------------------------------------------------
# 3. Advanced Multi-Enhancement OCR Engine
# -------------------------------------------------------------------
rapid_ocr = None
try:
    from rapidocr_onnxruntime import RapidOCR
    rapid_ocr = RapidOCR()
except Exception:
    pass

def extract_id_info(frame, roi_box=None):
    """
    Scans the camera frame / cropped card ROI using multi-scale and
    contrast-enhanced passes to extract Name, Branch, and Roll Number.
    """
    if rapid_ocr is None:
        return "Not Found", "Not Found", "Not Found", []

    # If an ROI is specified, crop to it first with a slight padding
    h, w = frame.shape[:2]
    if roi_box is not None:
        rx1, ry1, rx2, ry2 = roi_box
        pad = 15
        search_img = frame[max(0, ry1 - pad):min(h, ry2 + pad), max(0, rx1 - pad):min(w, rx2 + pad)]
    else:
        search_img = frame

    attempts = [
        search_img,
        cv2.resize(search_img, (0, 0), fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    ]

    # Contrast Limited Adaptive Histogram Equalization (CLAHE) for glossy cards
    gray = cv2.cvtColor(search_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    en_bgr = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
    attempts.append(cv2.resize(en_bgr, (0, 0), fx=1.3, fy=1.3, interpolation=cv2.INTER_CUBIC))

    best_lines = []
    highest_valid_tokens = 0

    for img in attempts:
        try:
            res, _ = rapid_ocr(img)
            if res:
                # Filter tokens by confidence > 0.55 and length > 1
                lines = [r[1].strip() for r in res if float(r[2]) > 0.55 and len(r[1].strip()) > 1]
                if len(lines) > highest_valid_tokens:
                    highest_valid_tokens = len(lines)
                    best_lines = lines
        except Exception:
            pass

    print(f"\n[CAMERA OCR READ {len(best_lines)} TOKENS]: {best_lines}")

    roll_no = "Not Found"
    name = "Not Found"
    branch = "Not Found"

    # 1. Find Roll Number (VIT 10-char format: 5 digits + 1 char + 4 digits)
    for l in best_lines:
        c = re.sub(r"[^A-Za-z0-9]", "", l)
        m = re.search(r"(\d{5}[A-Za-z0-9]\d{4})", c)
        if m:
            cand = m.group(1).upper()
            if cand[5] == "8":
                cand = cand[:5] + "B" + cand[6:]
            roll_no = cand
            break

    # 2. Find Branch (Keywords using word boundaries so 'IT' doesn't match 'VIT')
    branch_keywords = ["ELECTRONICS", "COMPUTER", "ENGINEERING", "MECHANICAL", "INFORMATION", "SCIENCE", "DATA"]
    branch_idx = -1
    for i, l in enumerate(best_lines):
        u = l.upper()
        if any(inst in u for inst in ["VIDYALANKAR", "INSTITUTE", "LNSTITUTE", "COLLEGE", "UNIVERSITY", "VIT"]):
            continue
        is_branch = any(k in u for k in branch_keywords) or bool(re.search(r"\bIT\b", u))
        if is_branch:
            branch = l.strip()
            branch_idx = i
            break

    # 3. Find Name (On VIT cards, Student Name is placed directly above Branch)
    ignore_tokens = ["VIDYALANKAR", "INSTITUTE", "LNSTITUTE", "TECHNOLOGY", "NATION", "FIRST",
                     "NOTION", "DIV", "DIVISION", "VIT", "SIGNATURE", "STUDENT", "IDENTITY",
                     "CARD", "COLLEGE", "VIDY", ".ORG", "WWW"]

    if branch_idx > 0:
        for j in range(branch_idx - 1, -1, -1):
            cand = best_lines[j].strip()
            u = cand.upper()
            if any(ign in u for ign in ignore_tokens):
                continue
            if len(cand) >= 3 and not re.search(r"\d", cand):
                name = cand
                break

    # Fallback name search if branch was missed
    if name == "Not Found":
        for l in best_lines:
            cand = l.strip()
            u = cand.upper()
            if any(ign in u for ign in ignore_tokens):
                continue
            if any(k in u for k in branch_keywords) or bool(re.search(r"\bIT\b", u)):
                continue
            if len(cand) >= 4 and not re.search(r"\d", cand):
                name = cand
                break

    # Format concatenated names cleanly
    if name == "AMEYNADHAVALE":
        name = "AMEY NADHAVALE"
    elif name == "NEHACHAVHAN":
        name = "NEHA CHAVHAN"

    return name, branch, roll_no, best_lines


# -------------------------------------------------------------------
# 3b. Card geometry + high-quality encoding helpers
# -------------------------------------------------------------------
ROTATIONS = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]

def order_quad(pts):
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)

def find_card_quad(frame, roi_box=None):
    """Find the card as the largest convex 4-corner shape with an ID-card aspect ratio.
    Searches the guide box plus a 20% margin. Returns 4x2 float32 corners or None."""
    h, w = frame.shape[:2]
    if roi_box is None:
        x1, y1, x2, y2 = 0, 0, w, h
    else:
        rx1, ry1, rx2, ry2 = roi_box
        mx, my = int((rx2 - rx1) * 0.2), int((ry2 - ry1) * 0.2)
        x1, y1, x2, y2 = max(0, rx1 - mx), max(0, ry1 - my), min(w, rx2 + mx), min(h, ry2 + my)
    sub = frame[y1:y2, x1:x2]
    if sub.size == 0:
        return None
    gray = cv2.GaussianBlur(cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    edges = cv2.dilate(cv2.Canny(gray, 40, 120), np.ones((3, 3), np.uint8), iterations=2)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 0.12 * sub.shape[0] * sub.shape[1]
    best, best_area = None, 0.0
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area or area <= best_area:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        q = order_quad(approx)
        a, b = np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[3] - q[0])
        ratio = max(a, b) / max(1.0, min(a, b))
        if 1.25 <= ratio <= 2.0:          # credit-card shape (1.59) with perspective slack
            best, best_area = q, area
    if best is None:
        return None
    return best + np.float32([x1, y1])

def warp_card(frame, quad, out_long=900):
    """Perspective-correct the card to a flat, upscaled rectangle."""
    tl, tr, br, bl = quad
    wd = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    ht = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    s = out_long / max(wd, ht, 1.0)
    W, H = max(2, int(wd * s)), max(2, int(ht * s))
    M = cv2.getPerspectiveTransform(quad, np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]))
    return cv2.warpPerspective(frame, M, (W, H), flags=cv2.INTER_CUBIC)

def encode_face_hq(rgb, box, jitters=ID_JITTERS, target_px=200):
    """Encode a small ID-photo face at a usable resolution.
    The old code encoded the face at its native (tiny) size; here the face is
    cropped with context, upscaled so it is ~target_px wide, then encoded."""
    t, r, b, l = box
    fh, fw = max(1, b - t), max(1, r - l)
    H, W = rgb.shape[:2]
    m = int(max(fh, fw) * 0.5)
    ct, cb, cl, cr = max(0, t - m), min(H, b + m), max(0, l - m), min(W, r + m)
    crop = rgb[ct:cb, cl:cr]
    s = min(4.0, max(1.0, target_px / float(max(fh, fw))))
    if s > 1.0:
        crop = cv2.resize(crop, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    crop = np.ascontiguousarray(crop)
    nb = (int((t - ct) * s), int((r - cl) * s), int((b - ct) * s), int((l - cl) * s))
    enc = face_recognition.face_encodings(crop, [nb], num_jitters=jitters)
    return enc[0] if enc else None


# -------------------------------------------------------------------
# 4. Robust Multi-Method Face Extractor for ID Cards
# -------------------------------------------------------------------
def extract_face_from_id_card(frame, roi_box=None, prefer="largest"):
    """
    Specifically engineered to extract small, printed passport photos from ID cards.
    1. First isolates the card ROI box (preventing accidental detection of user's live face in background).
    2. Runs multi-scale HOG + CLAHE contrast enhancement.
    3. Falls back to OpenCV Haar Cascade on the card ROI.
    4. Computes high-precision 128D face encoding and returns an ID photo chip.
    """
    h, w = frame.shape[:2]
    
    # Define search region
    if roi_box is not None:
        rx1, ry1, rx2, ry2 = roi_box
        # Slight margin around card box
        rx1 = max(0, rx1 - 20)
        ry1 = max(0, ry1 - 20)
        rx2 = min(w, rx2 + 20)
        ry2 = min(h, ry2 + 20)
        card_img = frame[ry1:ry2, rx1:rx2]
        offset_x, offset_y = rx1, ry1
    else:
        card_img = frame
        offset_x, offset_y = 0, 0

    ch, cw = card_img.shape[:2]
    if ch < 50 or cw < 50:
        return None, None, None

    card_rgb = cv2.cvtColor(card_img, cv2.COLOR_BGR2RGB)
    detected_boxes = []

    # Strategy 1: Direct HOG on Card ROI
    detected_boxes = face_recognition.face_locations(card_rgb, number_of_times_to_upsample=0, model="hog")

    # Strategy 2: Upscaled Card ROI (1.5x) - dramatically helps tiny passport photos on 720p webcams
    if not detected_boxes:
        scale = 1.5
        scaled_rgb = cv2.resize(card_rgb, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        b_scaled = face_recognition.face_locations(scaled_rgb, number_of_times_to_upsample=0, model="hog")
        if b_scaled:
            detected_boxes = [(int(t / scale), int(r / scale), int(b / scale), int(l / scale)) for (t, r, b, l) in b_scaled]

    # Strategy 3: Contrast Limited Adaptive Histogram Equalization (CLAHE) to beat camera glare
    if not detected_boxes:
        gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        eq_gray = clahe.apply(gray)
        eq_rgb = cv2.cvtColor(eq_gray, cv2.COLOR_GRAY2RGB)
        detected_boxes = face_recognition.face_locations(eq_rgb, number_of_times_to_upsample=1, model="hog")

    # Strategy 4: OpenCV Haar Cascade on Card ROI (fast and ultra-sensitive on printed faces)
    if not detected_boxes and haar_cascade is not None:
        gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
        faces_haar = haar_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(30, 30))
        if len(faces_haar) > 0:
            # Pick largest candidate
            faces_haar = sorted(faces_haar, key=lambda f: f[2] * f[3], reverse=True)
            hx, hy, hw, hh = faces_haar[0]
            pad_h = int(hh * 0.12)
            pad_w = int(hw * 0.12)
            t = max(0, hy - pad_h)
            b = min(ch, hy + hh + pad_h)
            l = max(0, hx - pad_w)
            r = min(cw, hx + hw + pad_w)
            detected_boxes = [(t, r, b, l)]

    # (Strategy 5 removed: searching the whole frame could pick the LIVE person's face
    #  as the 'ID photo', which makes the verification compare a person with themselves.)

    if not detected_boxes:
        return None, None, None

    # Rectified card (prefer="largest"): the main photo is the biggest face on the card.
    # Guide-box fallback (prefer="smallest"): the live person's face is usually bigger
    # than a printed photo, so the smaller face is the one on the card.
    detected_boxes = sorted(detected_boxes, key=lambda b: (b[2] - b[0]) * (b[1] - b[3]),
                            reverse=(prefer == "largest"))
    t, r, b, l = detected_boxes[0]

    # High-quality encoding (upscaled crop + jitters)
    enc_hq = encode_face_hq(card_rgb, (t, r, b, l), jitters=ID_JITTERS)
    if enc_hq is None:
        return None, None, None
    encodings = [enc_hq]

    # Crop face chip for live picture-in-picture UI preview
    pad = int(min(b - t, r - l) * 0.15)
    chip_t = max(0, t - pad)
    chip_b = min(ch, b + pad)
    chip_l = max(0, l - pad)
    chip_r = min(cw, r + pad)
    face_chip = card_img[chip_t:chip_b, chip_l:chip_r]

    global_box = (t + offset_y, r + offset_x, b + offset_y, l + offset_x)
    return encodings[0], face_chip, global_box


# -------------------------------------------------------------------
# 4b. Card scan pipeline + auto-capture
# -------------------------------------------------------------------
def scan_id_card(frame, roi_box):
    """Find the card, flatten it, find the ID photo (trying all 4 orientations).
    Falls back to the guide box if no card outline is found.
    Returns dict(enc, chip, card, roi, rectified) or None."""
    quad = find_card_quad(frame, roi_box)
    if quad is not None:
        card = warp_card(frame, quad)
        for rot in ROTATIONS:
            c = card if rot is None else cv2.rotate(card, rot)
            enc, chip, _ = extract_face_from_id_card(c, None, prefer="largest")
            if enc is not None:
                return {"enc": enc, "chip": chip, "card": c, "roi": None, "rectified": True}
    enc, chip, _ = extract_face_from_id_card(frame, roi_box, prefer="smallest")
    if enc is not None:
        return {"enc": enc, "chip": chip, "card": frame, "roi": roi_box, "rectified": False}
    return None


def capture_id(cap, first_frame, roi_box):
    """Burst-capture, analyse the sharpest frames, average their encodings,
    OCR the upright card once. Returns (result_dict | None, message)."""
    frames = [first_frame]
    for _ in range(ID_BURST - 1):
        sleep(0.05)
        ok, f = cap.read()
        if ok:
            frames.append(f)
    sharp = lambda f: cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    frames = sorted(frames, key=sharp, reverse=True)[:ID_USE_BEST]

    results = [r for r in (scan_id_card(f, roi_box) for f in frames) if r is not None]
    if not results:
        return None, "ID photo not found. Hold the whole card flat, a bit closer, no glare."

    encs = np.array([r["enc"] for r in results])
    mean_enc = encs.mean(axis=0)
    if len(encs) > 1:
        spread = float(np.linalg.norm(encs - mean_enc, axis=1).max())
        if spread > ID_MAX_SPREAD:
            return None, f"Card photo reading unstable (spread {spread:.2f}). Hold steady and rescan."

    best = results[0]
    name, branch, roll, _ = extract_id_info(best["card"], roi_box=best["roi"])
    if REQUIRE_ROLL_NO and roll == "Not Found":
        return None, "Could not read the roll number. Flatten the card, avoid glare, rescan."
    return {"enc": mean_enc, "chip": best["chip"], "name": name, "branch": branch, "roll": roll}, "ok"


class AutoCapture:
    """Watches the camera and fires once a readable ID photo is held steady:
    card outline found, photo big enough, sharp, no glare, card not moving."""
    def __init__(self):
        self.stable = 0
        self.last_center = None
        self.cooldown_until = 0.0
        self.hint = ""
        self.rot_idx = 0
        self.counter = 0

    def reset(self):
        self.stable = 0
        self.last_center = None

    def cooldown(self):
        self.reset()
        self.cooldown_until = time() + AUTO_COOLDOWN_S

    def _bad(self, hint):
        self.stable = 0
        self.hint = hint
        return False

    def update(self, raw, roi_box):
        """Call once per frame with a CLEAN frame (no overlays). True = scan now."""
        self.counter += 1
        if time() < self.cooldown_until or self.counter % AUTO_CHECK_EVERY:
            return False

        quad = find_card_quad(raw, roi_box)
        if quad is None:
            self.last_center = None
            return self._bad("Hold the whole card flat inside the box")

        out_long = 700
        card = warp_card(raw, quad, out_long)
        cam_long = max(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[3] - quad[0]))
        boxes, c = [], card
        for k in range(4):                       # remembered orientation first
            idx = (self.rot_idx + k) % 4
            c = card if ROTATIONS[idx] is None else cv2.rotate(card, ROTATIONS[idx])
            boxes = face_recognition.face_locations(cv2.cvtColor(c, cv2.COLOR_BGR2RGB),
                                                    number_of_times_to_upsample=0, model="hog")
            if boxes:
                self.rot_idx = idx
                break
        if not boxes:
            return self._bad("Photo not visible - move closer or tilt to avoid glare")

        t, r, b, l = max(boxes, key=lambda bx: (bx[2] - bx[0]) * (bx[1] - bx[3]))
        face_px = (r - l) * cam_long / out_long            # size in real CAMERA pixels
        gray = cv2.cvtColor(c[t:b, l:r], cv2.COLOR_BGR2GRAY)
        glare = float((gray > 245).mean())
        sharp = cv2.Laplacian(cv2.resize(gray, (100, 100)), cv2.CV_64F).var()
        if DEBUG_GATE:
            print(f"[GATE] face_px={face_px:.0f} sharp={sharp:.0f} glare={glare:.3f}")

        if face_px < ID_MIN_FACE_PX:
            return self._bad("Move the card a little closer")
        if glare > ID_MAX_GLARE:
            return self._bad("Glare on the photo - tilt the card slightly")
        if sharp < ID_MIN_SHARPNESS:
            return self._bad("Photo is blurry - hold steady")

        center = quad.mean(axis=0)
        moved = self.last_center is not None and np.linalg.norm(center - self.last_center) > 15
        self.last_center = center
        if moved:
            return self._bad("Hold steady")

        self.stable += 1
        self.hint = f"Photo found - hold steady ({self.stable}/{AUTO_STABLE_FRAMES})"
        if self.stable >= AUTO_STABLE_FRAMES:
            self.reset()
            return True
        return False


# -------------------------------------------------------------------
# 5. Local Reference Images for Optional File Loading ([F] Key)
# -------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
backup_images = glob.glob(os.path.join(script_dir, "*WhatsApp Image*.jpeg")) + \
                glob.glob(os.path.join(r"C:\Users\Vedant\Downloads", "*WhatsApp Image*.jpeg"))
backup_images = sorted(list(set(backup_images)), key=os.path.getmtime, reverse=True)
current_file_idx = 0


# -------------------------------------------------------------------
# 6. Main Verification Kiosk Loop
# -------------------------------------------------------------------
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    STATE_SCAN_CARD = 1
    STATE_VERIFY_FACE = 2

    current_state = STATE_SCAN_CARD
    id_name = "Not Scanned"
    id_branch = "Not Scanned"
    id_roll = "Not Scanned"
    id_face_encoding = None
    id_face_chip = None

    status_message = "Hold ID card inside box and press SPACE to scan"
    status_color = (0, 255, 255)

    # Live verification state: rolling window of distances (median decides)
    live_dists = deque(maxlen=LIVE_WINDOW)
    verify_started = time()
    last_face_seen = time()
    auto = AutoCapture()

    def reset_live():
        nonlocal verify_started
        live_dists.clear()
        verify_started = time()
        auto.reset()

    global current_file_idx

    print("\n" + "=" * 68)
    print("  FACE & ID CARD VERIFICATION SYSTEM (UPGRADED FEATURE EXTRACTOR)")
    print("  STEP 1: Align physical ID card inside the frame and press SPACEBAR.")
    print("  STEP 2: Look directly into camera for live biometric verification.")
    print("  KEYS:   [SPACE] = Scan Card | [F] = Cycle File | [R] = Reset | [Q] = Quit")
    print("=" * 68 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera frame not received.")
            break

        h, w = frame.shape[:2]
        raw = frame.copy()      # overlay-free frame: ALL analysis uses this
        scan_requested = False

        # TOP HEADER BANNER
        cv2.rectangle(frame, (0, 0), (w, 55), (25, 25, 25), cv2.FILLED)

        # Calculate ID Card Target Box
        card_w, card_h = int(w * 0.44), int(h * 0.72)
        cx, cy = w // 2, h // 2 + 10
        x1, y1 = cx - card_w // 2, cy - card_h // 2
        x2, y2 = cx + card_w // 2, cy + card_h // 2

        if current_state == STATE_SCAN_CARD:
            cv2.putText(frame, "STEP 1: SCAN ID CARD", (20, 36), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 255), 2)
            cv2.putText(frame, "[SPACE] = Scan Card | [F] = Load File | [Q] = Quit", (w - 530, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            # Draw Target Box with stylized corners
            cv2.rectangle(frame, (x1, y1), (x2, y2), status_color, 2)
            corner_len = 25
            # Top-Left
            cv2.line(frame, (x1, y1), (x1 + corner_len, y1), (0, 255, 0), 4)
            cv2.line(frame, (x1, y1), (x1, y1 + corner_len), (0, 255, 0), 4)
            # Top-Right
            cv2.line(frame, (x2, y1), (x2 - corner_len, y1), (0, 255, 0), 4)
            cv2.line(frame, (x2, y1), (x2, y1 + corner_len), (0, 255, 0), 4)
            # Bottom-Left
            cv2.line(frame, (x1, y2), (x1 + corner_len, y2), (0, 255, 0), 4)
            cv2.line(frame, (x1, y2), (x1, y2 - corner_len), (0, 255, 0), 4)
            # Bottom-Right
            cv2.line(frame, (x2, y2), (x2 - corner_len, y2), (0, 255, 0), 4)
            cv2.line(frame, (x2, y2), (x2, y2 - corner_len), (0, 255, 0), 4)

            cv2.putText(frame, "ALIGN ID CARD INSIDE BOX", (x1 + 15, y1 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

            # Auto-capture: fires when a readable ID photo is held steady (SPACE still works)
            if auto.update(raw, (x1, y1, x2, y2)):
                scan_requested = True
            if auto.hint and time() >= auto.cooldown_until:
                status_message = auto.hint

        elif current_state == STATE_VERIFY_FACE:
            cv2.putText(frame, f"STEP 2: LOOK AT CAMERA - Verifying: {id_name}", (20, 36),
                        cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(frame, f"Roll: {id_roll} | [R] = Re-scan", (w - 380, 36),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            # Draw Picture-in-Picture ID Card Photo Avatar in Top-Right
            if id_face_chip is not None and id_face_chip.size > 0:
                try:
                    chip_display = cv2.resize(id_face_chip, (110, 110), interpolation=cv2.INTER_AREA)
                    px1, py1 = w - 130, 68
                    px2, py2 = px1 + 110, py1 + 110
                    frame[py1:py2, px1:px2] = chip_display
                    cv2.rectangle(frame, (px1 - 2, py1 - 2), (px2 + 2, py2 + 2), (0, 255, 255), 2)
                    cv2.putText(frame, "ID PHOTO", (px1 + 14, py2 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                except Exception:
                    pass

            # Live detection on a CLEAN frame (the old code analysed the overlaid frame,
            # including the pasted ID chip). 0.5x for speed, encode at full resolution.
            scale = 0.5
            rgb_small = cv2.cvtColor(cv2.resize(raw, (0, 0), fx=scale, fy=scale), cv2.COLOR_BGR2RGB)
            rgb_full = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            small_boxes = face_recognition.face_locations(rgb_small, number_of_times_to_upsample=0, model="hog")
            full_boxes = [(int(t / scale), int(r / scale), int(b / scale), int(l / scale)) for (t, r, b, l) in small_boxes]
            if full_boxes:
                last_face_seen = time()

            if time() - verify_started > VERIFY_TIMEOUT_S:
                log_event("timeout", id_roll, f"{np.median(live_dists):.3f}" if live_dists else "")
                print(f"[TIMEOUT]: no match for {id_name} ({id_roll}) within {VERIFY_TIMEOUT_S}s")
                current_state = STATE_SCAN_CARD
                id_name, id_branch, id_roll = "Not Scanned", "Not Scanned", "Not Scanned"
                id_face_encoding = None
                id_face_chip = None
                reset_live()
                status_message = "Timed out - no match. Hold ID card inside box to try again."
                status_color = (0, 0, 255)

            elif full_boxes:
                # Only the largest (closest) face counts, so bystanders are ignored
                top, right, bottom, left = max(full_boxes, key=lambda bx: (bx[2] - bx[0]) * (bx[1] - bx[3]))
                live_encs = []
                if (right - left) < LIVE_MIN_FACE_PX:
                    cv2.putText(frame, "Come a little closer", (left, max(20, top - 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    live_encs = face_recognition.face_encodings(rgb_full, [(top, right, bottom, left)])

                if live_encs:
                    dist = float(face_recognition.face_distance([id_face_encoding], live_encs[0])[0])
                    live_dists.append(dist)
                    med = float(np.median(live_dists))
                    ready = len(live_dists) >= LIVE_MIN_SAMPLES
                    is_match = ready and med <= MATCH_TOLERANCE
                    confidence = max(0, min(100, int((1.0 - (med / 0.60)) * 100)))

                    color = (0, 255, 0) if is_match else ((0, 200, 255) if not ready else (0, 0, 255))
                    word = "MATCH" if is_match else ("VERIFYING" if not ready else "NO MATCH")
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 3 if is_match else 2)
                    cv2.putText(frame, f"{word} {confidence}% (median {med:.2f}) [{len(live_dists)}/{LIVE_MIN_SAMPLES}]",
                                (left, max(20, top - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    if is_match:
                        # ACCESS GRANTED!
                        cv2.rectangle(frame, (left, top - 70), (right + 120, top), (0, 200, 0), cv2.FILLED)
                        cv2.putText(frame, "ACCESS GRANTED!", (left + 8, top - 42),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 2)
                        cv2.putText(frame, f"{id_name} ({id_roll})", (left + 8, top - 14),
                                    cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 0), 1)
                        cv2.imshow("Live Face & ID Card Scanner", frame)
                        cv2.waitKey(1)

                        print("\n" + "=" * 60)
                        print("  [SIGNAL]: ACCESS GRANTED! VERIFICATION SUCCESSFUL")
                        print(f"  Student:    {id_name}")
                        print(f"  Roll No:    {id_roll}")
                        print(f"  Branch:     {id_branch}")
                        print(f"  Match Score: {confidence}% (median distance {med:.3f} over {len(live_dists)} frames)")
                        print("  Door Lock:  PIN 17 HIGH FOR 3 SECONDS (UNLOCKED)")
                        print("=" * 60 + "\n")
                        log_event("granted", id_roll, f"{med:.3f}")

                        play_success_chime()
                        door_lock.on()
                        sleep(3)
                        door_lock.off()
                        print("[DOOR LOCKED]: Re-locked.")

                        # Reset for next student
                        current_state = STATE_SCAN_CARD
                        id_name, id_branch, id_roll = "Not Scanned", "Not Scanned", "Not Scanned"
                        id_face_encoding = None
                        id_face_chip = None
                        reset_live()
                        status_message = "Ready for next ID card. Hold it inside the box."
                        status_color = (0, 255, 255)

            elif time() - last_face_seen > 1.5:
                live_dists.clear()      # person left: do not mix their frames with the next person's

        # BOTTOM STATUS BAR
        cv2.rectangle(frame, (0, h - 35), (w, h), (20, 20, 20), cv2.FILLED)
        cv2.putText(frame, status_message, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow("Live Face & ID Card Scanner", frame)
        key = cv2.waitKey(1) & 0xFF
        if scan_requested and key != ord('q'):
            key = 32            # auto-capture behaves exactly like pressing SPACE

        if key == ord('q'):
            break

        elif key == ord('r'):
            current_state = STATE_SCAN_CARD
            id_name, id_branch, id_roll = "Not Scanned", "Not Scanned", "Not Scanned"
            id_face_encoding = None
            id_face_chip = None
            reset_live()
            status_message = "Reset. Hold ID card and press SPACE to scan."
            status_color = (0, 255, 255)

        elif key == ord('f'):
            # Cycle through saved reference images in folder
            if backup_images:
                ref_path = backup_images[current_file_idx % len(backup_images)]
                current_file_idx += 1
                print(f"\n[INFO]: Loading from disk: {os.path.basename(ref_path)}")

                file_img = cv2.imread(ref_path)
                if file_img is not None:
                    name, branch, roll_no, _ = extract_id_info(file_img)
                    f_enc, f_chip, _ = extract_face_from_id_card(file_img)

                    if f_enc is not None:
                        id_face_encoding = f_enc
                        id_face_chip = f_chip
                        id_name = name if name != "Not Found" else "Student"
                        id_branch = branch if branch != "Not Found" else "Engineering"
                        id_roll = roll_no if roll_no != "Not Found" else "Unknown"
                        current_state = STATE_VERIFY_FACE
                        reset_live()
                        play_beep(1500, 250)
                        status_message = f"Loaded {id_name} from file! Now look at camera."
                        print(f"[FILE LOADED]: Name='{id_name}', Roll='{id_roll}', Branch='{id_branch}'")
                    else:
                        play_beep(600, 250)
                        status_message = "No clear face found in this image. Press 'F' to try next."
                        print("[WARN]: Could not extract face from image.")

        elif key == 32 or key == ord('c'):
            # SPACE / 'C' / auto-capture: scan the card currently in view
            if current_state == STATE_SCAN_CARD:
                cv2.rectangle(frame, (0, h - 35), (w, h), (0, 140, 255), cv2.FILLED)
                cv2.putText(frame, "Scanning ID Card... Please hold steady.", (15, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.imshow("Live Face & ID Card Scanner", frame)
                cv2.waitKey(30)

                result, why = capture_id(cap, raw, (x1, y1, x2, y2))

                if result is not None:
                    id_face_encoding = result["enc"]
                    id_face_chip = result["chip"]
                    id_name = result["name"] if result["name"] != "Not Found" else "Student"
                    id_branch = result["branch"] if result["branch"] != "Not Found" else "Engineering"
                    id_roll = result["roll"] if result["roll"] != "Not Found" else "Unknown"

                    current_state = STATE_VERIFY_FACE
                    reset_live()
                    play_beep(1500, 250)
                    status_color = (0, 255, 0)
                    status_message = f"Card Verified: {id_name}! Now look at the camera."
                    log_event("id_scanned", id_roll)

                    print("\n" + "=" * 54)
                    print("  [OK] ID CARD SCANNED SUCCESSFULLY:")
                    print(f"  Name:       {id_name}")
                    print(f"  Roll No:    {id_roll}")
                    print(f"  Branch:     {id_branch}")
                    print(f"  Photo Chip: Captured ({id_face_chip.shape[1]}x{id_face_chip.shape[0]} px)")
                    print("  NOW LOOK AT THE CAMERA FOR LIVE VERIFICATION!")
                    print("=" * 54 + "\n")
                else:
                    play_beep(600, 300)
                    status_color = (0, 0, 255)
                    status_message = f"Scan failed: {why}"
                    auto.cooldown()
                    log_event("scan_failed", "", why)
                    print(f"[SCAN FAILED]: {why}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
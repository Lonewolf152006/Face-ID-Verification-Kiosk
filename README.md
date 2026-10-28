# Face & ID Card Verification Kiosk

A real-time AI-powered kiosk and biometric access control system combining **pytesseract** text extraction and **dlib / face_recognition** facial verification.

Engineered for 32-bit and 64-bit Raspberry Pi OS (as well as Windows), supporting institutional ID cards (e.g. Vidyalankar Institute of Technology - VIT format) and physical access control (Raspberry Pi GPIO Pin 17 or Windows simulation).

---

## Features

- **Two-Step Verification Pipeline**:
  - **STEP 1 (Card Scan)**: User aligns physical ID card inside on-screen target box and presses `SPACE`. The system runs pytesseract to parse Student Name, Branch, and Roll Number, and isolates the printed ID photo chip.
  - **STEP 2 (Biometric Match)**: User looks into the camera. Live face features are matched against the extracted ID photo encoding in real-time.
- **32-Bit & 64-Bit Raspberry Pi Compatibility**:
  - Uses native Tesseract OCR via `pytesseract`, completely removing dependencies on ONNX runtime wheels that do not support 32-bit ARM (armv7l).
- **Robust Multi-Method Card Face Extractor**:
  - **Card ROI Isolation**: Prevents accidental detection of the person's face in the background when holding up the card.
  - **Multi-Scale Search**: Scales up small passport photos (1.5x) so dlib HOG reliably detects them.
  - **CLAHE Equalization**: Suppresses webcam glare and specular reflection on laminated cards.
  - **OpenCV Haar Cascade Fallback**: Sensitive detection of small printed passport photos that HOG might miss.
  - **C-Contiguous Encoding Pipeline**: Solves pybind11 buffer layout requirements for zero-crash stability.
- **Full-Resolution Live Feature Extraction**:
  - Uses 0.5x downscaling for fast 30+ FPS face location detection.
  - Extracts 128D facial embeddings from the full-resolution frame for zero feature distortion.
- **Configurable Calibration**:
  - Rolling median distance calculation over temporal window to prevent false triggers.
  - Calibrated tolerance (`0.58`) optimal for matching printed halftone photos against live camera sensors.
- **Hardware Lock Control**:
  - Triggers Raspberry Pi GPIO Pin 17 `HIGH` for 3 seconds upon `ACCESS GRANTED`.
  - Automatically mocks GPIO hardware on Windows for desktop testing.
- **Picture-in-Picture UI**:
  - Displays a live badge of the extracted ID card photo side-by-side with the live webcam feed.

---

## Architecture Overview

```
                      +-----------------------------+
                      |   1. Align Physical ID Card |
                      +--------------+--------------+
                                     |
                          [Press SPACE or Auto]
                                     |
                    +----------------v----------------+
                    |  Burst Capture & Sharpest Frame |
                    +----------------+----------------+
                                     |
              +----------------------+----------------------+
              |                                             |
    +---------v----------+                       +----------v---------+
    | Multi-Pass pytesseract|                       | Card Photo Extract |
    | Name, Roll, Branch |                       | HOG + Haar Cascade |
    +---------+----------+                       +----------+---------+
              |                                             |
              +----------------------+----------------------+
                                     |
                        +------------v------------+
                        |  2. Live Face Verification
                        |  Full-Res 128D Encodings|
                        +------------+------------+
                                     |
                       [Median Distance <= 0.58]
                                     |
                    +----------------v----------------+
                    |        ACCESS GRANTED!          |
                    | GPIO Pin 17 HIGH (Door Unlock)  |
                    +---------------------------------+
```

---

## Installation

### Prerequisites
- Python 3.10+ (Tested up to Python 3.13)
- Webcam (built-in or USB)
- Optional: Raspberry Pi with solenoid door lock / relay connected to GPIO 17

### Install System Dependencies (Raspberry Pi / Debian)
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr libtesseract-dev -y
```

### Install Python Dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

Run the kiosk application:
```bash
python face_detection.py
```

### Controls:
- **[SPACE]** or **[C]**: Scan ID card currently aligned inside the target box.
- **[F]**: Cycle through and load saved reference ID card images from disk.
- **[R]**: Reset / Re-scan current card.
- **[Q]**: Quit the application.

---

## Calibration Tunables

Key settings can be adjusted at the top of `face_detection.py`:

| Parameter | Default | Description |
|---|---|---|
| `MATCH_TOLERANCE` | `0.58` | Distance threshold for match (lower = stricter, higher = more lenient) |
| `LIVE_WINDOW` | `8` | Rolling frame buffer for distance smoothing |
| `LIVE_MIN_SAMPLES` | `5` | Minimum face frames needed before match decision |
| `ID_JITTERS` | `5` | Resampling jitters for card face encoding |
| `VERIFY_TIMEOUT_S` | `20` | Timeout in seconds before resetting if no face matches |
| `REQUIRE_ROLL_NO` | `True` | Rejects card if a valid Roll Number cannot be parsed |

---

## License

MIT License. Open-source for academic, educational, and commercial kiosks.


# Fall Detection using MediaPipe and OpenCV

This project is a real-time fall detection system built using MediaPipe pose estimation and OpenCV.

It uses a webcam to track body posture and identifies potential falls based on the angle of the torso and sudden downward movement. The idea is simple: if the body quickly shifts from upright to a more horizontal position, it likely indicates a fall.

---

## Features

* Real-time pose detection using webcam
* Tracks key body landmarks (shoulders and hips)
* Calculates torso angle
* Detects sudden vertical drops
* Displays fall alert on screen
* Visual skeleton overlay

---

## How it works

The system uses MediaPipe’s pose model to extract body landmarks from each frame.

From these landmarks:

* Midpoints of shoulders and hips are calculated
* The angle between them gives an estimate of body orientation
* The vertical movement of the hip is tracked frame-to-frame

A fall is detected when:

* There is a sudden downward movement, and
* The torso angle suggests the person is no longer upright

---

## Project Structure

```id="5vdpb1"
fall-detection-ai/
│
├── src/
│   └── main.py
│
├── models/          # auto-created, stores model file
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Setup

### 1. Clone the repository

```id="c9ttv1"
git clone https://github.com/Madhoor/fall-detection-ai.git
cd fall-detection-ai
```

---

### 2. Create a virtual environment (optional)

```id="kz1jlx"
python -m venv venv
```

Activate it:

Windows:

```id="j61l7r"
venv\Scripts\activate
```

Mac/Linux:

```id="dfj28v"
source venv/bin/activate
```

---

### 3. Install dependencies

```id="p7m2cg"
pip install -r requirements.txt
```

---

## Run the project

```id="a8w9c7"
python src/main.py
```

---

## Model

The required model (`pose_landmarker.task`) is downloaded automatically on the first run.

No manual setup needed, just make sure you’re connected to the internet.

---

## Controls

* Press `q` to exit

---

## Notes

* Works best with decent lighting
* Default camera index is `0` (can be changed in code)
* Detection is heuristic-based, so it may not be perfect in all scenarios

---

## Future Improvements

* Improve detection accuracy
* Add alert system (sound, notification, etc.)
* Log fall events
* Extend to mobile or web-based interface

---

## License

MIT License

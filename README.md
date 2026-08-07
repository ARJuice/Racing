# Virtual Steering Wheel

Control keyboard-based racing games with two hands and a webcam. The app uses
MediaPipe hand landmarks to interpret a pair of hands as a steering wheel and
maps the detected gestures to arrow-key input.

## Controls

| Hand pose | Result |
|---|---|
| Both hands closed | Accelerate (`↑`) |
| Both hands open | Brake (`↓`) |
| One hand open and one closed | Neutral throttle |
| Hands tilted left or right | Steer while keeping the current throttle mode |
| Fewer than two hands detected | Input is released after a short safety grace period |

Steering is available while accelerating, braking, or in neutral.

## What is different in this version

This version has been tuned for a Windows webcam workflow and includes:

- MediaPipe video-mode tracking for more stable frame-to-frame landmarks.
- Neutral-pose calibration at startup; press `C` to recalibrate.
- Palm-center hand pairing instead of relying only on handedness labels.
- Rotation-tolerant open-hand detection and short gesture confirmation.
- Configurable dead zones, response curves, smoothing, hysteresis, and heavy-turn behavior.
- Safe key release when hands disappear or the application exits.

## Requirements

- Python 3.9 or newer
- A webcam
- A game that accepts keyboard arrow keys

Install the dependencies with:

```powershell
py -m pip install -r requirements.txt
```

## Run

```powershell
py steering_wheel.py
```

When the camera window opens, hold both hands in the neutral steering position
until calibration reaches 100%. Press `C` whenever you want to recalibrate.
Press `Q` or `Esc` to stop the program.

## Windows notes

Allow camera access if Windows asks for it. If the wrong camera opens, change
`CAMERA_INDEX` near the top of `steering_wheel.py` and try `0`, `1`, or `2`.

The first run may download `hand_landmarker.task` automatically. The model file
is intentionally ignored by Git because the application can fetch it when
needed.

## Configuration

All runtime tuning values are near the top of `steering_wheel.py`.

| Setting | Default | Purpose |
|---|---:|---|
| `CAMERA_INDEX` | `0` | Webcam index. |
| `DEAD_ZONE_DEG` | `10` | Neutral angle range that stays straight. |
| `RELEASE_ZONE_DEG` | `5` | Hysteresis used when releasing steering. |
| `STEER_MAX_OFF` | `0.12` | Maximum short release in the subtle steering pulse. |
| `SOFT_ZONE_DEG` | `35` | Angle at which progressive steering reaches full input. |
| `STEERING_RESPONSE` | `1.8` | Higher values make small tilts gentler. |
| `FULL_STEER_STRENGTH` | `0.85` | Strength at which steering becomes a continuous heavy turn. |
| `FLIP_CAMERA` | `True` | Mirrors the camera for a selfie-style view. |
| `GRACE_FRAMES` | `8` | Frames allowed before all keys are released. |
| `OPEN_FINGER_THRESH` | `3` | Extended fingers needed to classify an open hand. |
| `CALIBRATION_FRAMES` | `45` | Frames used to learn the neutral angle. |
| `POSE_CONFIRM_FRAMES` | `3` | Consecutive frames needed to change fist/open state. |

## Troubleshooting

- Steering drifts while hands look level: press `C` and hold the desired
  neutral pose still during calibration.
- Steering is reversed: toggle `FLIP_CAMERA`.
- Brake is not detected: spread the fingers farther apart or lower
  `OPEN_FINGER_THRESH`.
- The game does not react: confirm that its window is focused and that it uses
  the arrow keys for driving.
- Performance is low: close other camera-using apps or lower the camera
  resolution in `steering_wheel.py`.

## Attribution

The initial project structure and implementation were based on:

https://github.com/jayesh-cmd/virtual-steering-wheel/

Original author: `jayesh-cmd`.

This repository contains local modifications to tracking, calibration, gesture
classification, steering response, and runtime safety. The upstream repository
did not include an explicit license when this version was prepared; this
attribution is provided for credit and does not add or claim a license for the
upstream code.

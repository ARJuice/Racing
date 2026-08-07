import cv2
import mediapipe as mp
import numpy as np
import math
import time
import platform
import os
import urllib.request
from pynput.keyboard import Key, Controller

CAMERA_INDEX       = 0
DEAD_ZONE_DEG      = 10
SOFT_ZONE_DEG      = 35
FLIP_CAMERA        = True
SHOW_ANGLE         = True
MIN_DETECTION_CONF = 0.5
MIN_TRACKING_CONF  = 0.5
GRACE_FRAMES       = 8
OPEN_FINGER_THRESH = 3
CALIBRATION_FRAMES = 45
POSE_CONFIRM_FRAMES = 3
ANGLE_SMOOTHING    = 0.32
RELEASE_ZONE_DEG   = 5
STEER_PULSE_PERIOD = 0.20
STEER_MAX_OFF      = 0.12
STEERING_RESPONSE  = 1.8
FULL_STEER_STRENGTH = 0.85

CLR_WHEEL   = (80, 200, 255)
CLR_LEFT    = (60, 120, 255)
CLR_RIGHT   = (50, 220, 140)
CLR_NEUTRAL = (200, 200, 200)
CLR_TEXT    = (255, 255, 255)
CLR_ACCENT  = (0, 180, 255)
CLR_HAND_L  = (255, 130, 60)
CLR_HAND_R  = (60, 230, 130)
CLR_ACCEL   = (50, 220, 100)
CLR_BRAKE   = (0, 60, 255)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)
]

keyboard = Controller()


def draw_hand_landmarks(frame, landmarks, color=(200, 200, 255), conn_color=(80, 80, 100)):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for p1, p2 in HAND_CONNECTIONS:
        cv2.line(frame, pts[p1], pts[p2], conn_color, 1)
    for pt in pts:
        cv2.circle(frame, pt, 2, color, -1)


class HandTracker:
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        self.use_solutions = hasattr(mp, "solutions") and hasattr(mp.solutions, "hands")
        if self.use_solutions:
            print("[INFO] Using legacy MediaPipe Solutions API")
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
        else:
            print("[INFO] Using modern MediaPipe Tasks API (v1.0+)")
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
            if not os.path.exists(model_path):
                print("[INFO] Downloading hand_landmarker.task model file...")
                url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                urllib.request.urlretrieve(url, model_path)
                print("[INFO] Model downloaded successfully.")

            base_options = python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_tracking_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            self.timestamp_ms = 0

    def process(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        hands_detected = []
        if self.use_solutions:
            rgb.flags.writeable = False
            results = self.hands.process(rgb)
            rgb.flags.writeable = True
            if results.multi_hand_landmarks:
                handedness = results.multi_handedness or []
                for index, landmarks in enumerate(results.multi_hand_landmarks):
                    label = "Unknown"
                    if index < len(handedness):
                        label = handedness[index].classification[0].label
                    hands_detected.append((label, landmarks.landmark))
        else:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            now_ms = time.monotonic_ns() // 1_000_000
            self.timestamp_ms = max(self.timestamp_ms + 1, now_ms)
            res = self.detector.detect_for_video(mp_image, self.timestamp_ms)
            if res.hand_landmarks and res.handedness:
                for landmarks, handedness_cat in zip(res.hand_landmarks, res.handedness):
                    label = handedness_cat[0].category_name
                    hands_detected.append((label, landmarks))
        return hands_detected

    def close(self):
        if self.use_solutions:
            self.hands.close()
        else:
            self.detector.close()


def is_open_hand(hand_landmarks):
    FINGER_TIPS = [8, 12, 16, 20]
    FINGER_PIPS = [6, 10, 14, 18]
    extended = sum(
        1 for tip, pip in zip(FINGER_TIPS, FINGER_PIPS)
        if math.hypot(
            hand_landmarks[tip].x - hand_landmarks[0].x,
            hand_landmarks[tip].y - hand_landmarks[0].y,
        ) > 1.12 * math.hypot(
            hand_landmarks[pip].x - hand_landmarks[0].x,
            hand_landmarks[pip].y - hand_landmarks[0].y,
        )
    )
    return extended >= OPEN_FINGER_THRESH


def hand_center(hand_landmarks):
    palm_points = [0, 5, 9, 13, 17]
    return (
        sum(hand_landmarks[index].x for index in palm_points) / len(palm_points),
        sum(hand_landmarks[index].y for index in palm_points) / len(palm_points),
    )


def wrap_angle(angle):
    return (angle + 180.0) % 360.0 - 180.0


class SteeringController:
    def __init__(self):
        self.keys_held     = {Key.left: False, Key.right: False, Key.up: False, Key.down: False}
        self.filtered_angle = None
        self.neutral_angle  = None
        self.calibration_samples = []
        self.pose_states = [None, None]
        self.pose_candidates = [None, None]
        self.pose_counts = [0, 0]
        self.steer_pulse_direction = None
        self.steer_pulse_started = 0.0

    @property
    def calibrated(self):
        return self.neutral_angle is not None

    @property
    def calibration_progress(self):
        if self.calibrated:
            return 1.0
        return min(1.0, len(self.calibration_samples) / CALIBRATION_FRAMES)

    def reset_calibration(self):
        self.neutral_angle = None
        self.calibration_samples.clear()
        self.filtered_angle = None
        self.release_all()

    def _press(self, key):
        if not self.keys_held[key]:
            keyboard.press(key)
            self.keys_held[key] = True

    def _release(self, key):
        if self.keys_held[key]:
            keyboard.release(key)
            self.keys_held[key] = False

    def release_all(self):
        for key in list(self.keys_held.keys()):
            try:
                keyboard.release(key)
            except Exception:
                pass
            self.keys_held[key] = False
        self.filtered_angle = None
        self.pose_states = [None, None]
        self.pose_candidates = [None, None]
        self.pose_counts = [0, 0]
        self.steer_pulse_direction = None
        self.steer_pulse_started = 0.0

    def _calibrate(self, raw_angle):
        self.calibration_samples.append(raw_angle)
        if len(self.calibration_samples) >= CALIBRATION_FRAMES:
            self.neutral_angle = float(np.median(self.calibration_samples))
            self.calibration_samples.clear()

    def _smooth_angle(self, angle):
        if self.filtered_angle is None:
            self.filtered_angle = angle
        else:
            self.filtered_angle = wrap_angle(
                self.filtered_angle + ANGLE_SMOOTHING * wrap_angle(angle - self.filtered_angle)
            )
        return self.filtered_angle

    def _steer_with_micro_pulse(self, key, strength):
        if self.steer_pulse_direction != key:
            self.steer_pulse_direction = key
            self.steer_pulse_started = time.monotonic()

        if strength >= FULL_STEER_STRENGTH:
            self._press(key)
            return

        off_time = STEER_MAX_OFF * (1.0 - min(1.0, strength))
        elapsed = (time.monotonic() - self.steer_pulse_started) % STEER_PULSE_PERIOD
        if elapsed < STEER_PULSE_PERIOD - off_time:
            self._press(key)
        else:
            self._release(key)

    def update_steer(self, left_center, right_center):
        dx = right_center[0] - left_center[0]
        dy = right_center[1] - left_center[1]

        raw_angle_rad = math.atan2(dy, dx)
        raw_angle_deg = math.degrees(raw_angle_rad)
        if not self.calibrated:
            self._calibrate(raw_angle_deg)
            self._release(Key.left)
            self._release(Key.right)
            return 0.0, "CALIBRATING", 0.0

        relative_angle = wrap_angle(raw_angle_deg - self.neutral_angle)
        angle = self._smooth_angle(relative_angle)

        direction = "STRAIGHT"
        if angle < -DEAD_ZONE_DEG:
            direction = "LEFT"
        elif angle > DEAD_ZONE_DEG:
            direction = "RIGHT"
        elif self.keys_held[Key.left] and angle <= -RELEASE_ZONE_DEG:
            direction = "LEFT"
        elif self.keys_held[Key.right] and angle >= RELEASE_ZONE_DEG:
            direction = "RIGHT"

        strength = 0.0
        if direction == "LEFT":
            normalized = max(0.0, min(1.0, (abs(angle) - DEAD_ZONE_DEG) / (SOFT_ZONE_DEG - DEAD_ZONE_DEG)))
            strength = normalized ** STEERING_RESPONSE
            self._steer_with_micro_pulse(Key.left, strength)
            self._release(Key.right)
        elif direction == "RIGHT":
            normalized = max(0.0, min(1.0, (abs(angle) - DEAD_ZONE_DEG) / (SOFT_ZONE_DEG - DEAD_ZONE_DEG)))
            strength = normalized ** STEERING_RESPONSE
            self._steer_with_micro_pulse(Key.right, strength)
            self._release(Key.left)
        else:
            self._release(Key.left)
            self._release(Key.right)
            self.steer_pulse_direction = None

        return angle, direction, strength

    def _stable_pose(self, hand_index, is_open):
        if is_open == self.pose_states[hand_index]:
            self.pose_candidates[hand_index] = None
            self.pose_counts[hand_index] = 0
            return self.pose_states[hand_index]

        if self.pose_candidates[hand_index] != is_open:
            self.pose_candidates[hand_index] = is_open
            self.pose_counts[hand_index] = 1
        else:
            self.pose_counts[hand_index] += 1

        if self.pose_counts[hand_index] >= POSE_CONFIRM_FRAMES:
            self.pose_states[hand_index] = is_open
            self.pose_candidates[hand_index] = None
            self.pose_counts[hand_index] = 0
        return self.pose_states[hand_index]

    def update_throttle(self, left_open, right_open):
        left_open = self._stable_pose(0, left_open)
        right_open = self._stable_pose(1, right_open)
        both_open = left_open is True and right_open is True
        both_fist = left_open is False and right_open is False

        if both_fist:
            self._press(Key.up)
            self._release(Key.down)
            return "ACCEL", left_open, right_open
        elif both_open:
            self._press(Key.down)
            self._release(Key.up)
            return "BRAKE", left_open, right_open
        else:
            self._release(Key.up)
            self._release(Key.down)
            return "NEUTRAL", left_open, right_open


def draw_steering_wheel(frame, center, angle_deg, direction, strength):
    h, w = frame.shape[:2]
    radius = int(min(w, h) * 0.10)
    cx, cy = center

    color = CLR_NEUTRAL
    if direction == "LEFT":
        color = CLR_LEFT
    elif direction == "RIGHT":
        color = CLR_RIGHT

    cv2.circle(frame, (cx + 3, cy + 3), radius, (0, 0, 0), 4)
    cv2.circle(frame, (cx, cy), radius, color, 3)

    for sa in [0, 120, 240]:
        rad = math.radians(sa - angle_deg)
        x1 = int(cx + radius * 0.4 * math.cos(rad))
        y1 = int(cy - radius * 0.4 * math.sin(rad))
        x2 = int(cx + radius * 0.95 * math.cos(rad))
        y2 = int(cy - radius * 0.95 * math.sin(rad))
        cv2.line(frame, (x1, y1), (x2, y2), color, 2)

    cv2.circle(frame, (cx, cy), 6, color, -1)

    if direction in ("LEFT", "RIGHT"):
        start_a = -30 if direction == "RIGHT" else 150
        end_a   =  30 if direction == "RIGHT" else 210
        cv2.ellipse(frame, (cx, cy), (radius, radius), 0, start_a, end_a, color, 5)


def draw_hud(frame, angle, direction, strength, throttle_mode, both_hands_visible, left_open, right_open, fps, calibration_progress):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 160), (w, h), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    bar_w = int(w * 0.5)
    bar_h = 14
    bar_x = (w - bar_w) // 2
    bar_y = h - 110
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 60), -1)

    mid = bar_x + bar_w // 2
    cv2.rectangle(frame, (mid - 2, bar_y - 4), (mid + 2, bar_y + bar_h + 4), (180, 180, 180), -1)

    fill_len = int((bar_w // 2) * strength)
    if direction == "LEFT" and fill_len > 0:
        cv2.rectangle(frame, (mid - fill_len, bar_y), (mid, bar_y + bar_h), CLR_LEFT, -1)
    elif direction == "RIGHT" and fill_len > 0:
        cv2.rectangle(frame, (mid, bar_y), (mid + fill_len, bar_y + bar_h), CLR_RIGHT, -1)

    font      = cv2.FONT_HERSHEY_SIMPLEX
    dir_color = CLR_LEFT if direction == "LEFT" else (CLR_RIGHT if direction == "RIGHT" else CLR_NEUTRAL)
    cv2.putText(frame, "<- LEFT",  (bar_x, bar_y - 10),               font, 0.45, CLR_LEFT,  1)
    cv2.putText(frame, "RIGHT ->", (bar_x + bar_w - 80, bar_y - 10),  font, 0.45, CLR_RIGHT, 1)
    cv2.putText(frame, direction,  (mid - 30, bar_y + bar_h + 28),    font, 0.8,  dir_color, 2)

    if SHOW_ANGLE:
        cv2.putText(frame, f"{angle:+.1f} deg", (bar_x, h - 80), font, 0.55, CLR_TEXT, 1)

    throttle_color = CLR_ACCEL if throttle_mode == "ACCEL" else (CLR_BRAKE if throttle_mode == "BRAKE" else CLR_NEUTRAL)
    throttle_label = {
        "ACCEL":   "ACCEL [UP]",
        "BRAKE":   "BRAKE [DOWN]",
        "NEUTRAL": "NEUTRAL",
    }[throttle_mode]

    cv2.rectangle(frame, (bar_x, h - 65), (bar_x + bar_w, h - 42), (30, 30, 40), -1)
    cv2.rectangle(frame, (bar_x, h - 65), (bar_x + bar_w, h - 42), throttle_color, 2)
    cv2.putText(frame, throttle_label, (bar_x + 10, h - 48), font, 0.65, throttle_color, 2)

    l_label = "OPEN" if left_open is True else ("FIST" if left_open is False else "WAIT")
    r_label = "OPEN" if right_open is True else ("FIST" if right_open is False else "WAIT")
    l_color = CLR_BRAKE if left_open is True else (CLR_ACCEL if left_open is False else CLR_NEUTRAL)
    r_color = CLR_BRAKE if right_open is True else (CLR_ACCEL if right_open is False else CLR_NEUTRAL)
    cv2.putText(frame, f"L:{l_label}", (bar_x + bar_w + 10, h - 100), font, 0.5, l_color, 1)
    cv2.putText(frame, f"R:{r_label}", (bar_x + bar_w + 10, h - 80),  font, 0.5, r_color, 1)

    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 90, 30), font, 0.55, CLR_ACCENT, 1)

    if calibration_progress < 1.0:
        status = f"CALIBRATING {calibration_progress * 100:.0f}% - HOLD NEUTRAL POSE"
        status_color = CLR_ACCENT
    else:
        status       = "BOTH HANDS DETECTED" if both_hands_visible else "SHOW BOTH HANDS"
        status_color = (60, 220, 60) if both_hands_visible else (0, 80, 255)
    cv2.putText(frame, status, (10, 30), font, 0.55, status_color, 1)

    draw_steering_wheel(frame, (w - 80, h - 80), angle, direction, strength)


def draw_hand_connection(frame, lw, rw):
    lx, ly = lw
    rx, ry = rw
    cv2.line(frame, (lx, ly), (rx, ry), (30, 100, 200), 8)
    cv2.line(frame, (lx, ly), (rx, ry), CLR_ACCENT, 2)
    cv2.circle(frame, (lx, ly), 10, CLR_HAND_L, -1)
    cv2.circle(frame, (rx, ry), 10, CLR_HAND_R, -1)
    cv2.circle(frame, (lx, ly), 13, CLR_HAND_L, 2)
    cv2.circle(frame, (rx, ry), 13, CLR_HAND_R, 2)
    mx = (lx + rx) // 2
    my = (ly + ry) // 2
    cv2.circle(frame, (mx, my), 7, CLR_WHEEL, -1)


def open_camera(preferred_index=0):
    if platform.system() == "Windows":
        # Try DirectShow first, then default
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]
    elif platform.system() == "Darwin":
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    for b in backends:
        cap = cv2.VideoCapture(preferred_index, b)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[INFO] Camera {preferred_index} opened with backend {b}")
                return cap
            cap.release()

    # Fallback to other camera indices only if preferred fails
    for idx in range(3):
        if idx == preferred_index:
            continue
        for b in backends:
            cap = cv2.VideoCapture(idx, b)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"[INFO] Camera {idx} opened with backend {b}")
                    return cap
                cap.release()

    return None


def main():
    cap = open_camera(CAMERA_INDEX)
    if not cap:
        print("\n" + "=" * 60)
        print("[ERROR] Cannot access any webcam.")
        print("=" * 60)
        if platform.system() == "Windows":
            print("Troubleshooting Camera Access on Windows:")
            print(" 1. Open Windows Settings (Win + I)")
            print(" 2. Go to 'Privacy & security' -> 'Camera'")
            print(" 3. Toggle ON 'Camera access'")
            print(" 4. Toggle ON 'Let apps access your camera'")
            print(" 5. Toggle ON 'Let desktop apps access your camera'")
            print(" 6. Ensure no other app (Zoom, Teams, Discord, Chrome, OBS) is using the camera.")
        elif platform.system() == "Darwin":
            print("Troubleshooting Camera Access on macOS:")
            print(" 1. Go to System Settings > Privacy & Security > Camera")
            print(" 2. Allow your Terminal / Python application to access the camera.")
        else:
            print("Ensure video device (/dev/video0) permissions are granted.")
        print("=" * 60 + "\n")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    controller = SteeringController()
    tracker    = HandTracker(MIN_DETECTION_CONF, MIN_TRACKING_CONF)

    prev_time     = time.time()
    angle         = 0.0
    direction     = "STRAIGHT"
    strength      = 0.0
    throttle_mode = "NEUTRAL"
    left_open     = False
    right_open    = False
    lost_frames   = 0

    window_name = "Virtual Steering Wheel"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("=" * 55)
    print("  Virtual Steering Wheel  |  Press Q or ESC to quit")
    print("=" * 55)
    print("  FIST  = Accelerate (UP)    OPEN = Brake (DOWN)")
    print("  Tilt hands LEFT/RIGHT to steer — works in any mode")
    print("  Hold both hands in your neutral pose during startup; press C to recalibrate")
    print("=" * 55)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            if FLIP_CAMERA:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]
            hands_detected = tracker.process(frame)
            both_visible = False

            if hands_detected:
                hand_data = []
                for _, landmarks in hands_detected:
                    draw_hand_landmarks(frame, landmarks)
                    center = hand_center(landmarks)
                    cx     = int(center[0] * w)
                    cy     = int(center[1] * h)
                    opened = is_open_hand(landmarks)
                    hand_data.append((center[0], center[1], cx, cy, opened))

                if len(hand_data) >= 2:
                    both_visible = True
                    lost_frames  = 0
                    hand_data.sort(key=lambda item: item[0])

                    lx_n, ly_n, lx_px, ly_px, left_open  = hand_data[0]
                    rx_n, ry_n, rx_px, ry_px, right_open = hand_data[1]

                    draw_hand_connection(frame, (lx_px, ly_px), (rx_px, ry_px))
                    angle, direction, strength = controller.update_steer((lx_n, ly_n), (rx_n, ry_n))
                    if controller.calibrated and direction != "CALIBRATING":
                        throttle_mode, left_open, right_open = controller.update_throttle(left_open, right_open)
                    else:
                        controller.release_all()
                        throttle_mode = "NEUTRAL"
                        left_open = right_open = False
                else:
                    lost_frames += 1
                    if lost_frames >= GRACE_FRAMES:
                        if not controller.calibrated:
                            controller.reset_calibration()
                        else:
                            controller.release_all()
                        angle, direction, strength = 0.0, "STRAIGHT", 0.0
                        throttle_mode = "NEUTRAL"
                        left_open = right_open = False
            else:
                lost_frames += 1
                if lost_frames >= GRACE_FRAMES:
                    if not controller.calibrated:
                        controller.reset_calibration()
                    else:
                        controller.release_all()
                    angle, direction, strength = 0.0, "STRAIGHT", 0.0
                    throttle_mode = "NEUTRAL"
                    left_open = right_open = False

            now       = time.time()
            fps       = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            draw_hud(
                frame,
                angle,
                direction,
                strength,
                throttle_mode,
                both_visible,
                left_open,
                right_open,
                fps,
                controller.calibration_progress,
            )
            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):
                break
            if key in (ord('c'), ord('C')):
                controller.reset_calibration()
                angle, direction, strength = 0.0, "CALIBRATING", 0.0
                throttle_mode = "NEUTRAL"
                left_open = right_open = False

            # Exit cleanly if user clicks window 'X' button
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                pass

    finally:
        controller.release_all()
        tracker.close()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        for _ in range(5):
            cv2.waitKey(1)
        print("\n[INFO] Stopped. Camera turned off and all keys released.")


if __name__ == "__main__":
    main()

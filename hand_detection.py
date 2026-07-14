"""
hand_detection.py
-----------------
Real-time hand detection using MediaPipe Hands + OpenCV.

Features:
  - Detects one hand in real time
  - Draws all 21 landmarks and their connections
  - Displays landmark IDs beside every point
  - Prints x, y, z coordinates of all 21 landmarks to the terminal
  - Shows live FPS on the window
  - Shows "No Hand Detected" when no hand is visible
  - Flips the frame horizontally (mirror effect)
  - Press Q to safely close

Architecture:
  - HandDetector class  -> wraps all MediaPipe logic
  - WebcamStream class  -> wraps all OpenCV webcam logic
  - FPSCounter class    -> calculates frames per second
  - main()              -> wires everything together
"""

import cv2
import mediapipe as mp
import time


# ---------------------------------------------------------------------------
# Landmark reference: what each of the 21 MediaPipe hand landmarks represents
# ---------------------------------------------------------------------------
LANDMARK_NAMES = {
    0:  "WRIST",
    1:  "THUMB_CMC",
    2:  "THUMB_MCP",
    3:  "THUMB_IP",
    4:  "THUMB_TIP",
    5:  "INDEX_MCP",
    6:  "INDEX_PIP",
    7:  "INDEX_DIP",
    8:  "INDEX_TIP",
    9:  "MIDDLE_MCP",
    10: "MIDDLE_PIP",
    11: "MIDDLE_DIP",
    12: "MIDDLE_TIP",
    13: "RING_MCP",
    14: "RING_PIP",
    15: "RING_DIP",
    16: "RING_TIP",
    17: "PINKY_MCP",
    18: "PINKY_PIP",
    19: "PINKY_DIP",
    20: "PINKY_TIP",
}


# ---------------------------------------------------------------------------
# HandDetector -- encapsulates all MediaPipe Hands logic
# ---------------------------------------------------------------------------
class HandDetector:
    """
    Wraps MediaPipe Hands to detect hand landmarks on a single BGR frame.

    Parameters
    ----------
    max_hands      : int   -- maximum number of hands to detect (we use 1)
    detection_conf : float -- minimum confidence to accept a detection
    tracking_conf  : float -- minimum confidence to continue tracking a hand
    """

    def __init__(self, max_hands=1, detection_conf=0.7, tracking_conf=0.5):
        self.mp_hands   = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        # Green dots for landmarks
        self.landmark_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0), thickness=2, circle_radius=5
        )
        # White lines for connections
        self.connection_spec = self.mp_drawing.DrawingSpec(
            color=(255, 255, 255), thickness=2
        )

    def find_hands(self, frame_bgr):
        """
        Run MediaPipe Hands on one BGR frame.

        MediaPipe requires RGB input, so we convert before processing.
        Returns the annotated frame, a list of 21 landmark dicts, and
        a boolean indicating whether a hand was detected.
        """
        # Convert BGR -> RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        landmark_list   = []
        hand_detected   = False
        annotated_frame = frame_bgr.copy()

        if results.multi_hand_landmarks:
            hand_detected = True
            frame_h, frame_w = annotated_frame.shape[:2]
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw skeleton (connections + landmark dots)
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.landmark_spec,
                self.connection_spec,
            )

            # Extract coordinates and draw IDs
            for idx, lm in enumerate(hand_landmarks.landmark):
                cx = int(lm.x * frame_w)
                cy = int(lm.y * frame_h)
                landmark_list.append({
                    "id":   idx,
                    "name": LANDMARK_NAMES[idx],
                    "x":    round(lm.x, 4),
                    "y":    round(lm.y, 4),
                    "z":    round(lm.z, 4),
                    "cx":   cx,
                    "cy":   cy,
                })
                cv2.putText(
                    annotated_frame, str(idx),
                    (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_PLAIN, 0.9,
                    (255, 255, 255), 1,
                )

        return annotated_frame, landmark_list, hand_detected

    def print_landmarks(self, landmark_list):
        """Print all 21 landmark coordinates to the terminal."""
        id_label   = "ID"
        name_label = "Name"
        x_label    = "X"
        y_label    = "Y"
        z_label    = "Z"
        print("\n--- Hand Landmarks ---")
        print(f"{id_label:<4} {name_label:<15} {x_label:>7} {y_label:>7} {z_label:>8}")
        print("-" * 45)
        id_key   = "id"
        name_key = "name"
        x_key    = "x"
        y_key    = "y"
        z_key    = "z"
        for lm in landmark_list:
            print(
                f"{lm[id_key]:<4} {lm[name_key]:<15} "
                f"{lm[x_key]:>7.4f} {lm[y_key]:>7.4f} {lm[z_key]:>8.4f}"
            )

    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()


# ---------------------------------------------------------------------------
# WebcamStream -- encapsulates OpenCV webcam logic
# ---------------------------------------------------------------------------
class WebcamStream:
    """Manages opening, reading, and releasing the webcam."""

    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open webcam at index {camera_index}.")
        print(f"Webcam opened successfully (index {camera_index}).")

    def read_frame(self):
        """Capture one frame. Returns (ret, frame)."""
        return self.cap.read()

    def release(self):
        """Release the capture and close all OpenCV windows."""
        self.cap.release()
        cv2.destroyAllWindows()
        print("Webcam released. All windows closed.")


# ---------------------------------------------------------------------------
# FPSCounter
# ---------------------------------------------------------------------------
class FPSCounter:
    """Calculates FPS based on elapsed time between frames."""

    def __init__(self):
        self._prev_time = time.time()
        self.fps = 0.0

    def update(self):
        """Call once per frame. Returns current FPS."""
        now = time.time()
        delta = now - self._prev_time
        self.fps = 1.0 / delta if delta > 0 else 0.0
        self._prev_time = now
        return self.fps


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------
def draw_fps(frame, fps):
    """Draw FPS counter top-left (yellow-green)."""
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)


def draw_no_hand_message(frame):
    """Display centred red No Hand Detected text."""
    h, w = frame.shape[:2]
    text = "No Hand Detected"
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    origin = ((w - tw) // 2, (h + th) // 2)
    cv2.putText(frame, text, origin, font, scale, (0, 0, 255), thickness)


def draw_title(frame):
    """Draw status bar at bottom of frame."""
    h = frame.shape[0]
    cv2.putText(frame, "Virtual Sign Language Tutor  |  Press Q to quit",
                (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    try:
        webcam = WebcamStream(camera_index=0)
    except IOError as err:
        print(f"FATAL: {err}")
        return

    detector    = HandDetector(max_hands=1, detection_conf=0.7, tracking_conf=0.5)
    fps_counter = FPSCounter()
    print("Hand detection running. Press Q to quit.\n")

    while True:
        ret, frame = webcam.read_frame()
        if not ret:
            print("ERROR: Failed to read frame.")
            break

        # Mirror effect
        frame = cv2.flip(frame, 1)

        # Detect hand and annotate frame
        annotated_frame, landmark_list, hand_detected = detector.find_hands(frame)

        if hand_detected:
            detector.print_landmarks(landmark_list)

        fps = fps_counter.update()
        draw_fps(annotated_frame, fps)
        draw_title(annotated_frame)

        if not hand_detected:
            draw_no_hand_message(annotated_frame)

        cv2.imshow("Virtual Sign Language Tutor -- Hand Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\nQ pressed -- shutting down.")
            break

    detector.close()
    webcam.release()


if __name__ == "__main__":
    main()

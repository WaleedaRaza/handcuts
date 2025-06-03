import cv2
import mediapipe as mp
import pyautogui
import time
import math
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# PyAutoGUI settings
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.01

# Configuration
CONFIG = {
    "MIN_DETECTION_CONF": 0.75,  # Mediapipe detection confidence
    "MIN_TRACKING_CONF": 0.6,    # Mediapipe tracking confidence
    "THUMB_ANGLE_THRESHOLD": 160.0,  # Stricter for thumb extension
    "FIST_DISTANCE_THRESHOLD": 0.8,  # Fingers must be close to wrist for fist
    "COOLDOWN_SECONDS": 0.5,     # Time between gestures
}

# macOS key mappings
KEY_MAP = {
    "swipe_right": ("ctrl", "right"),  # Next desktop
    "swipe_left": ("ctrl", "left"),    # Previous desktop
    "copy": ("command", "c"),
    "select_all": ("command", "a"),
    "paste": ("command", "v"),
}

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def distance(a, b):
    """Calculate Euclidean distance between two landmarks."""
    return math.hypot(a.x - b.x, a.y - b.y)

def angle_between(a, b, c):
    """Calculate angle (degrees) at point b formed by segments ab and bc."""
    ab = (a.x - b.x, a.y - b.y)
    cb = (c.x - b.x, c.y - b.y)
    dot = ab[0] * cb[0] + ab[1] * cb[1]
    mag_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
    mag_cb = math.sqrt(cb[0] ** 2 + cb[1] ** 2)
    if mag_ab * mag_cb == 0:
        return 0.0
    cos_ = dot / (mag_ab * mag_cb)
    cos_ = max(min(cos_, 1.0), -1.0)
    return math.degrees(math.acos(cos_))

def get_finger_states(landmarks):
    """
    Returns a list: [thumb, index, middle, ring, pinky]
    1 = up (extended), 0 = down (folded)
    """
    states = [0] * 5
    wrist = landmarks[0]

    # Thumb: Use angle at CMC (1) and MCP (2) to detect extension
    thumb_cmc = landmarks[1]
    thumb_mcp = landmarks[2]
    thumb_tip = landmarks[4]
    thumb_angle = angle_between(thumb_cmc, thumb_mcp, thumb_tip)
    states[0] = 1 if thumb_angle > CONFIG["THUMB_ANGLE_THRESHOLD"] else 0

    # Other fingers: Use distance heuristic (tip farther from wrist than MCP)
    finger_tips = [8, 12, 16, 20]  # Index, middle, ring, pinky tips
    finger_mcp = [5, 9, 13, 17]    # Corresponding MCP joints
    for i, (tip, base) in enumerate(zip(finger_tips, finger_mcp), 1):
        tip_dist = distance(landmarks[tip], wrist)
        base_dist = distance(landmarks[base], wrist)
        states[i] = 1 if tip_dist > base_dist * 1.2 else 0

    return states

def is_thumb_fist(landmarks, fingers):
    """Thumb up, all other fingers tightly folded (fist)."""
    if fingers != [1, 0, 0, 0, 0]:
        return False
    # Ensure index, middle, ring, pinky tips are close to wrist (tight fist)
    wrist = landmarks[0]
    finger_tips = [8, 12, 16, 20]
    finger_mcp = [5, 9, 13, 17]
    for tip, base in zip(finger_tips, finger_mcp):
        tip_dist = distance(landmarks[tip], wrist)
        base_dist = distance(landmarks[base], wrist)
        if tip_dist > base_dist * CONFIG["FIST_DISTANCE_THRESHOLD"]:
            return False
    return True

def get_thumb_direction(landmarks):
    """Determine thumb direction (left/right) based on tip relative to wrist."""
    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    dx = thumb_tip.x - wrist.x
    return "right" if dx > 0 else "left"

def is_index_only(fingers):
    return fingers == [0, 1, 0, 0, 0]

def is_three_finger_select(fingers):
    return fingers == [0, 1, 1, 1, 0]

def is_pinky_only(fingers):
    return fingers == [0, 0, 0, 0, 1]

def execute_action(action):
    """Execute action with error handling."""
    if not action:
        return
    try:
        keys = KEY_MAP.get(action)
        if keys:
            pyautogui.hotkey(*keys)
            logging.info(f"Executed action: {action}")
    except Exception as e:
        logging.error(f"Action failed: {action}, Error: {e}")

def main():
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=CONFIG["MIN_DETECTION_CONF"],
        min_tracking_confidence=CONFIG["MIN_TRACKING_CONF"],
    )
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Failed to open camera")
        return

    prev_action = ""
    last_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to read frame")
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            gesture_text = "No Gesture"
            finger_text = ""
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmarks = hand_landmarks.landmark
                    fingers = get_finger_states(landmarks)
                    logging.debug(f"Finger states: {fingers}")
                    finger_text = f"Fingers: {fingers}"

                    current_time = time.time()
                    if current_time - last_time > CONFIG["COOLDOWN_SECONDS"]:
                        # Prioritize other gestures over thumb-fist
                        if is_index_only(fingers):
                            execute_action("copy")
                            gesture_text = "Copy"
                            prev_action = "copy"
                            last_time = current_time

                        elif is_three_finger_select(fingers):
                            execute_action("select_all")
                            gesture_text = "Select All"
                            prev_action = "select_all"
                            last_time = current_time

                        elif is_pinky_only(fingers):
                            execute_action("paste")
                            gesture_text = "Paste"
                            prev_action = "paste"
                            last_time = current_time

                        elif is_thumb_fist(landmarks, fingers):
                            direction = get_thumb_direction(landmarks)
                            action = f"swipe_{direction}"
                            execute_action(action)
                            gesture_text = f"Swipe {direction.capitalize()}"
                            prev_action = action
                            last_time = current_time

            # Display gesture and finger state feedback
            cv2.putText(frame, f"Action: {gesture_text}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, finger_text, (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Hand Gesture Control", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break
    except KeyboardInterrupt:
        logging.info("Program terminated")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()

if __name__ == "__main__":
    main()

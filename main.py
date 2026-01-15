import pyautogui
import cv2
import numpy as np
import time
import os
import sys
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

# -------------------- SAFETY --------------------
pyautogui.FAILSAFE = True  # move mouse to top-left corner to abort
pyautogui.PAUSE = 0.05

# -------------------- PATHS --------------------
TARGETS = {
    "I AM AVAILABLE": r"data/i'm_available.png",
    "OK": r"data/ok.png"
}
SCREENSHOT_DIR = r"data/screenshots"
LOG_DIR = r"data/logs"

os.makedirs(SCREENSHOT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# -------------------- LOGGING --------------------
LOG_FILE = os.path.join(LOG_DIR, "screen_monitor.log")
logger = logging.getLogger("ScreenMonitor")
logger.setLevel(logging.INFO)

# Prevent duplicate handlers if script is re-run in same interpreter
if not logger.handlers:
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    try:
        console_handler.stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

# -------------------- LOAD TARGETS --------------------
templates = {}
for name, path in TARGETS.items():
    img = cv2.imread(path)
    if img is None:
        logger.error(f"Target image not found: {path}")
        raise FileNotFoundError(path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    templates[name] = {"edges": edges, "shape": edges.shape[:2]}

logger.info(f"Loaded target templates: {', '.join(TARGETS.keys())}")

# -------------------- CONFIG --------------------
MATCH_THRESHOLD = 0.70
LOOP_SLEEP_SEC = 30

# Avoid rapid repeated clicks if UI lags
CLICK_COOLDOWN_SEC = 15
_last_click_ts = {k: 0.0 for k in TARGETS.keys()}

# -------------------- FUNCTIONS --------------------
def capture_screen() -> str:
    """Capture the current screen and save it with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(SCREENSHOT_DIR, f"screenshot_{timestamp}.png")
    screenshot = pyautogui.screenshot()
    screenshot.save(filepath)
    logger.info(f"Screenshot captured: {filepath}")
    return filepath

def multi_scale_match(screen_gray_edges: np.ndarray, target_edges: np.ndarray):
    """Perform robust multi-scale template matching on edge maps."""
    (tH, tW) = target_edges.shape[:2]
    found = None

    for scale in np.linspace(0.8, 1.2, 11):  # 80–120% zoom
        resized = cv2.resize(
            screen_gray_edges, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        )
        r = screen_gray_edges.shape[1] / float(resized.shape[1])

        if resized.shape[0] < tH or resized.shape[1] < tW:
            continue

        result = cv2.matchTemplate(resized, target_edges, cv2.TM_CCOEFF_NORMED)
        _, maxVal, _, maxLoc = cv2.minMaxLoc(result)

        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)

    return found  # (confidence, location, ratio)

def find_best_target(screen_bgr: np.ndarray):
    """Return best target match info across all templates."""
    screen_gray = cv2.cvtColor(screen_bgr, cv2.COLOR_BGR2GRAY)
    screen_edges = cv2.Canny(screen_gray, 50, 150)

    best_target = None
    best_conf = 0.0
    best_coords = None

    for name, t in templates.items():
        found = multi_scale_match(screen_edges, t["edges"])
        if found:
            maxVal, maxLoc, r = found
            if maxVal > best_conf:
                best_conf = maxVal
                best_target = name
                best_coords = (maxLoc, r, t["shape"])

    return best_target, best_conf, best_coords

def click_target(best_target: str, best_conf: float, best_coords):
    """Click the best matched target if above threshold and not in cooldown."""
    if not best_target or best_conf < MATCH_THRESHOLD:
        logger.info(f"No button found. Best confidence={best_conf:.3f}")
        return False

    now = time.time()
    if (now - _last_click_ts.get(best_target, 0.0)) < CLICK_COOLDOWN_SEC:
        logger.info(
            f"Cooldown active for '{best_target}'. Skipping click. conf={best_conf:.3f}"
        )
        return False

    maxLoc, r, (tH, tW) = best_coords
    startX, startY = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    endX, endY = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    x_center = startX + (endX - startX) // 2
    y_center = startY + (endY - startY) // 2

    pyautogui.click(x_center, y_center)
    _last_click_ts[best_target] = now

    logger.info(f"Clicked '{best_target}' at ({x_center}, {y_center}) [conf={best_conf:.3f}]")
    return True

def find_and_click(screenshot_path: str) -> bool:
    """Load screenshot, find best target, click if valid."""
    screen = cv2.imread(screenshot_path)
    if screen is None:
        logger.warning(f"Failed to read screenshot: {screenshot_path}")
        return False

    best_target, best_conf, best_coords = find_best_target(screen)
    # Log the best candidate even if below threshold
    if best_target:
        logger.info(f"Best match: '{best_target}' conf={best_conf:.3f}")
    return click_target(best_target, best_conf, best_coords)

# -------------------- MAIN LOOP --------------------
if __name__ == "__main__":
    logger.info("Screen monitoring started. Press Ctrl+C to stop.")
    while True:
        try:
            screenshot_path = capture_screen()
            find_and_click(screenshot_path)
            time.sleep(LOOP_SLEEP_SEC)

        except KeyboardInterrupt:
            logger.info("Script stopped by user.")
            break

        except pyautogui.FailSafeException:
            logger.warning("PyAutoGUI FAILSAFE triggered (mouse moved to top-left). Exiting.")
            break

        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            time.sleep(LOOP_SLEEP_SEC)

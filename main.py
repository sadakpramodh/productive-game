import pyautogui
import cv2
import numpy as np
import time
import os
import sys
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler

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

file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8')
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
console_handler.stream.reconfigure(encoding='utf-8', errors='replace')

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# -------------------- LOAD TARGETS --------------------
templates = {}
for name, path in TARGETS.items():
    img = cv2.imread(path)
    if img is None:
        logger.error(f"❌ Target image not found: {path}")
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    templates[name] = {
        "edges": edges,
        "shape": edges.shape[:2]
    }
logger.info(f"✅ Loaded target templates: {', '.join(TARGETS.keys())}")

# -------------------- FUNCTIONS --------------------
def capture_screen():
    """Capture the current screen and save it with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(SCREENSHOT_DIR, f"screenshot_{timestamp}.png")
    screenshot = pyautogui.screenshot()
    screenshot.save(filepath)
    logger.info(f"🖼️ Screenshot captured: {filepath}")
    return filepath

def multi_scale_match(screen_gray, target_edges):
    """Perform robust multi-scale template matching."""
    (tH, tW) = target_edges.shape[:2]
    found = None
    for scale in np.linspace(0.8, 1.2, 11):  # 80–120% zoom
        resized = cv2.resize(screen_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        r = screen_gray.shape[1] / float(resized.shape[1])
        if resized.shape[0] < tH or resized.shape[1] < tW:
            continue

        result = cv2.matchTemplate(resized, target_edges, cv2.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)
    return found

def find_and_click(screenshot_path):
    """Check for all targets and click whichever is found."""
    screen = cv2.imread(screenshot_path)
    if screen is None:
        logger.warning(f"⚠️ Failed to read screenshot: {screenshot_path}")
        return False

    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    screen_edges = cv2.Canny(screen_gray, 50, 150)

    best_target = None
    best_conf = 0
    best_coords = None

    # Try all templates and keep the best match
    for name, t in templates.items():
        found = multi_scale_match(screen_edges, t["edges"])
        if found:
            (maxVal, maxLoc, r) = found
            if maxVal > best_conf:
                best_conf, best_target, best_coords = maxVal, name, (maxLoc, r, t["shape"])

    if best_target and best_conf >= 0.70:
        maxLoc, r, (tH, tW) = best_coords
        startX, startY = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        endX, endY = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
        x_center, y_center = (startX + (endX - startX) // 2, startY + (endY - startY) // 2)

        pyautogui.click(x_center, y_center)
        logger.info(f"✅ Clicked '{best_target}' at ({x_center}, {y_center}) [conf={best_conf:.3f}]")
        return True
    else:
        logger.info(f"❌ No button found. Best confidence={best_conf:.3f}")
        return False

# -------------------- MAIN LOOP --------------------
if __name__ == "__main__":
    logger.info("🔄 Screen monitoring started. Press Ctrl+C to stop.")
    while True:
        try:
            screenshot_path = capture_screen()
            find_and_click(screenshot_path)
            time.sleep(30)
        except KeyboardInterrupt:
            logger.info("🛑 Script stopped by user.")
            break
        except Exception as e:
            logger.exception(f"⚠️ Unexpected error: {e}")
            time.sleep(30)

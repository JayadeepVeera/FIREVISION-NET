#!/usr/bin/env python3
import os
import time
import threading
import sqlite3
from pathlib import Path
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import requests

try:
    import winsound
except ImportError:
    winsound = None

from backend.configs.settings import (
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    TELEGRAM_COOLDOWN_SECONDS,
    SQLITE_DB_PATH,
    CAM_ID,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    FPS,
    DB_ENABLED,
    TELEGRAM_ENABLED,
)

ALERT_SAVE_DIR = "backend/storage/alerts"
SAVE_ALERT_FRAMES = True
RECONNECT_INTERVAL_SEC = 2.0
MAX_READ_FAILS_BEFORE_RECONNECT = 20
STATUS_DEBOUNCE_SECONDS = 1.2
FRAME_SAVE_COOLDOWN_SECONDS = 10.0

ALARM_ENABLED = True
ALARM_BEEP_MS = 700
ALARM_BEEP_FREQ = 1800
ALARM_REPEAT_GAP = 0.45

DISPLAY_IOU_THRESHOLD = 0.10
MIN_FIRE_CONFIRM_FRAMES = 3
MIN_SMOKE_CONFIRM_FRAMES = 4


def status_to_message(status, fps=None):
    if fps is None:
        return f"FireVisionNet detected: {status}"
    return f"FireVisionNet detected: {status} | FPS={fps:.1f}"


class TelegramAlert:
    def __init__(self, bot_token, chat_id, cooldown=15, enabled=True):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.cooldown = cooldown
        self.enabled = enabled
        self.last_sent_at = {}

    def _validate(self):
        if not self.enabled:
            return False, "Telegram alerts are disabled"
        if not self.bot_token:
            return False, "Missing TELEGRAM_BOT_TOKEN"
        if not self.chat_id:
            return False, "Missing TELEGRAM_CHAT_ID"
        return True, "OK"

    def send_message(self, message):
        ok, reason = self._validate()
        if not ok:
            return False, reason

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": message}

        try:
            response = requests.post(url, json=payload, timeout=15)
            data = response.json()
            if response.status_code == 200 and data.get("ok"):
                return True, "Telegram message sent successfully"
            return False, f"Telegram API error: {data}"
        except Exception as e:
            return False, f"Telegram exception: {e}"

    def send_alert_once(self, alert_key, message):
        now = time.time()
        last_time = self.last_sent_at.get(alert_key)

        if last_time is not None:
            elapsed = now - last_time
            if elapsed < self.cooldown:
                return False, f"Cooldown active for {alert_key}: {self.cooldown - elapsed:.1f}s remaining"

        ok, info = self.send_message(message)
        if ok:
            self.last_sent_at[alert_key] = now
        return ok, info

    def reset_all(self):
        self.last_sent_at.clear()


class EventLogger:
    def __init__(self, db_path="firevision.db", enabled=True):
        self.enabled = enabled
        self.db_path = db_path
        self.conn = None

        if not self.enabled:
            return

        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            self._create_tables()
        except Exception as e:
            print(f"[WARN] Logger init failed: {e}")
            self.conn = None
            self.enabled = False

    def _create_tables(self):
        if self.conn is None:
            return

        query = """
        CREATE TABLE IF NOT EXISTS firevision_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL,
            source TEXT DEFAULT 'web_camera',
            fps REAL,
            extra_text TEXT
        );
        """
        cur = self.conn.cursor()
        cur.execute(query)
        self.conn.commit()

    def log_event(self, status, fps=None, source="web_camera", extra_text=""):
        if not self.enabled or self.conn is None:
            return

        query = """
        INSERT INTO firevision_events (created_at, status, source, fps, extra_text)
        VALUES (?, ?, ?, ?, ?);
        """
        try:
            cur = self.conn.cursor()
            cur.execute(
                query,
                (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status, source, fps, extra_text)
            )
            self.conn.commit()
        except Exception as e:
            print(f"[WARN] log_event failed: {e}")

    def close(self):
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
            self.conn = None


class CameraReader:
    def __init__(self, cam_id=0, width=1280, height=720, fps=30):
        self.cam_id = cam_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None
        self.lock = threading.Lock()
        self.frame = None
        self.running = False
        self.thread = None
        self.last_frame_time = 0.0
        self.read_fail_count = 0

    def _open_camera(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass

        self.cap = cv2.VideoCapture(self.cam_id, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        opened = self.cap.isOpened()
        if opened:
            self.read_fail_count = 0
            self.last_frame_time = time.time()
            print(f"[INFO] Camera connected: cam_id={self.cam_id}")
        else:
            print(f"[WARN] Camera open failed: cam_id={self.cam_id}")
        return opened

    def reconnect(self):
        print("[WARN] Attempting camera reconnect...")
        ok = self._open_camera()
        if ok:
            print("[INFO] Camera reconnect successful.")
        else:
            print("[WARN] Camera reconnect failed.")
        return ok

    def start(self):
        self.running = True
        self._open_camera()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        last_reconnect_try = 0.0

        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    now = time.time()
                    if now - last_reconnect_try >= RECONNECT_INTERVAL_SEC:
                        self.reconnect()
                        last_reconnect_try = now
                    time.sleep(0.1)
                    continue

                ok, frame = self.cap.read()
                if ok and frame is not None:
                    with self.lock:
                        self.frame = frame
                    self.last_frame_time = time.time()
                    self.read_fail_count = 0
                else:
                    self.read_fail_count += 1
                    time.sleep(0.03)

                    stale = (time.time() - self.last_frame_time) > RECONNECT_INTERVAL_SEC
                    if self.read_fail_count >= MAX_READ_FAILS_BEFORE_RECONNECT or stale:
                        now = time.time()
                        if now - last_reconnect_try >= RECONNECT_INTERVAL_SEC:
                            self.reconnect()
                            last_reconnect_try = now
            except Exception as e:
                print(f"[WARN] CameraReader update error: {e}")
                time.sleep(0.1)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.5)
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        print("[INFO] Camera released.")


class TrackMemory:
    def __init__(self, maxlen=14):
        self.maxlen = maxlen
        self.data = {}

    def _key(self, box):
        x, y, w, h = box
        return (x // 20, y // 20, w // 20, h // 20)

    def update(self, box, metrics):
        k = self._key(box)
        if k not in self.data:
            self.data[k] = deque(maxlen=self.maxlen)
        self.data[k].append(metrics)

    def get(self, box):
        return self.data.get(self._key(box), deque())

    def cleanup(self, live_boxes):
        live_keys = {self._key(b) for b in live_boxes}
        dead = [k for k in list(self.data.keys()) if k not in live_keys]
        for k in dead:
            del self.data[k]


class FireVisionNet:
    def __init__(self):
        self.prev_gray = None
        self.running = False
        self.cam = None

        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=400,
            varThreshold=22,
            detectShadows=False
        )

        self.fire_patch_cache = {}
        self.smoke_patch_cache = {}
        self.screen_patch_cache = {}

        self.real_fire_tracks = TrackMemory(maxlen=14)
        self.real_smoke_tracks = TrackMemory(maxlen=14)
        self.fake_fire_tracks = TrackMemory(maxlen=10)
        self.fake_smoke_tracks = TrackMemory(maxlen=10)

        self.real_fire_hist = deque(maxlen=14)
        self.real_smoke_hist = deque(maxlen=14)

        self.last_alert_print = 0.0
        self.last_status = "SAFE"
        self.last_stable_status = "SAFE"
        self.status_first_seen_at = None
        self.last_frame_save_at = 0.0

        self.alarm_thread = None
        self.alarm_stop_event = threading.Event()

        Path(ALERT_SAVE_DIR).mkdir(parents=True, exist_ok=True)

        self.telegram = TelegramAlert(
            bot_token=TELEGRAM_BOT_TOKEN,
            chat_id=TELEGRAM_CHAT_ID,
            cooldown=TELEGRAM_COOLDOWN_SECONDS,
            enabled=TELEGRAM_ENABLED
        )

        self.logger = EventLogger(
            db_path=SQLITE_DB_PATH,
            enabled=DB_ENABLED
        )

    def stop(self):
        self.running = False
        try:
            self.stop_alarm()
        except Exception:
            pass

        if self.cam is not None:
            try:
                self.cam.release()
            except Exception:
                pass
            self.cam = None

        try:
            self.logger.close()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        print("[INFO] FireVisionNet stopped cleanly.")

    def enhance_frame(self, frame):
        frame = cv2.bilateralFilter(frame, 5, 35, 35)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        merged = cv2.merge([l, a, b])
        out = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        out = cv2.convertScaleAbs(out, alpha=1.02, beta=0)
        return out

    def start_alarm(self):
        if not ALARM_ENABLED:
            return
        if self.alarm_thread is not None and self.alarm_thread.is_alive():
            return
        self.alarm_stop_event.clear()
        self.alarm_thread = threading.Thread(target=self._alarm_loop, daemon=True)
        self.alarm_thread.start()

    def _alarm_loop(self):
        while not self.alarm_stop_event.is_set():
            try:
                if winsound is not None:
                    winsound.Beep(ALARM_BEEP_FREQ, ALARM_BEEP_MS)
                else:
                    print("\a", end="", flush=True)
                    time.sleep(ALARM_BEEP_MS / 1000.0)
                time.sleep(ALARM_REPEAT_GAP)
            except Exception:
                time.sleep(0.2)

    def stop_alarm(self):
        self.alarm_stop_event.set()
        if self.alarm_thread is not None:
            self.alarm_thread.join(timeout=1.0)
            self.alarm_thread = None

    def clamp_box(self, box, shape):
        h, w = shape[:2]
        x, y, bw, bh = box
        x = max(0, x)
        y = max(0, y)
        bw = max(1, min(bw, w - x))
        bh = max(1, min(bh, h - y))
        return (x, y, bw, bh)

    def iou(self, a, b):
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        x1 = max(ax, bx)
        y1 = max(ay, by)
        x2 = min(ax + aw, bx + bw)
        y2 = min(ay + ah, by + bh)
        iw = max(0, x2 - x1)
        ih = max(0, y2 - y1)
        inter = iw * ih
        union = (aw * ah) + (bw * bh) - inter + 1e-6
        return inter / union

    def merge_boxes(self, boxes, iou_thresh=0.28):
        if not boxes:
            return []

        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        merged = []

        while boxes:
            base = boxes.pop(0)
            bx, by, bw, bh = base
            changed = True

            while changed:
                changed = False
                keep = []
                for other in boxes:
                    if self.iou((bx, by, bw, bh), other) >= iou_thresh:
                        ox, oy, ow, oh = other
                        x1 = min(bx, ox)
                        y1 = min(by, oy)
                        x2 = max(bx + bw, ox + ow)
                        y2 = max(by + bh, oy + oh)
                        bx, by, bw, bh = x1, y1, x2 - x1, y2 - y1
                        changed = True
                    else:
                        keep.append(other)
                boxes = keep

            merged.append((bx, by, bw, bh))

        return merged

    def detect_displays(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 7, 50, 50)
        edges = cv2.Canny(blur, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 2500:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            rect = cv2.minAreaRect(cnt)
            rw, rh = rect[1]
            if rw < 50 or rh < 50:
                continue

            rect_area = max(rw * rh, 1.0)
            rect_ratio = max(rw, rh) / (min(rw, rh) + 1e-6)
            contour_fill = area / rect_area

            x, y, w, h = cv2.boundingRect(cnt)
            if w < 60 or h < 60:
                continue

            bbox_ratio = w / float(h + 1e-6)

            cond_quad = len(approx) == 4 and 0.45 <= bbox_ratio <= 2.4 and contour_fill > 0.55
            cond_rect = 0.45 <= rect_ratio <= 2.6 and contour_fill > 0.70

            if cond_quad or cond_rect:
                boxes.append((x, y, w, h))

        return self.merge_boxes(boxes, iou_thresh=0.30)

    def fire_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bgr = frame.astype(np.float32)

        b = bgr[:, :, 0]
        g = bgr[:, :, 1]
        r = bgr[:, :, 2]

        h = hsv[:, :, 0].astype(np.float32)
        s = hsv[:, :, 1].astype(np.float32)
        v = hsv[:, :, 2].astype(np.float32)

        cond = (
            (r > 155) &
            (r >= g * 1.10) &
            (g >= b * 0.72) &
            (((h <= 32) | (h >= 175))) &
            (s > 95) &
            (v > 150) &
            ((r - b) > 42)
        )

        mask = cond.astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        return mask

    def smoke_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        low_sat = cv2.inRange(
            hsv,
            np.array([0, 0, 80], dtype=np.uint8),
            np.array([180, 42, 205], dtype=np.uint8)
        )

        blur = cv2.GaussianBlur(gray, (11, 11), 0)
        soft = cv2.absdiff(gray, blur)
        soft = cv2.threshold(soft, 10, 255, cv2.THRESH_BINARY_INV)[1]

        motion_like = cv2.bitwise_and(low_sat, soft)
        motion_like = cv2.morphologyEx(motion_like, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        motion_like = cv2.erode(motion_like, np.ones((3, 3), np.uint8), iterations=1)
        return motion_like

    def motion_mask(self, frame):
        fg = self.bg_sub.apply(frame)
        fg = cv2.threshold(fg, 220, 255, cv2.THRESH_BINARY)[1]
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.erode(fg, np.ones((3, 3), np.uint8), iterations=1)
        fg = cv2.dilate(fg, np.ones((3, 3), np.uint8), iterations=1)
        return fg

    def find_regions(self, mask, min_area=50):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))
        return self.merge_boxes(boxes, iou_thresh=0.25)

    def region_ratio(self, mask, box):
        x, y, w, h = box
        roi = mask[y:y + h, x:x + w]
        if roi.size == 0:
            return 0.0
        return float(np.count_nonzero(roi) / ((w * h) + 1e-6))

    def motion_score(self, gray, box):
        if self.prev_gray is None:
            return 0.0
        x, y, w, h = box
        prev_roi = self.prev_gray[y:y + h, x:x + w]
        curr_roi = gray[y:y + h, x:x + w]
        if prev_roi.size == 0 or curr_roi.size == 0 or prev_roi.shape != curr_roi.shape:
            return 0.0
        return float(np.mean(cv2.absdiff(prev_roi, curr_roi)))

    def upward_motion_score(self, gray, box):
        if self.prev_gray is None:
            return 0.0

        x, y, w, h = box
        if w < 8 or h < 8:
            return 0.0

        prev_roi = self.prev_gray[y:y + h, x:x + w]
        curr_roi = gray[y:y + h, x:x + w]
        if prev_roi.size == 0 or curr_roi.size == 0 or prev_roi.shape != curr_roi.shape:
            return 0.0

        pts = cv2.goodFeaturesToTrack(prev_roi, maxCorners=18, qualityLevel=0.01, minDistance=3)
        if pts is None:
            return 0.0

        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_roi, curr_roi, pts, None)
        if nxt is None or st is None:
            return 0.0

        upward_vals = []
        for p0, p1, s in zip(pts, nxt, st):
            if s[0] == 1:
                dy = p1[0][1] - p0[0][1]
                if dy < 0:
                    upward_vals.append(-dy)

        if not upward_vals:
            return 0.0
        return float(np.mean(upward_vals))

    def patch_temporal_score(self, mask, box, cache, prefix):
        x, y, w, h = box
        roi = mask[y:y + h, x:x + w]
        if roi.size == 0:
            return 0.0

        small = cv2.resize(roi, (24, 24), interpolation=cv2.INTER_AREA)
        key = f"{prefix}_{x//16}_{y//16}_{w//16}_{h//16}"

        if key not in cache:
            cache[key] = small
            return 0.0

        prev = cache[key]
        if prev.shape != small.shape:
            cache[key] = small
            return 0.0

        diff = cv2.absdiff(prev, small)
        cache[key] = small
        return float(np.mean(diff))

    def edge_density(self, gray, box):
        x, y, w, h = box
        roi = gray[y:y + h, x:x + w]
        if roi.size == 0:
            return 0.0
        edges = cv2.Canny(roi, 60, 160)
        return float(np.count_nonzero(edges) / (roi.size + 1e-6))

    def border_edge_ratio(self, gray, box, border=4):
        x, y, w, h = box
        roi = gray[y:y + h, x:x + w]
        if roi.size == 0 or w < 14 or h < 14:
            return 0.0

        edges = cv2.Canny(roi, 60, 160)
        mask = np.zeros_like(edges)
        b = min(border, max(1, min(w, h) // 6))
        mask[:b, :] = 255
        mask[-b:, :] = 255
        mask[:, :b] = 255
        mask[:, -b:] = 255

        border_edges = cv2.bitwise_and(edges, mask)
        return float(np.count_nonzero(border_edges) / (np.count_nonzero(edges) + 1e-6))

    def color_flatness_score(self, frame, box):
        x, y, w, h = box
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            return 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        s_std = np.std(hsv[:, :, 1])
        v_std = np.std(hsv[:, :, 2])
        flat = max(0.0, 1.0 - ((s_std + v_std) / 90.0))
        return float(min(1.0, flat))

    def brightness_std(self, gray, box):
        x, y, w, h = box
        roi = gray[y:y + h, x:x + w]
        if roi.size == 0:
            return 0.0
        return float(np.std(roi))

    def rectangularity_score(self, box, display_boxes):
        if not display_boxes:
            return 0.0
        vals = [self.iou(box, d) for d in display_boxes]
        return float(max(vals)) if vals else 0.0

    def temporal_screen_score(self, frame_small, box_small):
        x, y, w, h = box_small
        roi = frame_small[y:y + h, x:x + w]
        if roi.size == 0:
            return 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        key = f"screen_{x//12}_{y//12}_{w//12}_{h//12}"

        if key not in self.screen_patch_cache:
            self.screen_patch_cache[key] = small
            return 0.0

        prev = self.screen_patch_cache[key]
        self.screen_patch_cache[key] = small
        diff = cv2.absdiff(prev, small)
        return float(np.mean(diff))

    def haze_score(self, gray, box):
        x, y, w, h = box
        roi = gray[y:y + h, x:x + w]
        if roi.size == 0:
            return 0.0

        local_std = float(np.std(roi))
        lap_var = float(cv2.Laplacian(roi, cv2.CV_64F).var())
        mean_val = float(np.mean(roi))

        haze = 0.0
        if local_std < 18:
            haze += 0.45
        if lap_var < 35:
            haze += 0.35
        if 85 <= mean_val <= 180:
            haze += 0.20
        return float(min(1.0, haze))

    def fire_metrics(self, frame, gray, box, fire_mask_full, display_boxes, proc_small, sx, sy):
        x, y, w, h = box
        roi_gray = gray[y:y + h, x:x + w]
        bright = float(np.mean(roi_gray)) if roi_gray.size > 0 else 0.0

        small_box = (
            int(x / sx), int(y / sy),
            max(1, int(w / sx)), max(1, int(h / sy))
        )

        return {
            "fire_ratio": self.region_ratio(fire_mask_full, box),
            "motion": self.motion_score(gray, box),
            "upward": self.upward_motion_score(gray, box),
            "flicker": self.patch_temporal_score(fire_mask_full, box, self.fire_patch_cache, "fire"),
            "bright": bright,
            "area": w * h,
            "edge_density": self.edge_density(gray, box),
            "border_edge_ratio": self.border_edge_ratio(gray, box),
            "flatness": self.color_flatness_score(frame, box),
            "rectangularity": self.rectangularity_score(box, display_boxes),
            "screen_temporal": self.temporal_screen_score(proc_small, small_box),
            "brightness_std": self.brightness_std(gray, box),
        }

    def smoke_metrics(self, frame, gray, box, smoke_mask_full, display_boxes, proc_small, sx, sy):
        small_box = (
            int(box[0] / sx), int(box[1] / sy),
            max(1, int(box[2] / sx)), max(1, int(box[3] / sy))
        )

        return {
            "smoke_ratio": self.region_ratio(smoke_mask_full, box),
            "motion": self.motion_score(gray, box),
            "upward": self.upward_motion_score(gray, box),
            "temporal": self.patch_temporal_score(smoke_mask_full, box, self.smoke_patch_cache, "smoke"),
            "area": box[2] * box[3],
            "edge_density": self.edge_density(gray, box),
            "border_edge_ratio": self.border_edge_ratio(gray, box),
            "flatness": self.color_flatness_score(frame, box),
            "rectangularity": self.rectangularity_score(box, display_boxes),
            "screen_temporal": self.temporal_screen_score(proc_small, small_box),
            "brightness_std": self.brightness_std(gray, box),
            "haze": self.haze_score(gray, box),
        }

    def is_fake_fire_metric(self, m):
        return (
            m["rectangularity"] >= 0.10 or
            m["screen_temporal"] >= 5.0 or
            (m["border_edge_ratio"] >= 0.34 and m["flatness"] >= 0.40) or
            (m["brightness_std"] < 20 and m["flatness"] >= 0.45)
        )

    def is_fake_smoke_metric(self, m):
        return (
            m["rectangularity"] >= 0.10 or
            m["screen_temporal"] >= 4.5 or
            m["haze"] >= 0.70 or
            (
                m["motion"] < 0.75 and
                m["upward"] < 0.02 and
                m["temporal"] < 0.45
            ) or
            (
                m["brightness_std"] < 16 and
                m["edge_density"] < 0.035
            )
        )

    def is_real_fire_metric(self, m):
        return (
            m["area"] >= 80 and
            m["fire_ratio"] >= 0.16 and
            m["motion"] >= 1.8 and
            m["upward"] >= 0.04 and
            m["flicker"] >= 0.85 and
            m["bright"] >= 125 and
            m["border_edge_ratio"] < 0.28 and
            m["flatness"] < 0.58 and
            m["rectangularity"] < 0.08 and
            m["brightness_std"] >= 18 and
            not self.is_fake_fire_metric(m)
        )

    def is_real_smoke_metric(self, m):
        return (
            m["area"] >= 180 and
            m["smoke_ratio"] >= 0.11 and
            0.8 <= m["motion"] <= 10.0 and
            m["upward"] >= 0.03 and
            m["temporal"] >= 0.55 and
            m["border_edge_ratio"] < 0.24 and
            m["flatness"] < 0.62 and
            m["rectangularity"] < 0.08 and
            m["brightness_std"] >= 14 and
            not self.is_fake_smoke_metric(m)
        )

    def safe_send_telegram(self, alert_key, msg):
        try:
            ok, info = self.telegram.send_alert_once(alert_key, msg)
            if ok:
                print(f"[INFO] Telegram sent: {alert_key}")
            else:
                print(f"[INFO] Telegram skipped: {info}")
        except Exception as e:
            print(f"[WARN] Telegram send failed: {e}")

    def safe_log_event(self, status, fps, extra):
        try:
            self.logger.log_event(status=status, fps=fps, source="live_cam", extra_text=extra)
        except Exception as e:
            print(f"[WARN] Logger failed: {e}")

    def save_alert_frame(self, frame, status):
        if not SAVE_ALERT_FRAMES:
            return None

        now = time.time()
        if now - self.last_frame_save_at < FRAME_SAVE_COOLDOWN_SECONDS:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{status.replace(' ', '_').lower()}_{timestamp}.jpg"
        path = os.path.join(ALERT_SAVE_DIR, filename)

        try:
            ok = cv2.imwrite(path, frame)
            if ok:
                self.last_frame_save_at = now
                return path
        except Exception as e:
            print(f"[WARN] Failed to save alert frame: {e}")

        return None

    def update_stable_status(self, candidate_status):
        now = time.time()

        if candidate_status == self.last_stable_status:
            self.status_first_seen_at = None
            return self.last_stable_status

        if candidate_status == "SAFE":
            self.status_first_seen_at = None
            return "SAFE"

        if self.status_first_seen_at is None or candidate_status != self.last_status:
            self.status_first_seen_at = now
            self.last_status = candidate_status
            return self.last_stable_status

        if now - self.status_first_seen_at >= STATUS_DEBOUNCE_SECONDS:
            self.status_first_seen_at = None
            self.last_status = candidate_status
            return candidate_status

        self.last_status = candidate_status
        return self.last_stable_status

    def apply_header_overlay(self, frame, stable_status, fps):
        h, w = frame.shape[:2]
        panel_h = 90
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_h), (18, 20, 28), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        color_map = {
            "SAFE": (46, 204, 113),
            "REAL FIRE": (43, 57, 255),
            "REAL SMOKE": (160, 160, 160),
            "FAKE FIRE": (0, 190, 255),
            "FAKE SMOKE": (0, 210, 210),
        }
        s_color = color_map.get(stable_status, (46, 204, 113))

        cv2.putText(frame, "FireVisionNet", (24, 34),
                    cv2.FONT_HERSHEY_DUPLEX, 0.82, (255, 255, 255), 2)

        cv2.rectangle(frame, (24, 46), (220, 78), s_color, -1)
        cv2.putText(frame, stable_status, (34, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

        cv2.putText(frame, f"FPS {fps:.1f}", (245, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 2)

        cv2.putText(frame, "Alert only for REAL FIRE / REAL SMOKE", (360, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.57, (210, 210, 210), 1)

    def draw_label_box(self, frame, box, label, color, thickness=2):
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        label_y1 = max(0, y - 30)
        label_y2 = y
        label_x2 = x + tw + 16

        cv2.rectangle(frame, (x, label_y1), (label_x2, label_y2), color, -1)
        cv2.putText(frame, label, (x + 8, y - 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

    def handle_notifications(self, stable_status, fps, frame=None):
        now = time.time()

        if stable_status in ("REAL FIRE", "REAL SMOKE"):
            if now - self.last_alert_print > 0.6 or self.last_stable_status != stable_status:
                print(f"[ALERT] {stable_status} | FPS={fps:.1f}")
                self.last_alert_print = now
        else:
            if now - self.last_alert_print > 2.0:
                print(f"[INFO] {stable_status} | FPS={fps:.1f}")
                self.last_alert_print = now

        if stable_status != self.last_stable_status:
            if stable_status in ("REAL FIRE", "REAL SMOKE"):
                self.start_alarm()
                msg = status_to_message(stable_status, fps)
                saved_path = self.save_alert_frame(frame, stable_status) if frame is not None else None

                if stable_status == "REAL FIRE":
                    self.safe_send_telegram("REAL_FIRE", msg)
                else:
                    self.safe_send_telegram("REAL_SMOKE", msg)

                extra = msg if saved_path is None else f"{msg} | snapshot={saved_path}"
                self.safe_log_event(stable_status, fps, extra)
            else:
                self.stop_alarm()
                self.telegram.reset_all()

        if stable_status not in ("REAL FIRE", "REAL SMOKE"):
            self.stop_alarm()

        self.last_stable_status = stable_status

    def process(self, frame):
        frame = self.enhance_frame(frame)
        out = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        proc = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
        sx = frame.shape[1] / proc.shape[1]
        sy = frame.shape[0] / proc.shape[0]

        motion_small = self.motion_mask(proc)
        fire_small = self.fire_mask(proc)
        smoke_small = self.smoke_mask(proc)
        display_small = self.detect_displays(proc)

        moving_fire_small = cv2.bitwise_and(fire_small, motion_small)
        moving_smoke_small = cv2.bitwise_and(smoke_small, motion_small)

        fire_boxes_small = self.find_regions(moving_fire_small, min_area=14)
        smoke_boxes_small = self.find_regions(moving_smoke_small, min_area=90)

        fire_boxes = [(int(x * sx), int(y * sy), int(w * sx), int(h * sy)) for (x, y, w, h) in fire_boxes_small]
        smoke_boxes = [(int(x * sx), int(y * sy), int(w * sx), int(h * sy)) for (x, y, w, h) in smoke_boxes_small]
        display_boxes = [(int(x * sx), int(y * sy), int(w * sx), int(h * sy)) for (x, y, w, h) in display_small]

        fire_boxes = self.merge_boxes(fire_boxes, iou_thresh=0.25)
        smoke_boxes = self.merge_boxes(smoke_boxes, iou_thresh=0.25)
        display_boxes = self.merge_boxes(display_boxes, iou_thresh=0.25)

        fire_mask_full = self.fire_mask(frame)
        smoke_mask_full = self.smoke_mask(frame)

        real_fire_boxes = []
        real_smoke_boxes = []
        fake_fire_boxes = []
        fake_smoke_boxes = []

        live_real_fire = []
        live_real_smoke = []
        live_fake_fire = []
        live_fake_smoke = []

        for box in fire_boxes:
            metrics = self.fire_metrics(frame, gray, box, fire_mask_full, display_boxes, proc, sx, sy)

            is_display_overlap = any(self.iou(box, d) >= DISPLAY_IOU_THRESHOLD for d in display_boxes)

            if is_display_overlap or self.is_fake_fire_metric(metrics):
                self.fake_fire_tracks.update(box, metrics)
                live_fake_fire.append(box)
                hist = self.fake_fire_tracks.get(box)
                positives = sum(1 for x in hist if self.is_fake_fire_metric(x))
                if positives >= 2:
                    fake_fire_boxes.append(box)
                continue

            self.real_fire_tracks.update(box, metrics)
            live_real_fire.append(box)
            hist = self.real_fire_tracks.get(box)
            positives = sum(1 for x in hist if self.is_real_fire_metric(x))
            if positives >= MIN_FIRE_CONFIRM_FRAMES:
                real_fire_boxes.append(box)

        self.real_fire_tracks.cleanup(live_real_fire)
        self.fake_fire_tracks.cleanup(live_fake_fire)

        for box in smoke_boxes:
            metrics = self.smoke_metrics(frame, gray, box, smoke_mask_full, display_boxes, proc, sx, sy)

            is_display_overlap = any(self.iou(box, d) >= DISPLAY_IOU_THRESHOLD for d in display_boxes)

            if is_display_overlap or self.is_fake_smoke_metric(metrics):
                self.fake_smoke_tracks.update(box, metrics)
                live_fake_smoke.append(box)
                hist = self.fake_smoke_tracks.get(box)
                positives = sum(1 for x in hist if self.is_fake_smoke_metric(x))
                if positives >= 2:
                    fake_smoke_boxes.append(box)
                continue

            self.real_smoke_tracks.update(box, metrics)
            live_real_smoke.append(box)
            hist = self.real_smoke_tracks.get(box)
            positives = sum(1 for x in hist if self.is_real_smoke_metric(x))
            if positives >= MIN_SMOKE_CONFIRM_FRAMES:
                real_smoke_boxes.append(box)

        self.real_smoke_tracks.cleanup(live_real_smoke)
        self.fake_smoke_tracks.cleanup(live_fake_smoke)

        real_fire_boxes = self.merge_boxes(real_fire_boxes, iou_thresh=0.25)
        real_smoke_boxes = self.merge_boxes(real_smoke_boxes, iou_thresh=0.25)
        fake_fire_boxes = self.merge_boxes(fake_fire_boxes, iou_thresh=0.25)
        fake_smoke_boxes = self.merge_boxes(fake_smoke_boxes, iou_thresh=0.25)

        self.real_fire_hist.append(1 if len(real_fire_boxes) > 0 else 0)
        self.real_smoke_hist.append(1 if len(real_smoke_boxes) > 0 else 0)

        fire_on = sum(self.real_fire_hist) >= MIN_FIRE_CONFIRM_FRAMES
        smoke_on = sum(self.real_smoke_hist) >= MIN_SMOKE_CONFIRM_FRAMES

        candidate_status = "SAFE"
        if fire_on:
            candidate_status = "REAL FIRE"
        elif smoke_on:
            candidate_status = "REAL SMOKE"
        elif len(fake_fire_boxes) > 0:
            candidate_status = "FAKE FIRE"
        elif len(fake_smoke_boxes) > 0:
            candidate_status = "FAKE SMOKE"

        stable_status = self.update_stable_status(candidate_status)

        for box in fake_fire_boxes:
            self.draw_label_box(out, box, "FAKE FIRE", (0, 200, 255), thickness=2)

        for box in fake_smoke_boxes:
            self.draw_label_box(out, box, "FAKE SMOKE", (0, 220, 220), thickness=2)

        for box in real_fire_boxes:
            self.draw_label_box(out, box, "REAL FIRE", (0, 0, 255), thickness=3)

        for box in real_smoke_boxes:
            self.draw_label_box(out, box, "REAL SMOKE", (180, 180, 180), thickness=3)

        self.prev_gray = gray.copy()
        return out, stable_status

    def run(self, cam_id=0):
        self.running = True
        self.cam = CameraReader(
            cam_id=cam_id,
            width=FRAME_WIDTH,
            height=FRAME_HEIGHT,
            fps=FPS
        ).start()

        time.sleep(1.0)

        cv2.namedWindow("FireVisionNet", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("FireVisionNet", 1280, 720)

        while self.running:
            try:
                t0 = time.time()

                ok, frame = self.cam.read()
                if not ok or frame is None:
                    blank = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                    cv2.putText(blank, "Waiting for camera / reconnecting...", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
                    cv2.imshow("FireVisionNet", blank)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    time.sleep(0.05)
                    continue

                out, stable_status = self.process(frame)
                fps = 1.0 / max(time.time() - t0, 1e-6)

                display_frame = cv2.resize(out, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
                self.apply_header_overlay(display_frame, stable_status, fps)

                try:
                    self.handle_notifications(stable_status, fps, frame=display_frame)
                except Exception as e:
                    print(f"[WARN] handle_notifications failed: {e}")

                cv2.imshow("FireVisionNet", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            except Exception as e:
                print(f"[WARN] Frame loop error: {e}")
                time.sleep(0.05)
                continue

        self.stop()


if __name__ == "__main__":
    app = FireVisionNet()
    try:
        app.run(CAM_ID)
    except KeyboardInterrupt:
        app.stop()
    except Exception as e:
        print(f"[FATAL] Unhandled exception: {e}")
        app.stop()
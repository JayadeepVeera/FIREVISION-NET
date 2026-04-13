#!/usr/bin/env python3
import time
import threading
import base64
from collections import deque

import cv2
import numpy as np

from configs.settings import (
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

from src.alerts.telegram_alert import TelegramAlert
from src.database.logger import EventLogger
from src.utils.helpers import status_to_message


class CameraReader:
    def __init__(self, cam_id=0, width=1280, height=720, fps=30):
        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.lock = threading.Lock()
        self.frame = None
        self.running = self.cap.isOpened()
        self.thread = None

    def start(self):
        if self.running:
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
        return self

    def _update(self):
        while self.running:
            ok, frame = self.cap.read()
            if ok and frame is not None:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()


class TrackMemory:
    def __init__(self, maxlen=12):
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
        dead = [k for k in self.data.keys() if k not in live_keys]
        for k in dead:
            del self.data[k]


class FireVisionNet:
    def __init__(self):
        self.prev_gray = None

        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=400,
            varThreshold=24,
            detectShadows=False
        )
        

        self.fire_patch_cache = {}
        self.smoke_patch_cache = {}
        self.screen_patch_cache = {}

        self.real_fire_tracks = TrackMemory(maxlen=12)
        self.fake_fire_tracks = TrackMemory(maxlen=12)
        self.real_smoke_tracks = TrackMemory(maxlen=12)
        self.fake_smoke_tracks = TrackMemory(maxlen=12)

        self.real_fire_hist = deque(maxlen=12)
        self.fake_fire_hist = deque(maxlen=12)
        self.real_smoke_hist = deque(maxlen=12)
        self.fake_smoke_hist = deque(maxlen=12)

        self.last_alert_print = 0.0
        self.last_status = "SAFE"
        self.last_logged_status = None

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

    def enhance_frame(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        out = cv2.convertScaleAbs(out, alpha=1.03, beta=0)
        return out
    
    def encode_frame_to_base64(self, frame):
        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            return None
        return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

    def draw_status(self, frame, text, color):
        cv2.rectangle(frame, (12, 12), (360, 72), color, -1)
        cv2.rectangle(frame, (12, 12), (360, 72), (255, 255, 255), 2)
        cv2.putText(frame, text, (24, 50), cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 255, 255), 2)

    def draw_fps(self, frame, fps):
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

    def handle_notifications(self, status, fps):
        now = time.time()

        if status != "SAFE":
            if now - self.last_alert_print > 0.5 or self.last_status != status:
                print(f"[ALERT] {status} | FPS={fps:.1f}")
                self.last_alert_print = now

        if status != self.last_status:
            msg = status_to_message(status, fps)

            if status == "REAL FIRE":
                self.telegram.send_alert_once("REAL_FIRE", msg)
            elif status == "FAKE FIRE":
                self.telegram.send_alert_once("FAKE_FIRE", msg)
            elif status == "REAL SMOKE":
                self.telegram.send_alert_once("REAL_SMOKE", msg)
            elif status == "FAKE SMOKE":
                self.telegram.send_alert_once("FAKE_SMOKE", msg)

            if status != "SAFE":
                self.logger.log_event(status=status, fps=fps, source="live_cam", extra_text=msg)

        self.last_status = status

    def find_regions(self, mask, min_area=50):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))
        return boxes

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

    def overlaps_any(self, box, boxes, thresh=0.15):
        return any(self.iou(box, b) >= thresh for b in boxes)

    def best_overlap(self, box, boxes):
        best_iou = 0.0
        best_box = None
        for b in boxes:
            v = self.iou(box, b)
            if v > best_iou:
                best_iou = v
                best_box = b
        return best_iou, best_box

    def region_ratio(self, mask, box):
        x, y, w, h = box
        roi = mask[y:y+h, x:x+w]
        if roi.size == 0:
            return 0.0
        return float(np.count_nonzero(roi) / ((w * h) + 1e-6))

    def motion_score(self, gray, box):
        if self.prev_gray is None:
            return 0.0
        x, y, w, h = box
        prev_roi = self.prev_gray[y:y+h, x:x+w]
        curr_roi = gray[y:y+h, x:x+w]
        if prev_roi.size == 0 or curr_roi.size == 0 or prev_roi.shape != curr_roi.shape:
            return 0.0
        return float(np.mean(cv2.absdiff(prev_roi, curr_roi)))

    def irregularity_score(self, mask, box):
        x, y, w, h = box
        roi = mask[y:y+h, x:x+w]
        if roi.size == 0:
            return 0.0

        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        if area <= 1 or peri <= 1:
            return 0.0

        circularity = (4.0 * np.pi * area) / (peri * peri + 1e-6)
        return float(1.0 - circularity)

    def upward_motion_score(self, gray, box):
        if self.prev_gray is None:
            return 0.0

        x, y, w, h = box
        if w < 8 or h < 8:
            return 0.0

        prev_roi = self.prev_gray[y:y+h, x:x+w]
        curr_roi = gray[y:y+h, x:x+w]
        if prev_roi.size == 0 or curr_roi.size == 0 or prev_roi.shape != curr_roi.shape:
            return 0.0

        pts = cv2.goodFeaturesToTrack(prev_roi, maxCorners=20, qualityLevel=0.01, minDistance=3)
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
        roi = mask[y:y+h, x:x+w]
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

    def blur_variance(self, gray, box):
        x, y, w, h = box
        roi = gray[y:y+h, x:x+w]
        if roi.size == 0:
            return 9999.0
        return float(cv2.Laplacian(roi, cv2.CV_64F).var())

    def edge_density(self, gray, box):
        x, y, w, h = box
        roi = gray[y:y+h, x:x+w]
        if roi.size == 0:
            return 0.0
        edges = cv2.Canny(roi, 60, 160)
        return float(np.count_nonzero(edges) / (roi.size + 1e-6))

    def border_edge_ratio(self, gray, box, border=4):
        x, y, w, h = box
        roi = gray[y:y+h, x:x+w]
        if roi.size == 0 or w < 12 or h < 12:
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
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return 0.0
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        s_std = np.std(hsv[:, :, 1])
        v_std = np.std(hsv[:, :, 2])
        flat = max(0.0, 1.0 - ((s_std + v_std) / 90.0))
        return float(min(1.0, flat))

    def rectangularity_score(self, box, display_boxes):
        best_iou, _ = self.best_overlap(box, display_boxes)
        return float(best_iou)

    def temporal_screen_score(self, frame_small, box_small):
        x, y, w, h = box_small
        roi = frame_small[y:y+h, x:x+w]
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
        val = float(np.mean(diff))
        return val

    def detect_displays(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 7, 50, 50)
        edges = cv2.Canny(blur, 45, 135)
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
            if rw < 40 or rh < 40:
                continue

            rect_area = max(rw * rh, 1.0)
            rect_ratio = max(rw, rh) / (min(rw, rh) + 1e-6)
            contour_fill = area / rect_area

            x, y, w, h = cv2.boundingRect(cnt)
            if w < 50 or h < 50:
                continue

            bbox_ratio = w / float(h + 1e-6)

            cond_quad = len(approx) == 4 and 0.45 <= bbox_ratio <= 2.5 and contour_fill > 0.55
            cond_rect = 0.45 <= rect_ratio <= 2.7 and contour_fill > 0.68

            if cond_quad or cond_rect:
                boxes.append((x, y, w, h))

        merged = []
        for b in sorted(boxes, key=lambda z: z[2] * z[3], reverse=True):
            keep = True
            for m in merged:
                if self.iou(b, m) > 0.4:
                    keep = False
                    break
            if keep:
                merged.append(b)

        return merged

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
            (r > g * 1.08) &
            (g >= b * 0.82) &
            (((h <= 32) | (h >= 172))) &
            (s > 85) &
            (v > 150) &
            ((r - b) > 35)
        )

        mask = cond.astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
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

        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        soft = cv2.absdiff(gray, blur)
        soft = cv2.threshold(soft, 14, 255, cv2.THRESH_BINARY_INV)[1]

        mask = cv2.bitwise_and(low_sat, soft)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        return mask

    def motion_mask(self, frame):
        fg = self.bg_sub.apply(frame)
        fg = cv2.threshold(fg, 220, 255, cv2.THRESH_BINARY)[1]
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fg = cv2.dilate(fg, np.ones((3, 3), np.uint8), iterations=2)
        return fg

    def fire_metrics(self, frame, gray, box, fire_mask_full, display_boxes, proc_small, sx, sy):
        x, y, w, h = box
        roi_gray = gray[y:y+h, x:x+w]
        bright = float(np.mean(roi_gray)) if roi_gray.size > 0 else 0.0

        small_box = (
            int(x / sx), int(y / sy),
            max(1, int(w / sx)), max(1, int(h / sy))
        )

        screen_temporal = self.temporal_screen_score(proc_small, small_box)

        return {
            "fire_ratio": self.region_ratio(fire_mask_full, box),
            "motion": self.motion_score(gray, box),
            "irregular": self.irregularity_score(fire_mask_full, box),
            "upward": self.upward_motion_score(gray, box),
            "flicker": self.patch_temporal_score(fire_mask_full, box, self.fire_patch_cache, "fire"),
            "bright": bright,
            "area": w * h,
            "edge_density": self.edge_density(gray, box),
            "border_edge_ratio": self.border_edge_ratio(gray, box),
            "flatness": self.color_flatness_score(frame, box),
            "rectangularity": self.rectangularity_score(box, display_boxes),
            "screen_temporal": screen_temporal,
        }

    def smoke_metrics(self, frame, gray, box, smoke_mask_full, display_boxes, proc_small, sx, sy):
        x, y, w, h = box

        small_box = (
            int(x / sx), int(y / sy),
            max(1, int(w / sx)), max(1, int(h / sy))
        )

        screen_temporal = self.temporal_screen_score(proc_small, small_box)

        return {
            "smoke_ratio": self.region_ratio(smoke_mask_full, box),
            "motion": self.motion_score(gray, box),
            "upward": self.upward_motion_score(gray, box),
            "temporal": self.patch_temporal_score(smoke_mask_full, box, self.smoke_patch_cache, "smoke"),
            "blur_low": 1 if self.blur_variance(gray, box) < 140 else 0,
            "area": box[2] * box[3],
            "edge_density": self.edge_density(gray, box),
            "border_edge_ratio": self.border_edge_ratio(gray, box),
            "flatness": self.color_flatness_score(frame, box),
            "rectangularity": self.rectangularity_score(box, display_boxes),
            "screen_temporal": screen_temporal,
        }

    def is_screen_like_fire(self, m):
        return (
            m["rectangularity"] >= 0.16 or
            (m["border_edge_ratio"] >= 0.34 and m["flatness"] >= 0.42) or
            (m["screen_temporal"] >= 5.0 and m["flatness"] >= 0.35)
        )

    def is_screen_like_smoke(self, m):
        return (
            m["rectangularity"] >= 0.16 or
            (m["border_edge_ratio"] >= 0.32 and m["flatness"] >= 0.40) or
            (m["screen_temporal"] >= 4.5 and m["flatness"] >= 0.35)
        )

    def is_real_fire_metric(self, m):
        return (
            m["area"] >= 120 and
            m["fire_ratio"] >= 0.20 and
            m["motion"] >= 2.8 and
            m["irregular"] >= 0.14 and
            m["upward"] >= 0.10 and
            m["flicker"] >= 1.5 and
            m["bright"] >= 124 and
            m["border_edge_ratio"] < 0.28 and
            m["flatness"] < 0.55 and
            m["rectangularity"] < 0.12
        )

    def is_fake_fire_metric(self, m):
        return (
            self.is_screen_like_fire(m) or
            (
                m["fire_ratio"] >= 0.06 and
                (m["flicker"] >= 0.5 or m["motion"] >= 0.8)
            )
        )

    def is_real_smoke_metric(self, m):
        return (
            m["area"] >= 180 and
            m["smoke_ratio"] >= 0.15 and
            0.8 <= m["motion"] <= 8.0 and
            m["upward"] >= 0.04 and
            m["temporal"] >= 0.9 and
            m["blur_low"] == 1 and
            m["border_edge_ratio"] < 0.26 and
            m["flatness"] < 0.58 and
            m["rectangularity"] < 0.12
        )

    def is_fake_smoke_metric(self, m):
        return (
            self.is_screen_like_smoke(m) or
            (
                m["smoke_ratio"] >= 0.05 and
                (m["temporal"] >= 0.5 or m["motion"] >= 0.7)
            )
        )

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

        fire_boxes_small = self.find_regions(moving_fire_small, min_area=18)
        smoke_boxes_small = self.find_regions(moving_smoke_small, min_area=120)

        fire_boxes = [(int(x*sx), int(y*sy), int(w*sx), int(h*sy)) for (x, y, w, h) in fire_boxes_small]
        smoke_boxes = [(int(x*sx), int(y*sy), int(w*sx), int(h*sy)) for (x, y, w, h) in smoke_boxes_small]
        display_boxes = [(int(x*sx), int(y*sy), int(w*sx), int(h*sy)) for (x, y, w, h) in display_small]

        fire_mask_full = self.fire_mask(frame)
        smoke_mask_full = self.smoke_mask(frame)

        real_fire_boxes = []
        fake_fire_boxes = []
        real_smoke_boxes = []
        fake_smoke_boxes = []

        live_real_fire = []
        live_fake_fire = []
        live_real_smoke = []
        live_fake_smoke = []

        for box in fire_boxes:
            metrics = self.fire_metrics(frame, gray, box, fire_mask_full, display_boxes, proc, sx, sy)
            is_display_overlap = self.overlaps_any(box, display_boxes, thresh=0.12)
            screen_like = self.is_screen_like_fire(metrics)

            if is_display_overlap or screen_like:
                self.fake_fire_tracks.update(box, metrics)
                live_fake_fire.append(box)
                hist = self.fake_fire_tracks.get(box)
                positives = sum(1 for x in hist if self.is_fake_fire_metric(x))
                if positives >= 2:
                    fake_fire_boxes.append(box)
            else:
                self.real_fire_tracks.update(box, metrics)
                live_real_fire.append(box)
                hist = self.real_fire_tracks.get(box)
                positives = sum(1 for x in hist if self.is_real_fire_metric(x))
                if positives >= 4:
                    real_fire_boxes.append(box)

        self.real_fire_tracks.cleanup(live_real_fire)
        self.fake_fire_tracks.cleanup(live_fake_fire)

        for box in smoke_boxes:
            metrics = self.smoke_metrics(frame, gray, box, smoke_mask_full, display_boxes, proc, sx, sy)
            is_display_overlap = self.overlaps_any(box, display_boxes, thresh=0.12)
            screen_like = self.is_screen_like_smoke(metrics)

            if is_display_overlap or screen_like:
                self.fake_smoke_tracks.update(box, metrics)
                live_fake_smoke.append(box)
                hist = self.fake_smoke_tracks.get(box)
                positives = sum(1 for x in hist if self.is_fake_smoke_metric(x))
                if positives >= 3:
                    fake_smoke_boxes.append(box)
            else:
                self.real_smoke_tracks.update(box, metrics)
                live_real_smoke.append(box)
                hist = self.real_smoke_tracks.get(box)
                positives = sum(1 for x in hist if self.is_real_smoke_metric(x))
                if positives >= 4:
                    real_smoke_boxes.append(box)

        self.real_smoke_tracks.cleanup(live_real_smoke)
        self.fake_smoke_tracks.cleanup(live_fake_smoke)

        self.real_fire_hist.append(1 if len(real_fire_boxes) > 0 else 0)
        self.fake_fire_hist.append(1 if len(fake_fire_boxes) > 0 else 0)
        self.real_smoke_hist.append(1 if len(real_smoke_boxes) > 0 else 0)
        self.fake_smoke_hist.append(1 if len(fake_smoke_boxes) > 0 else 0)

        fire_on = sum(self.real_fire_hist) >= 4
        fake_fire_on = sum(self.fake_fire_hist) >= 2
        smoke_on = sum(self.real_smoke_hist) >= 4
        fake_smoke_on = sum(self.fake_smoke_hist) >= 3

        status = "SAFE"
        status_color = (0, 140, 0)

        if fire_on:
            status = "REAL FIRE"
            status_color = (0, 0, 220)
        elif fake_fire_on:
            status = "FAKE FIRE"
            status_color = (0, 165, 255)
        elif smoke_on:
            status = "REAL SMOKE"
            status_color = (130, 130, 130)
        elif fake_smoke_on:
            status = "FAKE SMOKE"
            status_color = (0, 200, 200)

        for (x, y, w, h) in display_boxes:
            cv2.rectangle(out, (x, y), (x + w, y + h), (255, 180, 0), 1)
            cv2.putText(out, "DISPLAY", (x, max(22, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1)

        for (x, y, w, h) in real_fire_boxes:
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(out, "REAL FIRE", (x, max(24, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 255), 2)

        for (x, y, w, h) in fake_fire_boxes:
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 165, 255), 2)
            cv2.putText(out, "FAKE FIRE", (x, max(24, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 165, 255), 2)

        for (x, y, w, h) in real_smoke_boxes:
            cv2.rectangle(out, (x, y), (x + w, y + h), (180, 180, 180), 2)
            cv2.putText(out, "REAL SMOKE", (x, max(24, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (180, 180, 180), 2)

        for (x, y, w, h) in fake_smoke_boxes:
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 220, 220), 2)
            cv2.putText(out, "FAKE SMOKE", (x, max(24, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 220), 2)

        self.draw_status(out, status, status_color)
        self.prev_gray = gray.copy()

        return out, status

    def run(self, cam_id=0):
        cam = CameraReader(cam_id=cam_id, width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=FPS).start()
        time.sleep(1.0)

        ok, frame = cam.read()
        if not ok or frame is None:
            print("camera not available")
            cam.release()
            return

        h, w = frame.shape[:2]
        cv2.namedWindow("FireVisionNet", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("FireVisionNet", w, h)

        while True:
            t0 = time.time()

            ok, frame = cam.read()
            if not ok or frame is None:
                continue

            out, status = self.process(frame)

            fps = 1.0 / max(time.time() - t0, 1e-6)
            self.draw_fps(out, fps)
            self.handle_notifications(status, fps)

            cv2.imshow("FireVisionNet", out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    FireVisionNet().run(CAM_ID)
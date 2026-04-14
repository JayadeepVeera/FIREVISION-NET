import base64
import binascii
import logging
import os
import time
import traceback
from functools import lru_cache
from typing import Dict

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from configs.settings import (
    SQLITE_DB_PATH,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    TELEGRAM_ENABLED,
    TELEGRAM_COOLDOWN_SECONDS,
    DB_ENABLED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger("firevision.api")

app = FastAPI(title="FireVisionNet API")

ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://firevision-net-1.onrender.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

last_alert_state = {"status": None, "ts": 0.0}


class FramePayload(BaseModel):
    image: str


def decode_base64_to_frame(image_str: str):
    if not image_str or not isinstance(image_str, str):
        raise ValueError("Empty image payload")

    raw = image_str.strip()
    if "," in raw:
        raw = raw.split(",", 1)[1]

    raw = "".join(raw.split())
    raw += "=" * (-len(raw) % 4)

    try:
        img_bytes = base64.b64decode(raw)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"Base64 decode failed: {str(e)}")

    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise ValueError("Decoded bytes are not a valid image")

    return frame


@lru_cache(maxsize=1)
def get_detector():
    try:
        from src.inference.firevision_net import FireVisionNet
        detector = FireVisionNet()
        logger.info("Detector initialized successfully")
        return detector
    except Exception as e:
        logger.exception("Detector init failed")
        raise RuntimeError(f"Detector init failed: {type(e).__name__}: {str(e)}")


@lru_cache(maxsize=1)
def get_logger():
    try:
        from src.database.logger import EventLogger
        db_dir = os.path.dirname(SQLITE_DB_PATH)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        event_logger = EventLogger(db_path=SQLITE_DB_PATH, enabled=DB_ENABLED)
        logger.info(f"Logger initialized successfully with db_path={SQLITE_DB_PATH}")
        return event_logger
    except Exception as e:
        logger.exception("Logger init failed")
        raise RuntimeError(f"Logger init failed: {type(e).__name__}: {str(e)}")


@lru_cache(maxsize=1)
def get_telegram():
    try:
        from src.alerts.telegram_alert import TelegramAlert
        telegram = TelegramAlert(
            bot_token=TELEGRAM_BOT_TOKEN,
            chat_id=TELEGRAM_CHAT_ID,
            cooldown=TELEGRAM_COOLDOWN_SECONDS,
            enabled=TELEGRAM_ENABLED,
        )
        logger.info("Telegram initialized successfully")
        return telegram
    except Exception as e:
        logger.exception("Telegram init failed")
        raise RuntimeError(f"Telegram init failed: {type(e).__name__}: {str(e)}")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} in {duration:.3f}s")
    return response


@app.get("/")
def root():
    return {
        "ok": True,
        "message": "FireVisionNet API running",
        "python_version": os.sys.version,
    }


@app.get("/health")
def health():
    checks = {
        "detector": {"ok": False, "error": None},
        "logger": {"ok": False, "error": None},
        "telegram": {"ok": False, "error": None},
    }

    try:
        get_detector()
        checks["detector"]["ok"] = True
    except Exception as e:
        checks["detector"]["error"] = str(e)

    try:
        get_logger()
        checks["logger"]["ok"] = True
    except Exception as e:
        checks["logger"]["error"] = str(e)

    try:
        get_telegram()
        checks["telegram"]["ok"] = True
    except Exception as e:
        checks["telegram"]["error"] = str(e)

    return {
        "ok": True,
        "message": "API reachable",
        "checks": checks,
        "cors": {
            "allowed_origins": ALLOWED_ORIGINS,
            "allow_credentials": False,
        },
        "db_path": SQLITE_DB_PATH,
    }


@app.get("/cors-debug")
def cors_debug(request: Request):
    return {
        "ok": True,
        "origin_received": request.headers.get("origin"),
        "allowed_origins": ALLOWED_ORIGINS,
    }


@app.get("/events")
def get_events():
    try:
        event_logger = get_logger()
        events = event_logger.get_recent_events(limit=50)
        return {"events": events}
    except Exception as e:
        logger.exception("Failed to fetch events")
        raise HTTPException(status_code=500, detail=f"Failed to fetch events: {str(e)}")


@app.delete("/events")
def clear_events():
    try:
        event_logger = get_logger()
        event_logger.clear_events()
        return {"ok": True, "message": "All events cleared"}
    except Exception as e:
        logger.exception("Failed to clear events")
        raise HTTPException(status_code=500, detail=f"Failed to clear events: {str(e)}")


@app.post("/test-telegram")
def test_telegram():
    try:
        telegram = get_telegram()
        ok, info = telegram.send_message("✨ FireVisionNet test message is working.")
        return {
            "ok": ok,
            "message": "Telegram test sent successfully." if ok else "Telegram test failed.",
            "info": info,
        }
    except Exception as e:
        logger.exception("Telegram error")
        raise HTTPException(status_code=500, detail=f"Telegram error: {str(e)}")


@app.post("/detect-frame")
def detect_frame(payload: FramePayload) -> Dict:
    try:
        detector = get_detector()
        event_logger = get_logger()
        telegram = get_telegram()

        frame = decode_base64_to_frame(payload.image)
        out, status, score = detector.process(frame)
        processed_image = detector.encode_frame_to_base64(out)

        now = time.time()
        dangerous_statuses = {"REAL FIRE", "FAKE FIRE", "REAL SMOKE", "FAKE SMOKE"}

        alert_sent = False
        event_logged = False

        event_logger.log_event(
            status=status,
            fps=None,
            source="web_camera",
            extra_text=f"Detected: {status}, score={score:.4f}",
        )
        event_logged = True

        if status in dangerous_statuses:
            should_send_alert = (
                last_alert_state["status"] != status
                or (now - last_alert_state["ts"] > TELEGRAM_COOLDOWN_SECONDS)
            )

            if should_send_alert:
                alert_text = (
                    f"🚨 FireVisionNet Alert\n\n"
                    f"Status: {status}\n"
                    f"Score: {score:.4f}\n"
                    f"Source: Live Camera\n"
                    f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                ok, info = telegram.send_message(alert_text)
                if ok:
                    last_alert_state["status"] = status
                    last_alert_state["ts"] = now
                    alert_sent = True

        return {
            "ok": True,
            "status": status,
            "score": score,
            "processed_image": processed_image,
            "event_logged": event_logged,
            "alert_sent": alert_sent,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("detect-frame failed")
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"{type(e).__name__}: {str(e)}",
                "trace": traceback.format_exc(),
            },
        )
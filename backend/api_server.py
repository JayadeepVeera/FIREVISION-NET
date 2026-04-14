import base64
import binascii
import time
import traceback
from functools import lru_cache
from typing import Dict

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from configs.settings import (
    SQLITE_DB_PATH,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
    TELEGRAM_ENABLED,
    TELEGRAM_COOLDOWN_SECONDS,
    DB_ENABLED,
    FRONTEND_ORIGINS,
)
from src.alerts.telegram_alert import TelegramAlert
from src.database.logger import EventLogger
from src.inference.live_cam import FireVisionNet

app = FastAPI(title="FireVisionNet API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS if FRONTEND_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

last_alert_state = {
    "status": None,
    "ts": 0.0,
}


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
    return FireVisionNet()


@lru_cache(maxsize=1)
def get_logger():
    return EventLogger(db_path=SQLITE_DB_PATH, enabled=DB_ENABLED)


@lru_cache(maxsize=1)
def get_telegram():
    return TelegramAlert(
        bot_token=TELEGRAM_BOT_TOKEN,
        chat_id=TELEGRAM_CHAT_ID,
        cooldown=TELEGRAM_COOLDOWN_SECONDS,
        enabled=TELEGRAM_ENABLED,
    )


@app.get("/")
def root():
    return {"ok": True, "message": "FireVisionNet API running"}


@app.get("/health")
def health():
    try:
        get_detector()
        get_logger()
        get_telegram()
        return {"ok": True, "message": "FireVisionNet API healthy"}
    except Exception as e:
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "trace": traceback.format_exc(),
        }


@app.get("/events")
def get_events():
    try:
        logger = get_logger()
        return {"events": logger.get_recent_events(limit=50)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch events: {str(e)}")


@app.delete("/events")
def clear_events():
    try:
        logger = get_logger()
        logger.clear_events()
        return {"ok": True, "message": "All events cleared"}
    except Exception as e:
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
        raise HTTPException(status_code=500, detail=f"Telegram error: {str(e)}")


@app.post("/detect-frame")
def detect_frame(payload: FramePayload) -> Dict:
    try:
        detector = get_detector()
        logger = get_logger()
        telegram = get_telegram()

        frame = decode_base64_to_frame(payload.image)
        out, status = detector.process(frame)
        processed_image = detector.encode_frame_to_base64(out)

        now = time.time()
        dangerous_statuses = {"REAL FIRE", "FAKE FIRE", "REAL SMOKE", "FAKE SMOKE"}

        alert_sent = False
        event_logged = False

        if status in dangerous_statuses:
            logger.log_event(
                status=status,
                fps=None,
                source="web_camera",
                extra_text=f"Detected: {status}",
            )
            event_logged = True

            should_send_alert = (
                last_alert_state["status"] != status
                or (now - last_alert_state["ts"] > TELEGRAM_COOLDOWN_SECONDS)
            )

            if should_send_alert:
                alert_text = (
                    f"🚨 FireVisionNet Alert\n\n"
                    f"Status: {status}\n"
                    f"Source: Live Camera\n"
                    f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                telegram.send_message(alert_text)
                last_alert_state["status"] = status
                last_alert_state["ts"] = now
                alert_sent = True

        return {
            "ok": True,
            "status": status,
            "processed_image": processed_image,
            "event_logged": event_logged,
            "alert_sent": alert_sent,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")
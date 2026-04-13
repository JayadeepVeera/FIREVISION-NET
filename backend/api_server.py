import base64
import time
from typing import Dict

import cv2
import numpy as np
from fastapi import FastAPI
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
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = FireVisionNet()
logger = EventLogger(db_path=SQLITE_DB_PATH, enabled=DB_ENABLED)
telegram = TelegramAlert(
    bot_token=TELEGRAM_BOT_TOKEN,
    chat_id=TELEGRAM_CHAT_ID,
    cooldown=TELEGRAM_COOLDOWN_SECONDS,
    enabled=TELEGRAM_ENABLED,
)

last_alert_state = {
    "status": None,
    "ts": 0.0,
}


class FramePayload(BaseModel):
    image: str


@app.get("/health")
def health():
    return {"ok": True, "message": "FireVisionNet API running"}


@app.get("/events")
def get_events():
    return {"events": logger.get_recent_events(limit=50)}


@app.delete("/events")
def clear_events():
    logger.clear_events()
    return {"ok": True, "message": "All events cleared"}


@app.post("/test-telegram")
def test_telegram():
    ok, info = telegram.send_message("✨ FireVisionNet test message is working.")
    return {
        "ok": ok,
        "message": "Telegram test sent successfully." if ok else "Telegram test failed.",
        "info": info,
    }


@app.post("/detect-frame")
def detect_frame(payload: FramePayload) -> Dict:
    try:
        raw = payload.image.split(",")[-1]
        img_bytes = base64.b64decode(raw)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"ok": False, "error": "Invalid image"}

        out, status = detector.process(frame)
        processed_image = detector.encode_frame_to_base64(out)

        now = time.time()
        dangerous_statuses = {"REAL FIRE", "FAKE FIRE", "REAL SMOKE", "FAKE SMOKE"}

        if status in dangerous_statuses:
            logger.log_event(
                status=status,
                fps=None,
                source="web_camera",
                extra_text=f"Detected: {status}"
            )

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

        return {
            "ok": True,
            "status": status,
            "processed_image": processed_image,
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}
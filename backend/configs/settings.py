import os

# Database
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "firevision.db")
DB_ENABLED = os.getenv("DB_ENABLED", "true").lower() == "true"

# Telegram
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "true").lower() == "true"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8608899985:AAGQdmvo9Uoq2stvEfopX-ebXWlxyDt2MkI")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "5610343610")
TELEGRAM_COOLDOWN_SECONDS = int(os.getenv("TELEGRAM_COOLDOWN_SECONDS", "15"))

# Camera (for live_cam.py)
CAM_ID = int(os.getenv("CAM_ID", "0"))
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))
FPS = int(os.getenv("FPS", "30"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
SHOW_FRAMES = os.getenv("SHOW_FRAMES", "false").lower() == "true"
MODEL_PATH = os.getenv("MODEL_PATH", "models/best.pt")

# CORS
FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

# API Server
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
import os

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "firevision.db")

TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "true").lower() == "true"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8608899985:AAGQdmvo9Uoq2stvEfopX-ebXWlxyDt2MkI")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "5610343610")
TELEGRAM_COOLDOWN_SECONDS = int(os.getenv("TELEGRAM_COOLDOWN_SECONDS", "15"))

DB_ENABLED = os.getenv("DB_ENABLED", "true").lower() == "true"

FRONTEND_ORIGINS = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]
import time
import requests


class TelegramAlert:
    def __init__(self, bot_token: str, chat_id: str, cooldown: float = 20.0, enabled: bool = True):
        self.bot_token = (bot_token or "").strip()
        self.chat_id = str(chat_id or "").strip()
        self.cooldown = cooldown
        self.enabled = enabled
        self.last_sent = {}

    def is_configured(self):
        return bool(self.bot_token) and bool(self.chat_id)

    def send_message(self, text: str):
        if not self.enabled:
            return False, "Telegram disabled"

        if not self.is_configured():
            return False, "Missing TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID"

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

        try:
            response = requests.post(
                url,
                data={"chat_id": self.chat_id, "text": text},
                timeout=10
            )
            return response.status_code == 200, response.text
        except Exception as e:
            return False, str(e)

    def send_alert_once(self, key: str, text: str):
        now = time.time()
        last = self.last_sent.get(key, 0)

        if now - last < self.cooldown:
            return False, "Cooldown active"

        ok, info = self.send_message(text)
        if ok:
            self.last_sent[key] = now
        return ok, info
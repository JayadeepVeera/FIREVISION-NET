import time
import logging
import requests
from typing import Tuple, Dict

logger = logging.getLogger("firevision.telegram")


class TelegramAlert:
    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        cooldown: int = 15,
        enabled: bool = True
    ):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.cooldown = cooldown
        self.enabled = enabled
        self.last_sent_at: Dict[str, float] = {}

    def _validate(self) -> Tuple[bool, str]:
        if not self.enabled:
            return False, "Telegram alerts are disabled"

        if not self.bot_token:
            return False, "Missing TELEGRAM_BOT_TOKEN"

        if not self.chat_id:
            return False, "Missing TELEGRAM_CHAT_ID"

        return True, "OK"

    def send_message(self, message: str) -> Tuple[bool, str]:
        ok, reason = self._validate()
        if not ok:
            logger.warning(f"Telegram validate failed: {reason}")
            return False, reason

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {"chat_id": self.chat_id, "text": message}

        try:
            response = requests.post(url, json=payload, timeout=15)
            data = response.json()

            if response.status_code == 200 and data.get("ok"):
                logger.info("Telegram message sent successfully")
                return True, "Telegram message sent successfully"

            logger.warning(f"Telegram API error: {data}")
            return False, f"Telegram API error: {data}"

        except Exception as e:
            logger.exception(f"Telegram exception: {e}")
            return False, f"Telegram exception: {e}"

    def send_alert_once(self, alert_key: str, message: str) -> Tuple[bool, str]:
        now = time.time()

        last_time = self.last_sent_at.get(alert_key)
        if last_time is not None:
            elapsed = now - last_time
            if elapsed < self.cooldown:
                remaining = self.cooldown - elapsed
                logger.info(f"Cooldown active for {alert_key}: {remaining:.1f}s remaining")
                return False, f"Cooldown active for {alert_key}: {remaining:.1f}s remaining"

        ok, info = self.send_message(message)
        if ok:
            self.last_sent_at[alert_key] = now
        return ok, info

    def reset_alert(self, alert_key: str) -> None:
        if alert_key in self.last_sent_at:
            del self.last_sent_at[alert_key]
            logger.info(f"Reset alert key: {alert_key}")

    def reset_all(self) -> None:
        self.last_sent_at.clear()
        logger.info("Reset all alert cooldowns")
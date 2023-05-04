"""Telegram Bot Class"""

import json
import requests
from config import TelegramConfig

class TelegramBot(TelegramConfig):
    """Telegram Bot Class."""

    def __init__(self) -> None:
        """Bot Init"""
        super().__init__()
        _request = self._request()
        self.safe_json(_request)
        self._admin_id: int
        self._chat_ids = []
        self._client_names = []
        if _request is not None:
            self._is_available = True
            for client in _request["result"]:
                try:
                    self._chat_ids.append(client["message"]["chat"]["id"])
                    self._chat_ids = list(dict.fromkeys(self._chat_ids)) # drop duplicates
                    self._client_names = list(dict.fromkeys(self._client_names))
                    # Set the admin ID
                    if client["message"]["chat"]["username"] == self.admin_username:
                        self._admin_id = client["message"]["chat"]["id"]
                except Exception as e:
                    print(f"Error in Bot for {client}: ", e)
                    continue
        else:
            self._is_available = False

    def _request(self) -> str:
        try:
            answer = requests.get(f"{self.bot_url}{self.bot_token}/getUpdates")
            content = answer.content.decode("utf8")
            data = json.loads(content)
            return data
        except Exception as e:
            print("Error in Bot request: ", e)
    
    def safe_json(self, data) -> None:
        """Save JSON to File."""
        with open(self.json_path, "w") as file:
            json.dump(data, file, indent=4)
    
    @property
    def admin_id(self) -> int:
        """Get Admin ID."""
        return self._admin_id
    
    @property
    def chat_ids(self) -> list:
        """Get Chat IDs."""
        return self._chat_ids
    
    @property
    def client_names(self) -> list:
        """Get Connected Clients."""
        return self._client_names

    def send_message(self, chat_id: int ,message: str) -> None:
        """Send Message to Telegram."""
        if self._is_available:
            params = {"chat_id": chat_id, "text": message}
            url = f"{self.bot_url}{self.bot_token}/sendMessage"
            requests.get(url, params=params)
    
    def notify_admin(self, message: str) -> None:
        """Notify Admin."""
        self.send_message(self.admin_id, message)
    
    def send_photo(self, chat_id: int, photo: str) -> None:
        """Send Photo to Telegram."""
        params = {"chat_id": chat_id}
        url = f"{self.bot_url}{self.bot_token}/sendPhoto"
        files = {"photo": open(photo, "rb")}
        requests.post(url, params=params, files=files)
    
    def broadcast_massage(self, message: str) -> None:
        """Broadcast Message to Telegram."""
        for chat_id in self.chat_ids:
            self.send_message(chat_id, message)
    
    def broadcast_photo(self, photo: str) -> None:
        """Broadcast Photo to Telegram."""
        for chat_id in self.chat_ids:
            self.send_photo(chat_id, photo)
    
    def admin_photo(self, photo: str) -> None:
        """Send Photo to Admin."""
        self.send_photo(self.admin_id, photo)
"""Logger module."""
import logging
from config import LogConfig
from telegram import TelegramBot

class Logger(LogConfig):
    """Logger class."""

    def __init__(self) -> None:
        """Construct of the Logger class."""
        super().__init__()
        logFormatter = logging.Formatter(self.format)
        rootLogger = logging.getLogger()

        fileHandler = logging.FileHandler(self.path)
        fileHandler.setFormatter(logFormatter)
        rootLogger.addHandler(fileHandler)

        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        rootLogger.addHandler(consoleHandler)

        logging.basicConfig(filemode='w')

        self._bot = TelegramBot()
        self._production = False
        self._development = False

        if self.level == "DEBUG":
            rootLogger.setLevel(logging.DEBUG)
        elif self.level == "INFO":
            rootLogger.setLevel(logging.INFO)
        elif self.level == "WARNING":
            rootLogger.setLevel(logging.WARNING)
        elif self.level == "ERROR":
            rootLogger.setLevel(logging.ERROR)

    @property
    def development(self):
        """Return the development."""
        return self._development
    
    @development.setter
    def development(self, value: bool) -> None:
        """Set the development."""
        self._development = value
    
    @property
    def production(self):
        """Return the production."""
        return self._production
    
    @production.setter
    def production(self, value: bool) -> None:
        """Set the production."""
        self._production = value

    def debug(self, message: str, bot=False) -> None:
        """Log a message."""
        logging.debug(message)
        if bot and self._development:
            self._bot.send_message(message)

    def info(self, message: str, bot=False, admin_only=False) -> None:
        """Log a message."""
        logging.info(message)
        if bot:
            try:
                if admin_only or self._development:
                    self._bot.notify_admin(message)
                elif(self._production):
                    admin_msg = f"--- Broadcasting message to:  ---\n {self._bot.client_names}"
                    logging.info(admin_msg)
                    self._bot.broadcast_massage(message)
            except Exception as e:
                logging.error(e)

    def warning(self, message: str, bot=False) -> None:
        """Log a message."""
        logging.warning(message)
        if bot and self._development:
            self._bot.notify_admin(message)

    def error(self, message: str) -> None:
        """Log a message."""
        logging.exception(message)

    def prediction_plot(self, photo: str, admin_only=False) -> None:
        """Log a message."""
        if self._production or self._development:
            try:
                if admin_only or self._development:
                    self._bot.admin_photo(photo)
                else:
                    self._bot.broadcast_photo(photo)
            except Exception as e:
                logging.error(e)

log = Logger()
from loguru import logger

# Set loguru logger
logger.add("logs/main.log", rotation="1 MB", retention="10 days", level="INFO")
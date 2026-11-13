import logging

# Don't call basicConfig - let the application configure logging
logger = logging.getLogger(__name__)

# Set a reasonable default level (doesn't interfere with app config)
if not logger.handlers:
    handler = logging.NullHandler()
    logger.addHandler(handler)

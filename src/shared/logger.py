import logging

from pythonjsonlogger.json import JsonFormatter

# Set up logging
logger = logging.getLogger(name=__name__)

logHandler = logging.StreamHandler()
formatter = JsonFormatter()  # Preferred format by Datadog
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)

logger.setLevel(
    logging.DEBUG,
)  # Ensure all log messages are captured (only errors are captured by default).

"""
Example logger usages:
logger.info("Hello, world!", extra={"input": "value", "output": "value"})
logger.error("An error occurred with the Gemini API.", extra={"error_message": "An error occurred."})
"""

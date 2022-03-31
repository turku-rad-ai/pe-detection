import logging
import os
from typing import Dict

PE_LOG_FILENAME = "pe_log.txt"
PE_LOG_DIR = "generated"
PE_DEBUG_LEVEL = logging.DEBUG


class LoggerSingleton:
    """Singleton class to have only one logger instance

    See: https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python

    Returns:
        descendant of LoggerSingleton: Singleton instance of subclass
    """

    _instances: Dict[type, object] = {}

    def __new__(class_, *args, **kwargs):
        if class_ not in class_._instances:
            class_._instances[class_] = super(LoggerSingleton, class_).__new__(
                class_, *args, **kwargs
            )

        return class_._instances[class_]


class PELogger(LoggerSingleton):
    logger = None

    def get_logger(self):
        if self.logger is None:
            os.makedirs(PE_LOG_DIR, exist_ok=True)
            self.logger = logging.getLogger()

            # create a file handler
            file_handler = logging.FileHandler(
                filename=os.path.join(PE_LOG_DIR, PE_LOG_FILENAME), mode="a"
            )
            file_handler.setLevel(PE_DEBUG_LEVEL)

            console_handler = logging.StreamHandler()
            console_handler.setLevel(PE_DEBUG_LEVEL)

            # create a logging format
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # add the handlers to the logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            self.logger.setLevel(PE_DEBUG_LEVEL)

        return self.logger

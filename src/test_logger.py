import sys
import os
sys.path.append(os.path.dirname(os.path.abspath("what_ifs")))

from core.logger import setup_logger

logger = setup_logger(name="example_logger")

def divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        logger.error("Division by zero error", exc_info=True)
        return None
if __name__ == "__main__":
    result = divide(10, 0)
    if result is None:
        logger.warning("Result is None due to an error in division")
    else:
        logger.info(f"Division result: {result}")

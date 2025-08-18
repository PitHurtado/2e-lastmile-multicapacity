"""Module for utility functions."""
import logging

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# logging.disable(logging.CRITICAL)

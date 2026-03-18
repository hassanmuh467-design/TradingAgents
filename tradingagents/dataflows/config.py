import tradingagents.default_config as default_config
from typing import Dict, Optional

from tradingagents.logging_config import get_logger

logger = get_logger("dataflows.config")

# Use default config but allow it to be overridden
_config: Optional[Dict] = None


def initialize_config():
    """Initialize the configuration with default values."""
    global _config
    if _config is None:
        _config = default_config.DEFAULT_CONFIG.copy()
        logger.info("Config initialized with default values")


def set_config(config: Dict):
    """Update the configuration with custom values."""
    global _config
    if _config is None:
        _config = default_config.DEFAULT_CONFIG.copy()
    _config.update(config)
    logger.info("Config updated with %d key(s)", len(config))


def get_config() -> Dict:
    """Get the current configuration."""
    if _config is None:
        initialize_config()
    return _config.copy()


# Initialize with default config
initialize_config()

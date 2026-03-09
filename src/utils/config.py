import yaml
from loguru import logger

class Config(dict):
    """
    Dict-like object that allows dot notation access to nested keys.
    Example: config.train.lr instead of config['train']['lr']
    """
    def __init__(self, d=None):
        if d:
            for k, v in d.items():
                if isinstance(v, dict):
                    v = Config(v)
                self[k] = v

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

def load_config(path: str) -> Config:
    """Load config from a YAML file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return Config(data)
    except Exception as e:
        logger.error(f"Failed to load config from {path}: {e}")
        raise

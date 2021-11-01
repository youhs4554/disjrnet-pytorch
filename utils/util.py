from typing import Any


class Container(object):
    def __init__(self, config=None, **extras):
        """Dictionary wrapper to access keys as attributes.
        Args:
            config (dict or Config): Configurations
            extras (kwargs): Extra configurations
        Examples:
            >>> cfg = Config({'lr': 0.01}, momentum=0.95)
            or
            >>> cfg = Config({'lr': 0.01, 'momentum': 0.95})
            then, use as follows:
            >>> print(cfg.lr, cfg.momentum)
        """
        if config is not None:
            if isinstance(config, dict):
                for k in config:
                    setattr(self, k, config[k])
            elif isinstance(config, self.__class__):
                self.__dict__ = config.__dict__.copy()
            else:
                raise ValueError("Unknown config")

        for k, v in extras.items():
            setattr(self, k, v)

    def get(self, key: str, default: Any) -> Any:
        return getattr(self, key, default)

    def set(self, key: str, value: Any) -> None:
        setattr(self, key, value)

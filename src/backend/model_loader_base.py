from abc import ABC, abstractmethod


class ModelLoader(ABC):
    _BACKENDS = {}

    def __init__(self, model_name, model_path=None, config=None, logger=None):
        if config is None:
            config = {}

        self._model_name = model_name.lower()
        self._model_path = model_path
        self._config = config
        self._logger = logger
        self._model = None

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            cls._BACKENDS[name.lower()] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, backend, model_name, model_path=None, config=None, logger=None):
        backend_key = backend.lower()

        if backend_key not in cls._BACKENDS:
            raise ValueError(f"Model loader backend '{backend}' not found. "
                             f"Available: {list(cls._BACKENDS.keys())}")

        return cls._BACKENDS[backend_key](model_name=model_name, model_path=model_path, config=config, logger=logger)

    @abstractmethod
    def load(self):
        pass

    @property
    def model(self):
        return self._model

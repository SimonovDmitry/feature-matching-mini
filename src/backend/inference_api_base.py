from abc import ABC, abstractmethod


class InferenceAPI(ABC):
    _BACKENDS = {}

    def __init__(self, logger, model_name, model, config=None):
        if config is None:
            config = {}

        self._logger = logger
        self._model_name = model_name
        self._model = model
        self._config = config

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            cls._BACKENDS[name.lower()] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, backend, logger, model_name, model, config=None):
        backend_key = backend.lower()

        if backend_key not in cls._BACKENDS:
            raise ValueError(f"Inference backend '{backend}' not found. "
                             f"Available: {list(cls._BACKENDS.keys())}")

        return cls._BACKENDS[backend_key](logger, model_name, model, config)

    @abstractmethod
    def run(self, inputs):
        pass
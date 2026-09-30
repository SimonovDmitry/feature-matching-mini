from abc import ABC, abstractmethod

class IOAdapter(ABC):
    _METHODS = {}

    def __init__(self, model_name, logger=None, config=None):
        if config is None:
            config = {}

        self._model_name = model_name.lower()
        self._config = config
        self._logger = logger
        self._model = None

    @classmethod
    def register(cls, name):
        def decorator(subclass):
            cls._METHODS[name.lower()] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, backend, model_name, logger=None, config=None):
        backend_key = backend.lower()

        if backend_key not in cls._METHODS:
            raise ValueError(f"Model loader backend '{backend}' not found. "
                             f"Available: {list(cls._METHODS.keys())}")

        return cls._METHODS[backend_key](model_name=model_name, config=config, logger=logger)

    @abstractmethod
    def preprocess(self, inputs):
        pass

    @abstractmethod
    def postprocess(self, outputs):
        pass

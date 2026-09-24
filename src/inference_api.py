from abc import ABC, abstractmethod

class InferenceAPI(ABC):
    _METHODS = {}

    def __init__(self, logger, model_name, model, config=None):
        self._logger = logger
        self._model_name = model_name

    def __init_subclass__(cls, register=True, **kwargs):
        super().__init_subclass__(**kwargs)
        if register:
            key = cls.__name__.replace("InferenceAPI", "").lower()
            if key:
                InferenceAPI._METHODS[key] = cls

    @staticmethod
    def create(backend, logger, model_name, model, config=None):
        if config is None:
            config = {}

        backend_key = backend.lower()
        if backend_key not in InferenceAPI._METHODS:
            raise ValueError(f"Inference backend '{backend}' not found. "
                             f"Available: {list(InferenceAPI._METHODS.keys())}")

        return InferenceAPI._METHODS[backend_key](logger, model_name, model, config)

    @abstractmethod
    def run(self, inputs):
        pass
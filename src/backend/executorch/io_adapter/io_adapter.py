from src.backend.io_adapter_base import IOAdapter


@IOAdapter.register("executorch")
class ExecuTorchIOAdapter(IOAdapter):
    def __init__(self, model_name, model_wrapper=None, logger=None, config=None):
        if config is None:
            config = {}

        super().__init__(model_name, model_wrapper, logger, config)

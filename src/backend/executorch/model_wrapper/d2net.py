from src.backend.model_wrapper_base import ModelWrapper
from src.backend.executorch.model_wrapper.model_wrapper import ExecuTorchModelWrapper


@ModelWrapper.register("d2net_executorch")
class D2NetExecuTorchModelWrapper(ExecuTorchModelWrapper):
    def load(self):
        pass

    def call(self, inputs):
        pass
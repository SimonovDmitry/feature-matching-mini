from src.backend.model_wrapper_base import ModelWrapper
from src.backend.executorch.model_wrapper.model_wrapper import ExecuTorchModelWrapper


@ModelWrapper.register("xfeat_executorch")
class XFeatExecuTorchModelWrapper(ExecuTorchModelWrapper):
    def load(self):
        pass

    def call(self, inputs):
        pass

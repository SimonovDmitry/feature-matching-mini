from pathlib import Path

import torch
from executorch.runtime import Runtime, Verification

from src.backend.model_wrapper_base import ModelWrapper
from src.backend.executorch.model_wrapper.model_wrapper import ExecuTorchModelWrapper


@ModelWrapper.register("d2net_executorch")
class D2NetExecuTorchModelWrapper(ExecuTorchModelWrapper):
    def __init__(self, model_name, model_path=None, config=None, logger=None):
        super().__init__(model_name, model_path, config, logger)
        self._method = None

    def load(self):
        model_path = self._model_path or self._config.get("executorch_model_path", self._config.get("model_path"))

        if model_path is None:
            raise ValueError("D2-Net ExecuTorch requires a model path")

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"ExecuTorch model does not exist: {path}")

        program = Runtime.get().load_program(str(path), verification=Verification.Minimal)
        self._method = program.load_method(self._config.get("method_name", "forward"))
        self._model = self._method

        return {'model': self._model}

    def call(self, inputs):
        img = inputs.get("image") if isinstance(inputs, dict) else inputs

        if not isinstance(img, torch.Tensor):
            raise TypeError("D2-Net ExecuTorch expects a torch.Tensor input")

        try:
            with torch.no_grad():
                return self._method.execute((img.contiguous().cpu(),))
        except Exception as e:
            self._logger.error(f"D2-Net inference error: {e}")
            raise

import numpy as np
import torch

from src.inference_api import InferenceAPI


class TorchInferenceAPI(InferenceAPI):
    def __init__(self, logger, model_name, model, config=None):
        if config is None:
            config = {}

        super().__init__(logger, model_name, model, config)

        self._device = torch.device(config.get('device', 'cpu'))
        self._model = model.to(self._device).eval()

    def run(self, inputs):
        tensor_inputs = self._inputs_to_tensor(inputs)

        with torch.no_grad():
            outputs = self._model(**tensor_inputs)

        return outputs

    @staticmethod
    def _inputs_to_tensor(inputs):
        result = {}
        for key, value in inputs.items():
            if isinstance(value, np.ndarray):
                tensor = torch.from_numpy(np.ascontiguousarray(value))
                result[key] = tensor
            else:
                result[key] = value
        return result

from abc import abstractmethod

from src.matchers import Matcher
import src.backend.torch.model_loader
import src.backend.torch.inference_api
import src.backend.torch.io_adapter
from src.backend.io_adapter_base import IOAdapter
from src.backend.inference_api_base import InferenceAPI
from src.backend.model_loader_base import ModelLoader


class DNNMatcher(Matcher):
    def __init__(self, matcher_name, logger, config=None, descriptor_name=None):
        if config is None:
            config = {}

        Matcher.__init__(self, matcher_name, logger, config, descriptor_name)

        backend = config.pop('backend', 'torch').lower()
        loader_name = f"{matcher_name.lower()}_{backend}"

        self._nfeatures = config.get('nfeatures', 4096)
        self._threshold = config.get('threshold', 0.005)

        self._loader = ModelLoader.create(backend=loader_name, model_name=matcher_name, config=config, logger=logger)
        self._model_components = self._loader.load()
        self._model = self._model_components.get('model')
        self._io_adapter = IOAdapter.create(backend=loader_name, model_name=matcher_name, config=config,
                                            logger=logger)
        self._inference = InferenceAPI.create(backend=loader_name, logger=logger, model_name=matcher_name,
                                              model=self._model, config=config)

    def _init_matcher(self):
        pass

    def match(self, features0, features1):
        inputs = {"features0": features0, "features1": features1, "descriptor_name": self._descriptor_name,
                  **self._model_components}
        inputs = self._io_adapter.preprocess(inputs)
        outputs = self._inference.run(inputs)
        outputs = self._io_adapter.postprocess(outputs)

        if outputs['matches'] is not None:
            self._logger.info(f"{self._matcher_name} match {len(outputs['matches'])} points")
        else:
            self._logger.warning(f"{self._matcher_name} match 0 points")
        return outputs

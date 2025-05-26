from abc import ABC, abstractmethod
from pathlib import Path

class RLModelMixin(ABC):
    @abstractmethod
    def load_from_checkpoint(self, path: Path, logger=None):
        """
        Load model weights from a checkpoint.
        The path is assumed to be standard path of the format {env.name}_{algo.name}_{model.name}_{m}x{n}.pt
        """
        pass

    @abstractmethod
    def save_checkpoint(self, path: Path, logger=None):
        """
        Save model weights to checkpoint.
        The path is assumed to be standard path of the format {env.name}_{algo.name}_{model.name}_{m}x{n}.pt
        """
        pass

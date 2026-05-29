from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

from .biotrainer_output_observer import BiotrainerOutputObserver, OutputData

class TensorboardWriter(BiotrainerOutputObserver):

    def __init__(self, log_dir: Path):
        super().__init__()
        self.writer = SummaryWriter(log_dir=str(log_dir))
        self._wrote_config = False

    def update(self, data: OutputData) -> None:
        config = data.current_model_result.config
        if self._wrote_config is False and len(config) > 0:
            self.writer.add_hparams({
                'model': config["model_choice"],
                'num_epochs': config["num_epochs"],
                'use_class_weights': config["use_class_weights"],
                'learning_rate': config["learning_rate"],
                'batch_size': config["batch_size"],
                'embedder_name': config["embedder_name"],
                'seed': config["seed"],
                'loss': config["loss_choice"],
                'optimizer': config["optimizer_choice"],
            }, {})
            self._wrote_config = True
        if data.training_iteration:
            split = data.training_iteration[0]  # TODO Add split to tensorboard
            epoch_metrics = data.training_iteration[1]
            self.writer.add_scalars("Epoch/train", epoch_metrics.training, epoch_metrics.epoch)
            self.writer.add_scalars("Epoch/validation", epoch_metrics.validation, epoch_metrics.epoch)
            self.writer.add_scalars("Epoch/comparison", {
                'training_loss': epoch_metrics.training['loss'],
                'validation_loss': epoch_metrics.validation['loss'],
            }, epoch_metrics.epoch)

    def close(self) -> None:
        self.writer.close()

from tqdm import tqdm

from pytorchito.engines.base import BaseEngine
from pytorchito.utils.importing import instantiate_dict_list_union


class Trainer(BaseEngine):

    def __init__(self, conf):
        super().__init__(conf)
        self.criteria = self._init_criteria()
        self.optimizers = self._init_optimizers()

    def run(self):
        self.logger.info('Training started.')
        for epoch in range(1, self.conf.train.epochs + 1):
            # Train
            log = self._train_epoch(epoch)
            self.logger.info("\n" + ", ".join([f"{k}={v}" for k, v in log.items()]))

            # Validate

            # Save checkpoint
            self._save_checkpoint()

    def _train_epoch(self, epoch):
        total_iters = 0
        total_datapoints = 0
        losses = {name: 0 for name in self.criteria}
        metrics = {name: 0 for name in self.metrics}

        for input, target in tqdm(self.dataloader, desc=f"Epoch {epoch}/{self.conf.train.epochs}"):

            input = input.to(self.device)
            target = target.to(self.device)

            # Optimizer zero grad
            for optimizer in self.optimizers:
                optimizer.zero_grad()

            # Predict
            pred = self.model(input)

            # Loss
            total_loss = 0
            for name in self.criteria:
                loss = self.criteria[name](pred, target)
                total_loss += loss
                losses[name] += loss.item()
            total_loss.backward()

            # Optimizer step
            for optimizer in self.optimizers:
                optimizer.step()

            # Metrics
            for name in self.metrics:
                metrics[name] += self.metrics[name](pred, target).item()

            # Update counters
            total_iters += 1
            total_datapoints += target.numel()

        # Average losses for the epoch
        for name in losses:
            losses[name] /= total_iters

        # Average metrics for the epoch
        for name in metrics:
            metrics[name] /= total_datapoints

        # Return items to log
        return {"losses": losses, "metrics": metrics}

    def _init_criteria(self):
        return instantiate_dict_list_union(self.conf.train.criteria, to_dict=True)

    def _init_optimizers(self):
        return instantiate_dict_list_union(self.conf.train.optimizers,
                                           params=self.model.parameters())

    def _save_checkpoint(self):
        pass

    def _set_mode(self):
        self.conf["_mode"] = "train"

from tqdm import tqdm
import torch

from pytorchito.engines.base import BaseEngine


class ValidatorTester(BaseEngine):

    def __init__(self, conf):
        super().__init__(conf)

    @torch.inference_mode()
    def run(self):
        total_datapoints = 0
        metrics = {name: 0 for name in self.metrics}

        for input, target in tqdm(self.dataloader, desc=self.conf["_mode"].capitalize()):

            input = input.to(self.device)
            target = target.to(self.device)

            # Predict
            pred = self.model(input)

            # Metrics
            for name in self.metrics:
                metrics[name] = self.metrics[name](pred, target)

            # Update counters
            total_iters += 1
            total_datapoints += target.numel()

        # Average metrics for the epoch
        for name in metrics:
            metrics[name] /= total_datapoints

        # Return items to log
        return {"metrics": metrics}


class Validator(ValidatorTester):

    def __init__(self, conf):
        super().__init__(conf)

    def _set_mode(self):
        self.conf["_mode"] = "val"


class Tester(ValidatorTester):

    def __init__(self, conf):
        super().__init__(conf)

    def _set_mode(self):
        self.conf["_mode"] = "test"

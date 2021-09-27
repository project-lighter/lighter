import torch
from pytorchito.engines.base import BaseEngine
from pytorchito.utils.io import instantiate, instantiate_dict_list_union


class Trainer(BaseEngine):
    def __init__(self, conf):
        super().__init__(conf)

        self.dataloader = self._get_dataloader()
        self.optimizer = instantiate(conf.train.optimizer, self.model.parameters())
        self.criteria = instantiate_dict_list_union(conf.train.criteria)
        if conf.train.metrics:
            self.metrics = instantiate_dict_list_union(conf.train.metrics)

    def run(self):
        self.logger.info('Training started.')
        for epoch in range(self.conf.train.epochs):
            print("Epoch", epoch)
            self._run_epoch()

    def _run_epoch(self):
        running_loss = 0

        for input, target in self.dataloader:
            input = input.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(input)

            loss = sum([criterion(out, target) for criterion in self.criteria])
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
        print("Running", running_loss)

    def _get_mode(self):
        self.conf._mode = "train"

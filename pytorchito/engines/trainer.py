from pytorchito.engines.base import BaseEngine
from pytorchito.utils.importing import instantiate, instantiate_dict_list_union


class Trainer(BaseEngine):

    def __init__(self, conf):
        super().__init__(conf)
        self.dataloader = self._init_dataloader()
        self.optimizer = instantiate(conf.train.optimizer, self.model.parameters())
        self.criteria = instantiate_dict_list_union(conf.train.criteria, to_dict=True)
        if conf.train.metrics:
            self.metrics = instantiate_dict_list_union(conf.train.metrics, to_dict=True)

    def run(self):
        self.logger.info('Training started.')
        for epoch in range(self.conf.train.epochs):
            print("Epoch", epoch)
            self._run_epoch()

    def _run_epoch(self):
        running_loss = {name: 0 for name in self.criteria.keys()}

        for input, target in self.dataloader:
            input = input.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            out = self.model(input)

            loss = {}
            for name, criterion in self.criteria.items():
                loss[name] = criterion(out, target)
                running_loss[name] += loss[name].item()

            sum(loss.values()).backward()
            self.optimizer.step()

        for name in running_loss:
            running_loss[name] /= len(self.dataloader)
        print(running_loss)

    def _set_mode(self):
        self.conf["_mode"] = "train"

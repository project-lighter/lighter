import lightly


class NTXentLoss(lightly.loss.ntx_ent_loss.NTXentLoss):
    def __init__(self,
                 temperature: float = 0.5,
                 memory_bank_size: int = 0,
                 gather_distributed: bool = False):
        super().__init__()

    def forward(self, pred, label=None):
        out0, out1 = pred
        return super().forward(out0, out1)
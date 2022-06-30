from torch import nn
from ..losses import *


class BaseModel2D(nn.Module):

    def __init__(self):
        super().__init__()

    def _init_weights(self, **kwargs):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def inference(self, x, **kwargs):
        logits = self(x)
        preds = torch.argmax(logits['seg'], dim=1, keepdim=True).to(torch.float)
        return preds

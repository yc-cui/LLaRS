import sys
import os
from contextvars import ContextVar

import torch
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import ROUTE_CLS_NUM_CLASSES
from ..loss_collector import get_route_cls_collector

# ContextVar: batch metadata for NaN debug (see get_batch_metadata in loss_collector).
_batch_metadata: ContextVar = ContextVar('batch_metadata', default=None)


class RouteFunc(nn.Module):

    def __init__(self, in_channels, text_dim, img_dim):
        super(RouteFunc, self).__init__()
        reduced_dim = 64
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc_text = nn.Linear(text_dim, reduced_dim)
        self.fc_img = nn.Linear(img_dim, reduced_dim)
        self.fc_x = nn.Linear(in_channels, reduced_dim)
        self.fc = nn.Linear(reduced_dim * 3, reduced_dim)
        self.fc_last = nn.Linear(reduced_dim, reduced_dim)
        self.cls_head = nn.Linear(reduced_dim, ROUTE_CLS_NUM_CLASSES)
        self.act = nn.GELU()

    def forward(self, x, text_embd, img_embd):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        text_embd = self.fc_text(text_embd)
        img_embd = self.fc_img(img_embd)
        x = self.fc_x(x)
        x = self.fc(torch.cat([text_embd, img_embd, x], dim=1))
        x = self.act(x)
        x = self.fc_last(x)
        cls_logits = self.cls_head(x)
        rc = get_route_cls_collector()
        if rc is not None:
            rc.add_logits(cls_logits)
        return x

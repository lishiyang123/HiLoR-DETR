"""
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""

import torch.nn as nn
from ..core import register


__all__ = ['DEIM', ]


@register()
class DEIM(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, \
        backbone: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)

        return x

    def load_state_dict(self, state_dict, strict=True, assign=False):
        """Strictly consume checkpoint tensors before deployment-only pruning.

        UFD's auxiliary projection is part of the training checkpoint but is
        never read by evaluation.  The decoder flag is enabled only by the
        dedicated deployment configuration.  Loading first keeps old
        checkpoints fully strict; pruning afterwards makes parameter counts
        reflect the actual inference graph.
        """
        result = super().load_state_dict(
            state_dict,
            strict=strict,
            assign=assign,
        )
        if getattr(self.decoder, 'ufd_deploy_strip', False):
            self.decoder.ufd_head = None
        return result

    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self

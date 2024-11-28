import torch.nn
from modules.backbones.wavenet import WaveNet
from modules.backbones.wavenext import WaveNeXt
from modules.backbones.lynxnet import LYNXNet
from modules.backbones.mergement import Mergement
from utils import filter_kwargs

BACKBONES = {
    'wavenet': WaveNet,
    'lynxnet': LYNXNet,
    'mergement': Mergement,
    'wavenext': WaveNeXt
}


def build_backbone(
        out_dims: int, num_feats: int,
        backbone_type: str, backbone_args: dict
) -> torch.nn.Module:
    backbone = BACKBONES[backbone_type]
    kwargs = filter_kwargs(backbone_args, backbone)
    return BACKBONES[backbone_type](out_dims, num_feats, **kwargs)

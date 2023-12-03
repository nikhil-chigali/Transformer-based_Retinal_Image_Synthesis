import torch

from .generator import ResViT_Generator
from .discriminator import NLayerDiscriminator
from utils import get_norm_layer


def define_G(
    config,
    input_nc,
    img_size,
    gpu_ids=[],
):
    netG = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert torch.cuda.is_available()

    netG = ResViT_Generator(
        config,
        input_dim=input_nc,
        img_size=img_size,
        output_dim=3,
    )

    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    return netG


def define_D(
    input_nc,
    img_size,
    n_layers_D=3,
    norm="batch",
    use_sigmoid=False,
    gpu_ids=[],
):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert torch.cuda.is_available()
    netD = NLayerDiscriminator(
        input_nc,
        n_layers_D,
        norm_layer=norm_layer,
        use_sigmoid=use_sigmoid,
        gpu_ids=gpu_ids,
    )

    if use_gpu:
        netD.cuda(gpu_ids[0])
    return netD

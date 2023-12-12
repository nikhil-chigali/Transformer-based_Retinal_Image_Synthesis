import torch
import torch.nn as nn
import functools


class NLayerDiscriminator(nn.Module):
    def __init__(
        self,
        input_nc,
        n_layers=3,
        norm_layer=nn.BatchNorm2d,
        use_sigmoid=False,
        gpu_ids=[],
    ):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        ndf = 4
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]

        sequence += [nn.Flatten(), nn.LazyLinear(512), nn.ReLU(True), nn.Linear(512, 1)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        if len(self.gpu_ids) and isinstance(x.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, x, self.gpu_ids)
        else:
            print(self.model(x).size())
            return self.model(x)

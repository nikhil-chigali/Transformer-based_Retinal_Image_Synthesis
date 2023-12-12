import torch.nn as nn

from .encoder import Encoder
from .art_block import ART_block


# Generator Network
class ResViT_Generator(nn.Module):
    def __init__(self, config, input_dim, img_size, output_dim):
        super(ResViT_Generator, self).__init__()
        self.config = config
        self.transformer_encoder = Encoder(config)
        norm_layer = nn.BatchNorm2d
        ngf = 4
        use_bias = False
        mult = 4

        # Layer1-Encoder1
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                input_dim,
                ngf,
                kernel_size=7,
                padding=0,
                bias=use_bias,
            ),
            norm_layer(ngf),
            nn.ReLU(inplace=True),
        ]
        setattr(self, "encoder_1", nn.Sequential(*model))

        # Layer2-Encoder2
        i = 0
        mult = 2**i
        model = [
            nn.Conv2d(
                ngf * mult,
                ngf * mult * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=use_bias,
            ),
            norm_layer(ngf * mult * 2),
            nn.ReLU(True),
        ]
        setattr(self, "encoder_2", nn.Sequential(*model))

        # Layer3-Encoder3
        i = 1
        mult = 2**i
        model = [
            nn.Conv2d(
                # ngf,
                ngf * mult,
                ngf * mult * 2,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=use_bias,
            ),
            norm_layer(ngf * mult * 2),
            nn.ReLU(True),
        ]
        setattr(self, "encoder_3", nn.Sequential(*model))

        # ART Blocks
        self.art_1 = ART_block(
            self.config, input_dim, img_size, transformer=self.transformer_encoder
        )
        self.art_2 = ART_block(self.config, input_dim, img_size, transformer=None)
        # self.art_3 = ART_block(self.config, input_dim, img_size, transformer=None)
        # self.art_4 = ART_block(self.config, input_dim, img_size, transformer=None)
        # self.art_5 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_6 = ART_block(
            self.config, input_dim, img_size, transformer=self.transformer_encoder
        )
        # self.art_7 = ART_block(self.config, input_dim, img_size, transformer=None)
        # self.art_8 = ART_block(self.config, input_dim, img_size, transformer=None)
        self.art_9 = ART_block(self.config, input_dim, img_size, transformer=None)

        # Layer13-Decoder1
        n_downsampling = 2
        i = 0
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [
            nn.ConvTranspose2d(
                ngf * mult,
                int(ngf * mult / 2),
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=use_bias,
            ),
            norm_layer(int(ngf * mult / 2)),
            nn.ReLU(True),
        ]
        setattr(self, "decoder_1", nn.Sequential(*model))

        # Layer14-Decoder2
        i = 1
        mult = 2 ** (n_downsampling - i)
        model = []
        model = [
            nn.ConvTranspose2d(
                ngf * mult,
                int(ngf * mult / 2),
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=use_bias,
            ),
            norm_layer(int(ngf * mult / 2)),
            nn.ReLU(True),
        ]
        setattr(self, "decoder_2", nn.Sequential(*model))

        # Layer15-Decoder3
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                # int(ngf * mult / 2),
                ngf,
                output_dim,
                kernel_size=7,
                padding=0,
            ),
            nn.Tanh(),
        ]
        setattr(self, "decoder_3", nn.Sequential(*model))

    def forward(self, x):
        # Pass input through cnn encoder of ResViT
        x = self.encoder_1(x)
        x = self.encoder_2(x)
        x = self.encoder_3(x)

        # Information Bottleneck
        x = self.art_1(x)
        x = self.art_2(x)
        # x = self.art_3(x)
        # x = self.art_4(x)
        # x = self.art_5(x)
        x = self.art_6(x)
        # x = self.art_7(x)
        # x = self.art_8(x)
        x = self.art_9(x)

        # decoder
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        return x

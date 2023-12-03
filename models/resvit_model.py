import torch
import torch.nn as nn
from torch.autograd import Variable

from .networks import define_G, define_D
from utils import get_scheduler, generator_loss, discriminator_loss, get_sample_noise


class ResVitModel(nn.Module):
    def __init__(self, model_config, data_config):
        super(ResVitModel, self).__init__()
        self.model_config = model_config
        self.data_config = data_config

        ## Initialize networks
        self.netG = define_G(
            config=model_config, input_nc=3, img_size=data_config.img_size, gpu_ids=[0]
        )
        self.netD = define_D(
            input_nc=3,
            img_size=data_config.img_size,
            n_layers_D=model_config.nlayers_D,
            use_sigmoid=model_config.use_sigmoid,
            gpu_ids=[0],
        )

        self.dtype = (
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        )
        print("---------- Networks initialized -------------")

    def forward(self, imgs):
        self.noiseA = get_sample_noise(imgs.shape)
        self.fakeA = self.netG(self.noiseA).type(self.dtype)
        self.realA = (2 * (imgs - 0.5)).type(self.dtype)
        return self.fakeA

    def configure_optimizers(self):
        ## Initialize optimizers
        self.schedulers = []
        self.optimizers = []

        self.optimizer_G = torch.optim.Adam(
            self.netG.parameters(),
            lr=self.model_config.training.lr,
            betas=(self.model_config.training.beta1, 0.999),
        )
        self.optimizer_D = torch.optim.Adam(
            self.netD.parameters(),
            lr=self.model_config.training.lr,
            betas=(self.model_config.training.beta1, 0.999),
        )

        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
        for optimizer in self.optimizers:
            self.schedulers.append(get_scheduler(optimizer, self.model_config.training))
        return self.optimizers, self.schedulers

    # [TODO] Learning rate scheduler
    def training_step(self, batch, batch_idx):
        self.forward(batch)

        ## Discriminator
        self.optimizer_D.zero_grad()
        logits_real = self.netD(self.realA).type(self.dtype)

        fake_images = self.fakeA.detach()
        logits_fake = self.netD(fake_images)

        d_total_error = discriminator_loss(logits_real, logits_fake)
        self.manual_backward(d_total_error)
        self.optimizer_D.step()

        ## Generator
        self.optimizer_G.zero_grad()

        gen_logits_fake = self.netD(self.fakeA)
        g_error = generator_loss(gen_logits_fake)
        self.manual_backward(g_error)
        self.optimizer_G.step()
        self.log_dict({"g_loss": g_error, "d_loss": d_total_error}, prog_bar=True)

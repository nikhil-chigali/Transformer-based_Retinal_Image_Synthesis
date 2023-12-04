import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision

from .networks import define_G, define_D
from utils import get_scheduler, generator_loss, discriminator_loss
import pytorch_lightning as pl


class ResVitModel(pl.LightningModule):
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

        self.device_dtype = (
            torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        )
        self.automatic_optimization = False
        print("---------- Networks initialized -------------")

    def forward(self, noise):
        self.noiseA = noise
        self.fakeA = self.netG(self.noiseA).type(self.device_dtype)
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

    def training_step(self, batch, batch_idx):
        noise, imgs = batch
        self.forward(noise)

        self.realA = (2 * (imgs - 0.5)).type(self.device_dtype)

        ## Discriminator
        self.optimizer_D.zero_grad()
        logits_real = self.netD(self.realA).type(self.device_dtype)

        fake_images = self.fakeA.detach()
        logits_fake = self.netD(fake_images)

        d_total_error = discriminator_loss(logits_real, logits_fake)
        self.manual_backward(d_total_error)
        # clip gradients
        self.clip_gradients(
            self.optimizer_D, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
        )
        self.optimizer_D.step()

        ## Generator
        self.optimizer_G.zero_grad()

        gen_logits_fake = self.netD(self.fakeA)
        g_error = generator_loss(gen_logits_fake)
        self.manual_backward(g_error)
        self.clip_gradients(
            self.optimizer_G, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
        )
        self.optimizer_G.step()

        loss_dict = {"train/g_loss": g_error, "train/d_loss": d_total_error}
        self.log_dict(loss_dict, prog_bar=True)
        return loss_dict

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            noise, imgs = batch
            self.forward(noise)

            self.realA = (2 * (imgs - 0.5)).type(self.device_dtype)

            ## Discriminator
            logits_real = self.netD(self.realA).type(self.device_dtype)

            fake_images = self.fakeA.detach()
            logits_fake = self.netD(fake_images)

            d_total_error = discriminator_loss(logits_real, logits_fake)

            ## Generator

            gen_logits_fake = self.netD(self.fakeA)
            g_error = generator_loss(gen_logits_fake)
            loss_dict = {"val/g_loss": g_error, "val/d_loss": d_total_error}

            self.log_dict(loss_dict, prog_bar=True)
            return loss_dict

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            noise, imgs = batch
            self.forward(noise)

            self.realA = (2 * (imgs - 0.5)).type(self.device_dtype)

            ## Discriminator
            logits_real = self.netD(self.realA).type(self.device_dtype)

            fake_images = self.fakeA.detach()
            logits_fake = self.netD(fake_images)

            d_total_error = discriminator_loss(logits_real, logits_fake)

            ## Generator

            gen_logits_fake = self.netD(self.fakeA)
            g_error = generator_loss(gen_logits_fake)
            loss_dict = {"test/g_loss": g_error, "test/d_loss": d_total_error}

            self.log_dict(loss_dict, prog_bar=True, on_epoch=True)
            return loss_dict

        def on_train_epoch_end(self):
            sch1, sch2 = self.lr_schedulers()

            # If the selected scheduler is a ReduceLROnPlateau scheduler.
            if isinstance(sch1, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sch1.step(self.trainer.callback_metrics["train/g_loss"])
            if isinstance(sch2, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sch2.step(self.trainer.callback_metrics["train/d_loss"])

            if (self.current_epoch + 1) % 10 == 0:
                grid = torchvision.utils.make_grid(self.fakeA.detach().cpu())
                self.logger.experiment.add_image(
                    "train/images", grid, self.current_epoch
                )

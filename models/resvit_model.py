import torch
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
        self.lambda_gp = model_config.lambda_gp
        self.automatic_optimization = False
        print("---------- Networks initialized -------------")

    def forward(self, noise):
        noise = noise.type(self.device_dtype)
        fakeA = self.netG(noise).type(self.device_dtype)
        return fakeA

    def configure_optimizers(self):
        ## Initialize optimizers
        self.schedulers = []
        self.optimizers = []

        # self.optimizer_G = torch.optim.Adam(
        #     self.netG.parameters(),
        #     lr=self.model_config.training.lr,
        #     betas=(self.model_config.training.beta1, 0.999),
        # )
        # self.optimizer_D = torch.optim.Adam(
        #     self.netD.parameters(),
        #     lr=self.model_config.training.lr,
        #     betas=(self.model_config.training.beta1, 0.999),
        # )

        self.optimizer_G = torch.optim.RMSprop(
            self.netG.parameters(),
            lr=self.model_config.training.lr,
        )
        self.optimizer_D = torch.optim.RMSprop(
            self.netD.parameters(),
            lr=self.model_config.training.lr,
        )
        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)
        for optimizer in self.optimizers:
            self.schedulers.append(get_scheduler(optimizer, self.model_config.training))
        return self.optimizers, self.schedulers

    def _gradient_penalty(self, real_imgs, generated_imgs, batch_size):
        alpha = torch.rand(batch_size, 1, 1, 1).type(
            self.device_dtype
        )  # Adjust the dimensions of alpha

        # Expand alpha to match the size of real_imgs
        alpha = alpha.expand_as(real_imgs).type(self.device_dtype)

        # Interpolate between real and generated images
        interpolated = alpha * real_imgs + (1 - alpha) * generated_imgs
        # interpolated.requires_grad = True

        # Compute the discriminator's output for interpolated samples
        d_interpolated = self.netD(interpolated)
        grad_outputs = torch.ones_like(d_interpolated, requires_grad=False).type(
            self.device_dtype
        )

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

        # Calculate gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_gp
        return gradient_penalty

    def _generator_step(self, realA, fakeA):
        ## Generator
        self.optimizer_G.zero_grad()

        validity = self.netD(fakeA)
        g_error = -torch.mean(validity)
        # g_error = generator_loss(gen_logits_fake)
        self.manual_backward(g_error)
        # self.clip_gradients(
        #     self.optimizer_G, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
        # )
        self.optimizer_G.step()
        return g_error

    def _discriminator_step(self, realA, fakeA):
        ## Discriminator
        self.optimizer_D.zero_grad()
        real_validity = self.netD(realA).type(self.device_dtype)

        fake_images = fakeA.detach()
        fake_validity = self.netD(fake_images)

        d_total_error = -torch.mean(real_validity) + torch.mean(fake_validity)
        gradient_penalty = self._gradient_penalty(realA, fakeA, realA.size(0))
        d_total_error += gradient_penalty
        # d_total_error = discriminator_loss(logits_real, logits_fake)
        self.manual_backward(d_total_error, retain_graph=True)
        # clip gradients
        # self.clip_gradients(
        #     self.optimizer_D, gradient_clip_val=0.5, gradient_clip_algorithm="norm"
        # )
        self.optimizer_D.step()
        return d_total_error

    def training_step(self, batch, batch_idx):
        noise, imgs = batch
        fakeA = self.forward(noise)

        realA = imgs.type(self.device_dtype)

        d_total_error = self._discriminator_step(realA, fakeA)
        self.log("train/d_loss", d_total_error, prog_bar=True)

        if batch_idx % 3 == 0:
            g_error = self._generator_step(realA, fakeA)
            self.log("train/g_error", g_error, prog_bar=True)

        # loss_dict = {"train/g_loss": g_error, "train/d_loss": d_total_error}
        # self.log_dict(loss_dict, on_step=True, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        noise, imgs = batch
        fakeA = self.forward(noise)

        realA = imgs.type(self.device_dtype)

        ## Discriminator
        real_validity = self.netD(realA).type(self.device_dtype)

        fake_images = fakeA.detach()
        fake_validity = self.netD(fake_images)
        d_total_error = -torch.mean(real_validity) + torch.mean(fake_validity)
        # gradient_penalty = self._gradient_penalty(realA, fakeA, realA.size(0))
        # d_total_error += gradient_penalty

        # d_total_error = discriminator_loss(logits_real, logits_fake)

        ## Generator

        validity = self.netD(fakeA)
        g_error = -torch.mean(validity)
        # gen_logits_fake = self.netD(fakeA)
        # g_error = generator_loss(gen_logits_fake)
        loss_dict = {"val/g_loss": g_error, "val/d_loss": d_total_error}

        self.log_dict(loss_dict, on_step=True, prog_bar=True)
        return loss_dict

    def test_step(self, batch, batch_idx):
        noise, imgs = batch
        fakeA = self.forward(noise)

        realA = imgs.type(self.device_dtype)

        ## Discriminator
        real_validity = self.netD(realA).type(self.device_dtype)

        fake_images = fakeA.detach()
        fake_validity = self.netD(fake_images)
        d_total_error = -torch.mean(real_validity) + torch.mean(fake_validity)
        # gradient_penalty = self._gradient_penalty(realA, fakeA, realA.size(0))
        # d_total_error += gradient_penalty

        # d_total_error = discriminator_loss(logits_real, logits_fake)

        ## Generator

        validity = self.netD(fakeA)
        g_error = -torch.mean(validity)
        # gen_logits_fake = self.netD(fakeA)
        # g_error = generator_loss(gen_logits_fake)
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
            print("EPOCH END HERE:", self.current_epoch)
            if (self.current_epoch + 1) % 1 == 0:
                grid = torchvision.utils.make_grid(fakeA.detach().cpu())
                self.logger.experiment.add_image(
                    "train/images", grid, self.current_epoch
                )

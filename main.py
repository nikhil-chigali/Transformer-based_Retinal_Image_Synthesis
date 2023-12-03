import torch
from utils.configs import model_configs, data_configs
from models import ResVitModel

model_config = model_configs()
data_config = data_configs()
model = ResVitModel(model_config, data_config)
data = -2 * torch.rand((4, 3, 256, 256)) + 1
print(model.netG(data.cuda()).shape)
print(model.netD(data.cuda()).shape)

from utils import generator_loss, discriminator_loss

x = model.netD(data.cuda())
print(generator_loss(x))
print(discriminator_loss(x, torch.randn_like(x)))


# out = netG(torch.randn((8, 3, 256, 256)).cuda())
# torch.save(netG.state_dict(), f="model.pt")
# netG.load_state_dict(torch.load("model.pt"))
# print("load success")
# def run_a_gan(D, G, D_solver, G_solver, discriminator_loss, generator_loss, loader_train, show_every=250,
#               batch_size=128, noise_size=96, num_epochs=10):
#     """
#     Train a GAN!

#     Inputs:
#     - D, G: PyTorch models for the discriminator and generator
#     - D_solver, G_solver: torch.optim Optimizers to use for training the
#       discriminator and generator.
#     - discriminator_loss, generator_loss: Functions to use for computing the generator and
#       discriminator loss, respectively.
#     - show_every: Show samples after every show_every iterations.
#     - batch_size: Batch size to use for training.
#     - noise_size: Dimension of the noise to use as input to the generator.
#     - num_epochs: Number of epochs over the training dataset to use for training.
#     """
#     images = []
#     iter_count = 0
#     for epoch in range(num_epochs):
#         for x, _ in loader_train:
#             if len(x) != batch_size:
#                 continue
#             D_solver.zero_grad()
#             real_data = x.type(dtype)
#             logits_real = D(2* (real_data - 0.5)).type(dtype)

#             g_fake_seed = sample_noise(batch_size, noise_size).type(dtype)
#             fake_images = G(g_fake_seed).detach()
#             logits_fake = D(fake_images.view(batch_size, 1, 28, 28))

#             d_total_error = discriminator_loss(logits_real, logits_fake)
#             d_total_error.backward()
#             D_solver.step()

#             G_solver.zero_grad()
#             g_fake_seed = sample_noise(batch_size, noise_size).type(dtype)
#             fake_images = G(g_fake_seed)

#             gen_logits_fake = D(fake_images.view(batch_size, 1, 28, 28))
#             g_error = generator_loss(gen_logits_fake)
#             g_error.backward()
#             G_solver.step()

#             if (iter_count % show_every == 0):
#                 print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_total_error.item(),g_error.item()))
#                 imgs_numpy = fake_images.data.cpu().numpy()
#                 images.append(imgs_numpy[0:16])

#             iter_count += 1

#     return images

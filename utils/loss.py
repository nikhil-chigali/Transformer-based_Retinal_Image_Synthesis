import torch


def bce_loss(input, target):
    """
    Numerically stable version of the binary cross-entropy loss function.

    As per https://github.com/pytorch/pytorch/issues/751
    See the TensorFlow docs for a derivation of this formula:
    https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

    Inputs:
    - input: PyTorch Tensor of shape (N,1) giving scores.
    - target: PyTorch Tensor of shape (N,1) containing 0 and 1 giving targets.

    Returns:
    - A PyTorch Tensor containing the mean BCE loss over the minibatch of input data.
    """
    neg_abs = -input.abs()
    loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
    return loss.mean()


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,1) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,1) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    loss = None
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # For the discriminator (D), the true target (y = 1) corresponds to "real" images.
    # So the scores of real images, the target is always 1 (a vector).
    real_labels = torch.ones_like(logits_real).type(dtype)
    # Compute the BCE for the scores of the real images.
    real_loss = bce_loss(logits_real, real_labels)

    # For D, the false target (y = 0) corresponds to "fake" images.
    # So the scores of fake images, the target is always 0 (a vector).
    fake_labels = torch.zeros_like(logits_fake).type(dtype)
    # Compute the BCE loss for the fake images.
    fake_loss = bce_loss(logits_fake, fake_labels)

    # Sum the losses.
    loss = real_loss + fake_loss

    return loss


def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    loss = None
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # For the generator (G), the true target (y = 1) corresponds to "fake" images
    fake_labels = torch.ones_like(logits_fake).type(dtype)
    # Compute the BCE for the scores of the fake images
    fake_loss = bce_loss(logits_fake, fake_labels)

    # The generator loss is "fake_loss"
    loss = fake_loss

    return loss

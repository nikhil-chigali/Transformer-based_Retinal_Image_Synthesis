import ml_collections


def model_configs():
    config = ml_collections.ConfigDict()
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1024
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.pretrained_path = "pretrained_models\\checkpoints\\imagenet21k\\R50_ViT.npz"
    config.patch_size = 16
    config.patches = ml_collections.ConfigDict({"size": (16, 16)})
    config.patches.grid = (16, 16)
    config.activation = "softmax"

    config.nlayers_D = 3
    config.use_sigmoid = True
    config.init_type = "xavier"

    config.training = ml_collections.ConfigDict()
    config.training.lr = 0.0002
    config.training.beta1 = 0.5
    config.training.lr_policy = "lambda"
    config.training.niter = 100
    config.training.niter_decay = 100
    config.training.epoch_count = 1
    config.training.lr_decay_iters = 50

    return config


def data_configs():
    config = ml_collections.ConfigDict()
    config.img_size = 256
    config.batch_size = 8
    return config

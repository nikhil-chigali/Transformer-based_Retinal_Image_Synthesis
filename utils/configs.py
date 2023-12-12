import ml_collections

EXP_NAME = "unrolledGANexp1"


def path_configs():
    config = ml_collections.ConfigDict()
    config.checkpoint_file = f"resvit_{EXP_NAME}.ckpt"
    config.checkpoint_dir = "checkpoints\\"
    config.data_path = "data\\Images"
    config.csv_path = "data\\image_paths.csv"

    return config


def model_configs():
    config = ml_collections.ConfigDict()
    config.hidden_size = 512
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1024
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.patch_size = 16
    config.patches = ml_collections.ConfigDict({"size": (16, 16)})
    config.patches.grid = (16, 16)
    config.activation = "softmax"

    config.nlayers_D = 3
    config.use_sigmoid = False
    config.init_type = "xavier"
    config.lambda_gp = 20

    config.training = ml_collections.ConfigDict()
    config.training.lr = 0.01
    config.training.beta1 = 0.5
    config.training.lr_policy = "plateau"
    config.training.niter = 100
    config.training.niter_decay = 100
    config.training.epoch_count = 10
    config.training.lr_decay_iters = 50
    config.training.proj_name = "RetinalImageSynthesis"
    config.training.exp_name = EXP_NAME

    return config


def data_configs():
    config = ml_collections.ConfigDict()
    config.img_size = 256
    config.batch_size = 4
    return config

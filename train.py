import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import os
from models import ResVitModel
from utils import data_configs, path_configs, model_configs


def train_model(
    trainloader,
    valloader,
    testloader,
    load_ckpt=False,
    mode="train",
    logger=None,
):
    data_cfg = data_configs()
    path_cfg = path_configs()
    model_cfg = model_configs()
    torch.set_float32_matmul_precision("medium")
    trainer = pl.Trainer(
        default_root_dir=path_cfg.checkpoint_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=1,
        max_epochs=model_cfg.training.epoch_count,
        callbacks=[
            ModelCheckpoint(
                dirpath=path_cfg.checkpoint_dir,
                filename=path_cfg.checkpoint_file,
                save_weights_only=True,
                # mode="min",
                # monitor="val/g_loss_step",
                every_n_epochs=1,
                save_on_train_epoch_end=True,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        logger=logger,
    )
    trainer.logger._log_graph = True

    pretrained_model = os.path.join(path_cfg.checkpoint_dir, path_cfg.checkpoint_file)
    print(f"Pretrained model path: {pretrained_model}")
    # Check if pretrained model already exists
    if load_ckpt:
        if os.path.isfile(pretrained_model):
            print(f"Found saved model checkpoint at {pretrained_model}, loading...")
            model = ResVitModel.load_from_checkpoint(
                pretrained_model,
                model_config=model_cfg,
                data_config=data_cfg,
            )
        else:
            raise FileNotFoundError(f"Model Checkpoint at {pretrained_model} not found")
    if mode == "train":
        model = ResVitModel(model_cfg, data_cfg)
        # print(model)
        trainer.fit(model, trainloader, valloader)
        trainer.save_model()
        # Loading the best model after training
        print("!!!!!!!!!!!!!!", trainer.checkpoint_callback.best_model_path)
        model = ResVitModel.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

    # Testing best model on val and test sets
    val_result = trainer.validate(model, valloader)
    test_result = trainer.test(model, testloader)
    result = {"final_test_loss": test_result, "final_val_loss": val_result}

    return model, result

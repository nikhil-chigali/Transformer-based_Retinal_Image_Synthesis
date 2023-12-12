from pytorch_lightning.loggers import WandbLogger

from utils import model_configs
from utils import get_dataloader, get_dataset
from train import train_model


def main():
    model_cfg = model_configs()
    logger = WandbLogger(
        name=model_cfg.training.exp_name,
        project=model_cfg.training.proj_name,
        log_model="all",
        # offline=True,
        save_dir="wandb_logs/",
    )
    trainset, testset = get_dataset()
    trainloader, valloader = get_dataloader(trainset, train=True)
    testloader = get_dataloader(testset, train=False)

    model, result = train_model(
        trainloader,
        valloader,
        testloader,
        load_ckpt=False,
        mode="train",
        logger=logger,
    )
    print(result)


if __name__ == "__main__":
    main()

# Copyright 2021, Takaaki Saeki

import os
import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse

from utils import get_model, log
from loss import MultiScaleSpectralLoss
from dataset import Dataset
from evaluate import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(config):

    # Get dataset
    train_dataset = Dataset('train.txt', config, device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=0
    )

    # Prepare model and optimizer
    model, optimizer = get_model(config, device, train=True)
    if config["train"]["data_parallel"]:
        model = nn.DataParallel(model)
    criteria = MultiScaleSpectralLoss().to(device)

    # Init logger
    train_tflog_path = os.path.join(config["path"]["tflog_path"], "train")
    val_tflog_path = os.path.join(config["path"]["tflog_path"], "val")
    train_logger = SummaryWriter(train_tflog_path)
    val_logger = SummaryWriter(val_tflog_path)
    os.makedirs(train_tflog_path, exist_ok=True)
    os.makedirs(val_tflog_path, exist_ok=True)
    os.makedirs(config["path"]["ckpt_path"], exist_ok=True)

    # Extract training config
    step_total = config["train"]["step_total"]
    plot_step = config["train"]["plot_step"]
    val_step = config["train"]["val_step"]
    save_step = config["train"]["save_step"]
    step = config["train"]["restore_step"]
    epoch = step // len(train_dataset)

    # Set progress bar
    step_bar = tqdm(total=config["train"]["step_total"], desc="Training", position=0)
    step_bar.n = config["train"]["restore_step"]
    step_bar.update()

    # main training loop
    while True:
        for batch in train_loader:

            torch.autograd.set_detect_anomaly(True)
            optimizer.zero_grad()

            output = model(batch[0], batch[2])
            loss = criteria(output, batch[3])

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip_thresh"])
            optimizer.step()

            message1 = "Step: {}, Epoch: {}, ".format(step, epoch)
            message2 = "Train Loss: {}".format(loss.item())
            step_bar.write(message1 + message2)

            log(train_logger, step, loss=loss)

            # fig
            if step % plot_step == 0:
                log(train_logger,
                    audio=output[0, :],
                    sampling_rate=config["preprocess"]["sampling_rate"],
                    tag="Training/audio_step{}_synthesized".format(step),
                )
                log(train_logger,
                    audio=batch[3][0, :],
                    sampling_rate=config["preprocess"]["sampling_rate"],
                    tag="Training/audio_step{}_groundtruth".format(step),
                )
                log(train_logger,
                    figwav=output[0, :],
                    sampling_rate=config["preprocess"]["sampling_rate"],
                    tag="Training/melspec_step{}_synthesized".format(step),
                )
                log(train_logger,
                    figwav=batch[3][0, :],
                    sampling_rate=config["preprocess"]["sampling_rate"],
                    tag="Training/melspec_step{}_groundtruth".format(step),
                )

            if step % val_step == 0:
                model.eval()
                message = evaluate(config, model, criteria, step, device, val_logger)
                step_bar.write(message)
                model.train()

            if step % save_step == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(
                        config["path"]["ckpt_path"],
                        "{}.pth.tar".format(step),
                    ),
                )

            step += 1
            step_bar.update(1)
            if step == step_total:
                quit()
        epoch += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_train",
        type=int,
        default=None,
        required=False
    )
    parser.add_argument(
        "--sp_dim",
        type=int,
        default=None,
        required=False
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=None,
        required=False
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=False
    )
    parser.add_argument(
        "--preprocessed_path",
        type=str,
        default=None,
        required=False
    )
    parser.add_argument(
        "--tflog_path",
        type=str,
        default=None,
        required=False
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        required=False
    )
    args = parser.parse_args()

    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    for name in ['num_train', 'corpus', 'sp_dim']:
        if getattr(args, name) is not None:
            config['preprocess'][name] = getattr(args, name)
    for path in ['dataset_path', 'preprocessed_path', 'tflog_path', 'ckpt_path']:
        if getattr(args, path) is not None:
            config['path'][path] = getattr(args, path)
    main(config)
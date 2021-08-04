# Copyright 2021, Takaaki Saeki

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import Dataset
from utils import log

def evaluate(config, model, criteria, step, device, val_logger=None):

    valid_dataset = Dataset('val.txt', config, device)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        collate_fn=valid_dataset.collate_fn,
        num_workers=0
    )

    # Evaluation
    loss_sums = 0.
    for batch in valid_loader:

        with torch.no_grad():
            
            output = model(batch[0], batch[2])
            loss = criteria(output, batch[3])
            loss_sums += loss.item() * len(batch)
    
    loss_mean = loss_sums / len(valid_dataset)

    message = "Validation Step {}, Loss: {}".format(
        step, loss_mean
    )

    log(val_logger, step, loss=loss_mean)

    log(val_logger,
        audio=output[0, :],
        sampling_rate=config["preprocess"]["sampling_rate"],
        tag="Validation/audio_step{}_synthesized".format(step),
    )
    log(val_logger,
        audio=batch[3][0, :],
        sampling_rate=config["preprocess"]["sampling_rate"],
        tag="Validation/audio_step{}_groundtruth".format(step),
    )
    log(val_logger,
        figwav=output[0, :],
        sampling_rate=config["preprocess"]["sampling_rate"],
        tag="Validation/melspec_step{}_synthesized".format(step),
    )
    log(val_logger,
        figwav=batch[3][0, :],
        sampling_rate=config["preprocess"]["sampling_rate"],
        tag="Validation/melspec_step{}_groundtruth".format(step),
    )

    return message
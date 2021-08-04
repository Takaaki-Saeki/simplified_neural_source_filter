import os
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
from model import SourceFilter
import numpy as np
import soundfile as sf

def get_model(config, device, train=False):

    model = SourceFilter(config, device).to(device)
    if config["train"]["restore_step"]:
        ckpt_path = os.path.join(
            config["path"]["ckpt_path"],
            "{}.pth.tar".format(config["train"]["restore_step"])
        )
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    if train:
        optim = torch.optim.Adam(model.parameters(), lr=config["train"]["learning_rate"])
        if config["train"]["restore_step"]:
            optim.load_state_dict(ckpt["optimizer"])
        for state in optim.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        model.train()
        return model, optim

    model.eval()
    model.requires_grad_ = False
    return model


def log(logger, step=None, loss=None, figwav=None, audio=None, sampling_rate=16000, tag=""):
    if loss is not None:
        logger.add_scalar("Loss/total_loss", loss, step)

    if audio is not None:
        audio = audio.detach().cpu().numpy()
        logger.add_audio(
            tag,
            audio / max(abs(audio)),
            sample_rate=sampling_rate,
        )
    
    if figwav is not None:
        figwav = figwav.detach().cpu().numpy()
        fig = plot_melspec(figwav / max(abs(figwav)))
        logger.add_figure(tag, fig)


def plot_melspec(wav, sampling_rate=16000, frame_length=400, fft_length=512, frame_shift=80):
    melspec = librosa.feature.melspectrogram(
        wav, 
        sr=sampling_rate,
        hop_length=frame_shift,
        win_length=frame_length,
        n_mels=128,
        fmax=sampling_rate//2
    )
    fig, ax = plt.subplots()
    melspec_db = librosa.power_to_db(melspec, ref=np.max)
    img = librosa.display.specshow(
        melspec_db,
        x_axis='time',
        y_axis='linear',
        sr=sampling_rate,
        hop_length=frame_shift,
        fmax=sampling_rate//2,
        ax=ax)
    ax.set_title("Melspectrogram", fontsize="medium")
    return fig



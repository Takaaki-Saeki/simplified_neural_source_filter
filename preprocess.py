import numpy as np
import os
import librosa
import soundfile as sf
import tqdm
import pyworld
import glob
import matplotlib.pyplot as plt
import random
import pickle
import argparse
import yaml

def main(config):
    
    # configs
    datapath = config["path"]["dataset_path"]
    out_dir = config["path"]["preprocessed_path"]
    num_train = config["preprocess"]["num_train"]
    SR = config["preprocess"]["sampling_rate"]
    frame_length = config["preprocess"]["frame_length"] # (sample)
    frame_shift = config["preprocess"]["frame_shift"] # (sample)
    fft_length = config["preprocess"]["fft_length"]  # (sample)
    segment_length = config["preprocess"]["segment_length"] # (second)
    sp_dim = config["preprocess"]["sp_dim"]

    os.makedirs(out_dir, exist_ok=True)

    if config["preprocess"]["corpus"] == "jvs":
        train_filelists = []
        val_filelists = []
        spk_idxs = list(range(1, 101))
        random.shuffle(spk_idxs)
        for n in range(100):
            idx = '0'*(3 - len(str(spk_idxs[n]))) + str(spk_idxs[n])
            if n < num_train:
                train_filelists.extend(glob.glob(os.path.join(datapath, 'jvs{}'.format(idx) ,'*.wav')))        
            else:
                val_filelists.extend(glob.glob(os.path.join(datapath, 'jvs{}'.format(idx) ,'*.wav')))
        filelists = train_filelists + val_filelists
    elif config["preprocess"]["corpus"] == "jsut":
        filelists = glob.glob(os.path.join(datapath, '*.wav'))
        random.seed(0)
        random.shuffle(filelists)
        train_filelists = filelists[:num_train]
        val_filelists = filelists[num_train:]
    else:
        raise NotImplementedError()

    with open(os.path.join(out_dir, "train.txt"), "w", encoding="utf-8") as f:
        for m in train_filelists:
            f.write(m + "\n")
    with open(os.path.join(out_dir, "val.txt"), "w", encoding="utf-8") as f:
        for m in val_filelists:
            f.write(m + "\n")

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'wav'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'f0'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'sp'), exist_ok=True)

    for idx, wavpath in enumerate(tqdm.tqdm(filelists)):

        if config["preprocess"]["corpus"] == "jvs":
            basename = os.path.basename(os.path.dirname(wavpath)) + "-" + os.path.splitext(os.path.basename(wavpath))[0]
        elif config["preprocess"]["corpus"] == "jsut":
            basename = os.path.splitext(os.path.basename(wavpath))[0]

        wav, SR = sf.read(wavpath)
        wav, _ = librosa.effects.trim(wav, top_db=20)
        num_seg = len(wav) // (segment_length*SR)
        f0s = []
        sps = []
        wavsegs = []
        for n in range(num_seg):
            wav_seg = wav[n*(segment_length*SR):(n+1)*(segment_length*SR)]
            _f0, _t = pyworld.dio(wav_seg.astype(np.double), fs=SR, frame_period=frame_shift/SR*1000)
            f0 = pyworld.stonemask(wav_seg.astype(np.double), _f0, _t, SR)
            sp = np.sqrt(pyworld.cheaptrick(wav_seg.astype(np.double), f0, _t, SR))
            melfilter = librosa.filters.mel(sr=SR, n_fft=fft_length, n_mels=sp_dim)
            melsp = np.dot(melfilter, sp.T)
            f0s.append(f0[:SR*segment_length//frame_shift])
            sps.append(melsp[:, :SR*segment_length//frame_shift])
            wavsegs.append(wav_seg)
        
        f0s = np.asarray(f0s).astype(np.float32)
        np.save(
            os.path.join(out_dir, "f0", "f0-{}.npy".format(basename)),
            f0s,
        )
        sps = np.asarray(sps).astype(np.float32)
        np.save(
            os.path.join(out_dir, "sp", "sp-{}.npy".format(basename)),
            sps,
        )
        wavsegs = np.asarray(wavsegs).astype(np.float32)
        np.save(
            os.path.join(out_dir, "wav", "wav-{}.npy".format(basename)),
            wavsegs,
        )

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
    args = parser.parse_args()

    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    for name in ['num_train', 'corpus', 'sp_dim']:
        if getattr(args, name) is not None:
            config['preprocess'][name] = getattr(args, name)
    for path in ['dataset_path', 'preprocessed_path']:
        if getattr(args, path) is not None:
            config['path'][path] = getattr(args, path)
    main(config)
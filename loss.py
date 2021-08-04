import torch
import torch.nn as nn
import torchaudio


class MultiScaleSpectralLoss(nn.Module):
    """
    Reference: DDSP: Differentiable Digital Signal Processing
    https://arxiv.org/abs/2001.04643
    """
    def __init__(self):
        super(MultiScaleSpectralLoss, self).__init__()
        self.alpha = 1.0
        self.fft_sizes = [2048, 512, 256, 128, 64]
        self.spectrograms = []
        for fftsize in self.fft_sizes:
            self.spectrograms.append(
                torchaudio.transforms.Spectrogram(n_fft=fftsize, hop_length=fftsize//4, power=2)
            )
        self.spectrograms = nn.ModuleList(self.spectrograms)
        self.l1loss = nn.L1Loss()
        self.eps = 1e-10

    def forward(self, wav_out, wav_target):
        loss = 0.
        for spectrogram in self.spectrograms:
            S_out = spectrogram(wav_out)
            S_target = spectrogram(wav_target)
            log_S_out = torch.log(S_out+self.eps)
            log_S_target = torch.log(S_target+self.eps)
            loss += (self.l1loss(S_out, S_target) + self.alpha * self.l1loss(log_S_out, log_S_target))
        return loss


if __name__ == '__main__':

    # test
    wav_out = torch.ones(1, 64000)
    wav_target = torch.ones(1, 64000)
    criteria = MultiScaleSpectralLoss()
    loss = criteria(wav_out, wav_target)
    print(loss)

 



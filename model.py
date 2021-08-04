import numpy as np
import torch
import torch.nn as nn
import yaml


class ConvLayers(nn.Module):
    def __init__(self, config, n_layer=10):
        super(ConvLayers, self).__init__()
        self.channel = config["model"]["cnn_out"]
        convs = []
        for n in range(n_layer):
            dil = n % 10
            convs.append(
                nn.Conv1d(self.channel, self.channel, 3, dilation=2**dil, stride=1, padding=2**dil),
            )
        self.conv_layers = nn.Sequential(*convs)

    def forward(self, excitation, feature):
        conv_out = self.conv_layers(excitation)
        output = conv_out + excitation + feature
        return torch.tanh(output)


class TransformBlock(nn.Module):
    def __init__(self, config, n_convlayer=10):
        super(TransformBlock, self).__init__()
        self.channel = config["model"]["cnn_out"]
        self.in_linear = nn.Linear(1, self.channel)
        self.out_linear = nn.Linear(self.channel, 1)
        self.conv_layers = ConvLayers(config, n_convlayer)

    def forward(self, excitation, feature):
        output = excitation.transpose(1, 2)
        output = self.in_linear(output).transpose(1, 2)
        output = self.conv_layers(output, feature)
        output = self.out_linear(output.transpose(1, 2))
        output = excitation + output.transpose(1, 2)
        return output


class SourceFilter(nn.Module):
    def __init__(self, config, device):
        super(SourceFilter, self).__init__()
        self.source = SourceModule(config, device)
        self.filter = FilterModule(config)
    
    def forward(self, f0s, feature):
        excitation = self.source(f0s)
        output = self.filter(excitation, feature)
        return output


class SourceModule(nn.Module):
    def __init__(self, config, device):
        super(SourceModule, self).__init__()
        self.n_harmonic = config["model"]["n_harmonic"]
        self.phi = config["model"]["phi"]
        self.alpha = config["model"]["alpha"]
        self.sigma = config["model"]["sigma"]
        self.SR = config["preprocess"]["sampling_rate"]
        self.frame_shift = config["preprocess"]["frame_shift"]
        self.device = device

        self.phi = torch.rand(self.n_harmonic, requires_grad=False) * -1. * np.pi
        self.phi[0] = 0.

        self.amplitude = nn.Parameter(torch.ones(self.n_harmonic+1), requires_grad=True)
        torch.nn.init.normal_(self.amplitude, 0.0, 1.0)

    def forward(self, f0s):
        f0s = torch.repeat_interleave(f0s, self.frame_shift, dim=1) # upsampling
        output = 0.
        for i in range(self.n_harmonic):
            output += self.amplitude[i] * self._signal(f0s*(i+1), self.phi[i])
        output = torch.tanh(output + self.amplitude[self.n_harmonic])
        return output

    def _signal(self, freq, phi):
        noise = torch.normal(0., self.sigma, size=freq.shape).to(self.device)
        eplus = self.alpha * torch.sin(torch.cumsum(2.*np.pi*freq/self.SR, dim=1) + phi) + noise
        argplus = torch.where(freq > 0, 1, 0).to(self.device)
        argzero = torch.where(freq == 0, 1, 0).to(self.device)
        excitation = eplus*argplus + argzero*self.alpha/(3.*self.sigma)*noise
        return excitation


class FilterModule(nn.Module):
    def __init__(self, config):
        super(FilterModule, self).__init__()
        self.in_dim = 1 + config["preprocess"]["sp_dim"]
        self.rnn_hidden = config["model"]["rnn_hidden"]
        self.cnn_out = config["model"]["cnn_out"]
        self.frame_shift = config["preprocess"]["frame_shift"]
        self.n_convlayer = config["model"]["n_convlayer"]
        self.n_transformblock = config["model"]["n_transformblock"]

        self.bilstm = nn.LSTM(self.in_dim, self.rnn_hidden, num_layers=1, bidirectional=True, batch_first=True)
        self.conv = nn.Conv1d(in_channels=self.rnn_hidden*2, out_channels=self.cnn_out, kernel_size=3, stride=1, padding=1)
        self.transform_blocks = nn.ModuleList([TransformBlock(config, self.n_convlayer) for n in range(self.n_transformblock)])

    def forward(self, excitation, feature):
        feature, _ = self.bilstm(feature.transpose(1, 2))
        feature = self.conv(feature.transpose(1, 2))
        feature = torch.repeat_interleave(feature, self.frame_shift, dim=2) # upsampling
        output = excitation.unsqueeze(1)
        for n in range(self.n_transformblock):
            output = self.transform_blocks[n](output, feature)
        return output.squeeze(1)


if __name__ == '__main__':

    # test for basic SourceFilter
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    f0s = torch.zeros((8, 800)).to(device) # expanded f0s: (B, time)
    feature = torch.zeros((8, 81, 800)).to(device)
    source_filter_model = SourceFilter(config, device).to(device)
    output = source_filter_model(f0s, feature)
    assert output.shape == (8, 64000)

    # test for FilterModule
    excitation = torch.zeros((8, 64000)).to(device)
    feature = torch.zeros((8, 81, 800)).to(device)
    filter_module = FilterModule(config).to(device)
    output = filter_module(excitation, feature)
    assert output.shape == (8, 64000)
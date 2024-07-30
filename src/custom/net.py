import torch
import timm
import torch.nn as nn
import torchaudio as ta
from speechbrain.lobes import features as sbfeatures

from .utils import min_max_norm, Mixup, Compose, OneOf, MaskFrequency, MaskTime


class SpectrogramCNN(nn.Module):
    def __init__(self, cfg, init_backbone=True):
        """
        Pytorch network class containing the transformation from waveform to
        mel spectrogram, as well as the forward pass through a CNN backbone.

        Data augmentation like mixup or masked frequency or time can also be
        applied here.

        Parameters
        ----------
        cfg: SimpleNameSpace containing all configurations
        init_backbone: bool (Default=True). Whether to download and initialize the backbone.
                       Not always necessary when debugging.
        """
        super(SpectrogramCNN, self).__init__()

        self.cfg = cfg
        self.n_classes = cfg.n_classes

        if cfg.use_leaf:
            self.wav2timefreq = sbfeatures.Leaf(
                in_channels=1,
                out_channels=cfg.n_mels,
                sample_rate=cfg.sample_rate,
                window_len=cfg.window_size * 1000 / cfg.sample_rate, # in ms
                window_stride=cfg.hop_length * 1000 / cfg.sample_rate, # in ms
                min_freq=cfg.fmin,
                max_freq=cfg.fmax,
                skip_transpose=True  # this gives output in [batch, channel, time] which should match the TA.mel output
            )
        else:
            # Initializes the transformation from waveform to mel spectrogram
            self.mel_spec = ta.transforms.MelSpectrogram(
                sample_rate=cfg.sample_rate,
                n_fft=cfg.n_fft,
                win_length=cfg.window_size,
                hop_length=cfg.hop_length,
                f_min=cfg.fmin,
                f_max=cfg.fmax,
                n_mels=cfg.n_mels,
                power=cfg.power,
                normalized=cfg.mel_normalized,
            )

            self.amplitude_to_db = ta.transforms.AmplitudeToDB(top_db=cfg.top_db)
            self.wav2timefreq = torch.nn.Sequential(self.mel_spec,
                                               self.amplitude_to_db)

        if init_backbone:
            # Initialize pre-trained CNN
            # Input and output layers are automatically adjusted
            self.backbone = timm.create_model(
                cfg.backbone,
                pretrained=cfg.pretrained,
                num_classes=cfg.n_classes,
                in_chans=cfg.in_chans,
            )

        # Spectrogram augmentation
        # Chooses one of MaskFrequency and MaskTime with a probability of cfg.specaug_prob
        # These functions mask a certain segment (frequency or time) in each sample of the batch.
        self.specaug = Compose(
            [OneOf(
                [MaskFrequency(p=1),
                 MaskTime(p=1)],
                p=cfg.specaug_prob),
            ])

        # Mixup augmentation
        # Mixes two random samples in the batch with a random mixing ratio
        # Not only changes the spectrogram, but also turn the 1Hot label vector into probabilities
        self.mixup = Mixup(cfg.mixup_prob)

    def forward(self, x, y=None):
        # (bs, channel, time)
        # Add channel dimension for CNN input
        x = x[:, None, :]

        # (bs, channel, mel, time)
        x = self.wav2timefreq(x)
        if self.cfg.use_leaf:
            x = x[:, None, :, :]

        if self.cfg.minmax_norm:
            x = min_max_norm(x, min_val=self.cfg.min, max_val=self.cfg.max)

        if self.training:
            # Mixup augmentation
            if self.cfg.mixup:
                # Mixup returns adapted spectrogram and label probabilities
                x, y = self.mixup(x, y)
            # Spectrogram augmentation, e.g. MaskFrequency/MaskTime
            if self.cfg.specaug:
                x = self.specaug(x, None)

        # Forward pass through the CNN backbone
        logits = self.backbone(x)

        if self.training:
            # During training has to also return the label in case they got modified during mixup
            return logits, y
        else:
            return logits

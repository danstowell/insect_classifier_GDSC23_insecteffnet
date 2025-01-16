import numpy as np
import pandas as pd
import torch
import librosa
import lightning.pytorch as pl
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import one_hot

from .utils import Compose, OneOf, NoiseInjection, GaussianNoise, PinkNoise, get_max_amplitude_window_index


def collate_fn(batch):
    # Helper function to collate individual samples into batches
    return {
        'wave': torch.stack([x['wave'] for x in batch]),
        'labels': torch.stack([x['labels'] for x in batch])
    }


class AudioDataset(Dataset):
    def __init__(self, df, cfg, mode: str = 'train'):
        """
        Custom pytorch Dataset class, which loads a single sample and applies pre-processing steps.
        Pre-processing includes initialization of the 1Hot label vector,
        additional zero-padding when waveform is too short, as well as data augmentation in form of
        either Noise injections, Gaussian noise or pink noise, respectively.

        Parameters
        ----------
        df: Pandas dataframe, containing the path to the .wav files as well as the label.
        cfg: SimpleNameSpace containing all configurations
        mode: str (Default=train). To differentiate between training and validation/test runs.
              When not 'train', no data augmentation is applied.
        """
        self.df = df
        self.cfg = cfg
        self.filepaths = df["path"].values
        self.labels = torch.zeros((df.shape[0], cfg.n_classes))
        self.weights = None
        self.mode = mode

        # Pre-loads 1Hot label vectors
        self.setup()

        # Data augmentation on waveforms
        # Choose one of NoiseInjection, GaussianNoise, PinkNoise with a probability
        # of cfg.noise_prob.
        self.wave_transforms = Compose(
            [OneOf(
                [NoiseInjection(p=1, max_noise_level=cfg.max_noise),
                 GaussianNoise(p=1, min_snr=cfg.min_snr, max_snr=cfg.max_snr),
                 PinkNoise(p=1, min_snr=cfg.min_snr, max_snr=cfg.max_snr)],
                p=cfg.noise_prob),
            ]
        )

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idxs):
        sample_rate = self.cfg.sample_rate
        # A single audio sample is loaded
        wave, _ = librosa.load(self.filepaths[idxs], sr=sample_rate, mono=True)
        wave = torch.from_numpy(wave)

        start = 0
        # Cross-check whether file is as long as expected, e.g. 5s
        # If not, apply zero-padding
        max_time = int(self.cfg.wav_crop_len * sample_rate)
        if wave.shape[0] <= max_time:
            pad = max_time - wave.shape[0]
            wave = torch.from_numpy(np.pad(wave, (0, pad)))
        else:
            if self.mode == 'test' and self.cfg.max_amp:
                # Allows to get the audio window with the maximum average amplitude
                # Can be helpful during inference and evaluation
                start = get_max_amplitude_window_index('',
                                                       waveform=wave,
                                                       samplerate=sample_rate,
                                                       window_length_sec=self.cfg.wav_crop_len,
                                                       scan_param=50,
                                                       verbose=False)

        # Only necessary due to the max amplitude method
        wave = wave[start:start + max_time]

        if self.mode == 'train':
            # When in training mode, apply data augmentation
            wave = self.wave_transforms(wave, sample_rate)

        sample = {'wave': wave,
                  'labels': self.labels[idxs],
                  }
        return sample

    def setup(self):
        # Sets up 1Hot label vector
        if self.mode == 'train' or self.mode == 'val':
            self.labels = one_hot(torch.tensor(self.df['label'].values),
                                  num_classes=self.cfg.n_classes).float()


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        """
        Lightning DataModule, containing any Pytorch Dataloader classes that are necessary during training,
        e.g. a train and validation dataloader.
        Deals with the corresponding splitting into train and validation and initializes pre-defined
        Pytorch dataloaders.
        The dataloaders deal with any necessary logic such as batching, parallelization across workers etc.

        The functions train_dataloader() and val_dataloader() will be internally called by the Lightning Trainer class
        during training.

        Parameters
        ----------
        cfg: SimpleNameSpace containing all configurations
        """
        super().__init__()
        self.cfg = cfg
        self.df_metadata = pd.read_csv(f'{cfg.data_path}/metadata.csv')

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # Pre-defined validation set can be included in the training run or
            # kept separate to tune hyperparameters
            if self.cfg.include_val:
                self.train = AudioDataset(self.df_metadata, mode='train', cfg=self.cfg)
            else:
                self.train = AudioDataset(self.df_metadata[self.df_metadata['subset'] == self.cfg.datasubsetnames['train']], mode='train',
                                          cfg=self.cfg)

            self.val = AudioDataset(self.df_metadata[self.df_metadata['subset'] == self.cfg.datasubsetnames['val']], mode='val',
                                    cfg=self.cfg)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.cfg.batch_size, shuffle=True, pin_memory=True,
                          num_workers=self.cfg.num_workers, persistent_workers=True, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.cfg.batch_size, shuffle=False, pin_memory=True,
                          num_workers=self.cfg.num_workers, persistent_workers=True, collate_fn=collate_fn)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from pytorch_lightning.callbacks import ProgressBar
from torch.utils.data import DataLoader

from GenreFeatureData import (
    GenreFeatureData,
)


class LSTMClassifierNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=8, num_layers=2):
        super(LSTMClassifierNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        self.linear = nn.Linear(self.hidden_dim, output_dim)
        if torch.cuda.is_available():
            print("\nTraining on GPU")
        else:
            print("\nNo GPU, training on CPU")

    def forward(self, input, hidden=None):
        lstm_out, hidden = self.lstm(input, hidden)
        logits = self.linear(lstm_out[-1])
        genre_scores = F.log_softmax(logits, dim=1)
        return genre_scores, hidden


class MusicGenreClassifer(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.model = LSTMClassifierNet(input_dim=33, hidden_dim=128, output_dim=8, num_layers=2)
        self.hidden = None
        self.loss_function = nn.NLLLoss()

        self.stateful = False

    def forward(self, x, hidden=None):
        prediction, self.hidden = self.model(x, hidden)
        return prediction

    def training_step(self, batch, batch_idx):
        X_local_minibatch, y_local_minibatch = batch
        X_local_minibatch = X_local_minibatch.permute(1, 0, 2)

        y_pred, self.hidden = self.model(X_local_minibatch, self.hidden)
        if not self.stateful:
            self.hidden = None
        else:
            h_0, c_0 = self.hidden
            h_0.detach_(), c_0.detach_()
            self.hidden = (h_0, c_0)

        y_local_minibatch = torch.max(y_local_minibatch, 1)[1]
        loss = self.loss_function(y_pred, y_local_minibatch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X_local_minibatch, y_local_minibatch = batch
        X_local_minibatch = X_local_minibatch.permute(1, 0, 2)

        y_pred, self.hidden = self.model(X_local_minibatch, self.hidden)
        if not self.stateful:
            self.hidden = None

        y_local_minibatch = torch.max(y_local_minibatch, 1)[1]
        val_loss = self.loss_function(y_pred, y_local_minibatch)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        return optimizer


class GTZANDataset(data.Dataset):

    def __init__(self, dataset_partition) -> object:
        self.partition = dataset_partition
        self.genre_features = GenreFeatureData()

        if (
                os.path.isfile(self.genre_features.train_X_preprocessed_data)
                and os.path.isfile(self.genre_features.train_Y_preprocessed_data)
                and os.path.isfile(self.genre_features.dev_X_preprocessed_data)
                and os.path.isfile(self.genre_features.dev_Y_preprocessed_data)
                and os.path.isfile(self.genre_features.test_X_preprocessed_data)
                and os.path.isfile(self.genre_features.test_Y_preprocessed_data)
        ):
            print("Preprocessed files exist, deserializing npy files")
            self.genre_features.load_deserialize_data()
        else:
            print("Preprocessing raw audio files")
            self.genre_features.load_preprocess_data()

        self.train_X = torch.from_numpy(self.genre_features.train_X).type(torch.Tensor)
        self.dev_X = torch.from_numpy(self.genre_features.dev_X).type(torch.Tensor)
        self.test_X = torch.from_numpy(self.genre_features.test_X).type(torch.Tensor)

        self.train_Y = torch.from_numpy(self.genre_features.train_Y).type(torch.LongTensor)
        self.dev_Y = torch.from_numpy(self.genre_features.dev_Y).type(torch.LongTensor)
        self.test_Y = torch.from_numpy(self.genre_features.test_Y).type(torch.LongTensor)

        if self.partition == 'train':
            print("Training X shape: " + str(self.genre_features.train_X.shape))
            print("Training Y shape: " + str(self.genre_features.train_Y.shape))
        elif self.partition == 'dev':
            print("Validation X shape: " + str(self.genre_features.dev_X.shape))
            print("Validation Y shape: " + str(self.genre_features.dev_Y.shape))
        elif self.partition == 'test':
            print("Test X shape: " + str(self.genre_features.test_X.shape))
            print("Test Y shape: " + str(self.genre_features.test_Y.shape))

    def __getitem__(self, index):
        X_training_example_at_index, y_training_example_at_index = None, None
        if self.partition == 'train':
            X_training_example_at_index = self.train_X[index, ]
            y_training_example_at_index = self.train_Y[index, ]

        elif self.partition == 'dev':
            X_training_example_at_index = self.dev_X[index, ]
            y_training_example_at_index = self.dev_Y[index, ]

        elif self.partition == 'test':
            X_training_example_at_index = self.test_X[index, ]
            y_training_example_at_index = self.test_Y[index, ]

        return X_training_example_at_index, y_training_example_at_index

    def __len__(self):
        if self.partition == 'train':
            return len(self.train_Y)
        elif self.partition == 'dev':
            return len(self.dev_Y)
        elif self.partition == 'test':
            return len(self.test_Y)


class MusicGenreDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 35) -> None:
        super().__init__()
        self.dev_dataset = None
        self.test_dataset = None
        self.train_dataset = None
        self.batch_size = batch_size

    def setup(self, stage):
        self.train_dataset = GTZANDataset('train')
        self.dev_dataset = GTZANDataset('dev')
        self.test_dataset = GTZANDataset('test')

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)


if __name__ == "__main__":
    model = MusicGenreClassifer()
    trainer = pl.Trainer(max_epochs=400, log_every_n_steps=4)
    genre_dm = MusicGenreDataModule()

    trainer.fit(model, genre_dm)

    # debug dataloaders
    # for i, batch in enumerate(train_dataloader):
    #     print(i, batch)
    # for i, batch in enumerate(dev_dataloader):
    #     print(i, batch)

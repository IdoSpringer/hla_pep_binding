import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.logging import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from Loader import SignedPairsDataset, DiabetesDataset, MHCPepDataset
from Models import PaddingAutoencoder, AE_Encoder, LSTM_Encoder, CNN_Encoder, ERGO
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from argparse import ArgumentParser

# keep up the good work :)


class ERGOPepMHC(pl.LightningModule):
    def __init__(self, hparams):
        super(ERGOPepMHC, self).__init__()
        self.hparams = hparams
        # Model Type
        self.encoding_model = hparams.encoding_model
        # Dimensions
        self.embedding_dim = hparams.embedding_dim
        self.encoding_dim = hparams.encoding_dim
        self.dropout = hparams.dropout
        self.lr = hparams.lr
        if self.encoding_model == 'CNN':
            # HLA Encoder
            self.hla_encoder = CNN_Encoder(self.embedding_dim, self.encoding_dim, vocab_size=19)
            # Peptide Encoder
            self.pep_encoder = CNN_Encoder(self.embedding_dim, self.encoding_dim)
        elif self.encoding_model == 'LSTM':
            self.lstm_dim = self.encoding_dim
            # HLA Encoder
            self.hla_encoder = LSTM_Encoder(self.embedding_dim, self.lstm_dim, self.dropout, vocab_size=19)
            # Peptide Encoder
            self.pep_encoder = LSTM_Encoder(self.embedding_dim, self.lstm_dim, self.dropout)
        # MLP
        self.mlp_dim = 2 * self.encoding_dim
        self.hidden_layer1 = nn.Linear(self.mlp_dim, int(np.sqrt(self.mlp_dim)))
        self.relu = torch.nn.LeakyReLU()
        self.output_layer = nn.Linear(int(np.sqrt(self.mlp_dim)), 1)
        self.dropout = nn.Dropout(p=self.dropout)

    def forward(self, hla_batch, pep_batch):
        # PEPTIDE Encoder:
        pep_encoding = self.pep_encoder(*pep_batch)
        # HLA Encoder:
        hla_encoding = self.hla_encoder(*hla_batch)
        # MLP Classifier
        tcr_pep_concat = torch.cat([hla_encoding, pep_encoding], 1)
        hidden_output = self.dropout(self.relu(self.hidden_layer1(tcr_pep_concat)))
        mlp_output = self.output_layer(hidden_output)
        output = torch.sigmoid(mlp_output)
        return output

    def step(self, batch):
        if self.encoding_model == 'CNN':
            peps, hla, sign, weight = batch
            # notice that now also peps could be missing
            len_p = torch.sum(peps, dim=1)
            len_h = torch.sum(hla, dim=1)
            full = (len_p * len_h).nonzero(as_tuple=True)
            hla_batch_ful = (hla[full],)
            pep_batch_ful = (peps[full],)
            batch_ful = (hla_batch_ful, pep_batch_ful)
            device = hla.device
            y_hat = torch.zeros(len(sign[full])).to(device)
            if len(full[0]):
                y_hat = self.forward(*batch_ful).squeeze()
            y = sign[full].squeeze()
            weight = weight[full]
        elif self.encoding_model == 'LSTM':
            peps, pep_lens, hla, hla_lens, sign, weight = batch
            full = hla_lens.nonzero(as_tuple=True)
            hla_batch_ful = (hla[full], hla_lens[full])
            pep_batch_ful = (peps[full], pep_lens[full])
            batch_ful = (hla_batch_ful, pep_batch_ful)
            device = hla_lens.device
            y_hat = torch.zeros(len(sign[full])).to(device)
            if len(full[0]):
                y_hat = self.forward(*batch_ful).squeeze()
            y = sign[full].squeeze()
            weight = weight[full]
        return y, y_hat, weight

    def training_step(self, batch, batch_idx):
        # REQUIRED
        self.train()
        y, y_hat, weight = self.step(batch)
        loss = F.binary_cross_entropy(y_hat, y, weight=weight)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        self.eval()
        y, y_hat, _ = self.step(batch)
        return {'val_loss': F.binary_cross_entropy(y_hat, y), 'y_hat': y_hat, 'y': y}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        for x in outputs:
            if x['y'].dim() == 0:
                print(x['y'])
        y = torch.cat([x['y'] for x in outputs])
        y_hat = torch.cat([x['y_hat'] for x in outputs])
        # auc = roc_auc_score(y.cpu(), y_hat.cpu())
        auc = roc_auc_score(y.detach().cpu().numpy(), y_hat.detach().cpu().numpy())
        print(auc)
        tensorboard_logs = {'val_loss': avg_loss, 'val_auc': auc}
        return {'avg_val_loss': avg_loss, 'val_auc': auc, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        pass

    def test_end(self, outputs):
        pass

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        trainfile = 'mhc_peptides_train.csv'
        train_dataset = MHCPepDataset(trainfile)
        if self.encoding_model == 'CNN':
            collate_fn = train_dataset.cnn_collate
        elif self.encoding_model == 'LSTM':
            collate_fn = train_dataset.lstm_collate
        return DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=10, collate_fn=collate_fn)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        testfile = 'mhc_peptides_test.csv'
        test_dataset = MHCPepDataset(testfile)
        if self.encoding_model == 'CNN':
            collate_fn = test_dataset.cnn_collate
        elif self.encoding_model == 'LSTM':
            collate_fn = test_dataset.lstm_collate
        return DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=10, collate_fn=collate_fn)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        pass


def mhc_experiment():
    parser = ArgumentParser()
    parser.add_argument('--encoding_model', type=str, default='CNN')
    parser.add_argument('--embedding_dim', type=int, default=10)
    parser.add_argument('--encoding_dim', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--version', type=str, default='0')
    hparams = parser.parse_args()
    model = ERGOPepMHC(hparams)
    if hparams.encoding_model == 'LSTM':
        name = "double_lstm_model"
    elif hparams.encoding_model == 'CNN':
        name = "double_cnn_model"
    logger = TensorBoardLogger("pep_mhc_logs", name=name, version=hparams.version)
    # logger = TensorBoardLogger("pep_mhc_logs", name="double_cnn_model", version=hparams.version)
    early_stop_callback = EarlyStopping(monitor='val_auc', patience=3, mode='max')
    trainer = Trainer(gpus=[6], logger=logger, early_stop_callback=early_stop_callback)
    trainer.fit(model)


if __name__ == '__main__':
    # ergo_ii_experiment()
    # diabetes_experiment()
    mhc_experiment()
    pass


# NOTE: fix sklearn import problem with this in terminal:
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/dsi/speingi/anaconda3/lib/

# see logs
# tensorboard --logdir dir

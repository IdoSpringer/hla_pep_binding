import torch
from torch.utils.data import Dataset, DataLoader
import collections
import pickle
from Sampler import get_diabetes_peptides
import pandas as pd


class MHCPepDataset(Dataset):
    def __init__(self, datafile):
        self.data = pd.read_csv(datafile, engine='python')
        self.amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
        self.atox = {amino: index for index, amino in enumerate(['PAD'] + self.amino_acids + ['X'])}
        self.hla = [letter for letter in 'HLA-BC*:0123456789']
        self.htox = {amino: index for index, amino in enumerate(['PAD'] + self.hla)}

    def __len__(self):
        return len(self.data)

    def invalid(self, seq, type):
        if type == 'aa':
            return pd.isna(seq) or any([aa not in self.amino_acids for aa in seq])
        elif type == 'hla':
            return pd.isna(seq) or any([l not in self.hla for l in seq])

    def __getitem__(self, index):
        def convert(seq, type):
            if seq == 'UNK':
                seq = [0]
            else:
                if type == 'aa':
                    seq = [self.atox[aa] for aa in seq]
                if type == 'hla':
                    seq = [self.htox[l] for l in seq]
            return seq
        peptide = self.data['Description'][index]
        len_p = len(peptide)
        if self.invalid(peptide, type='aa'):
            peptide = 'UNK'
            len_p = 1
        species = self.data['Name'][index]
        label = self.data['Qualitative Measure'][index]
        mhc = self.data['Allele Name'][index]
        len_m = len(mhc)
        if not (mhc.startswith('HLA-A*') or mhc.startswith('HLA-B*') or mhc.startswith('HLA-C*'))\
                or self.invalid(mhc, type='hla'):
            mhc = 'UNK'
            len_m = 0
        if label in ['Positive', 'Positive-High', 'Positive-Intermediate']:
            sign = 1
        elif label in ['Negative', 'Positive-Low']:
            sign = 0
        else:
            print(label)
        peptide = convert(peptide, type='aa')
        mhc = convert(mhc, type='hla')
        weight = 1
        # should we add negative factor?
        # if sign == 1:
        #     weight = 5
        # else:
        #     weight = 1
        sample = (peptide, len_p, mhc, len_m, float(sign), float(weight))
        return sample

    @staticmethod
    def get_max_length(x):
        return len(max(x, key=len))

    def pad_sequence(self, seq, max_len=None):
        def _pad(_it, _max_len):
            # ignore if too long and max_len is fixed
            if len(_it) > _max_len:
                return [0] * _max_len
            return _it + [0] * (_max_len - len(_it))
        if max_len is None:
            return [_pad(it, self.get_max_length(seq)) for it in seq]
        else:
            return [_pad(it, max_len) for it in seq]

    def one_hot_encoding(self, tcr, max_len=28):
        tcr_batch = list(tcr)
        padding = torch.zeros(len(tcr_batch), max_len, 20 + 1)
        # TCR is converted to numbers at this point
        # We need to match the autoencoder atox, therefore -1
        for i in range(len(tcr_batch)):
            # missing alpha
            if tcr_batch[i] == [0]:
                continue
            tcr_batch[i] = tcr_batch[i] + [self.atox['X']]
            for j in range(min(len(tcr_batch[i]), max_len)):
                padding[i, j, tcr_batch[i][j] - 1] = 1
        return padding

    def ae_collate(self, batch):
        tcra, len_a, tcrb, len_b, peptide, len_p, sign, weight = zip(*batch)
        lst = []
        lst.append(torch.FloatTensor(self.one_hot_encoding(tcra, max_len=34)))
        lst.append(torch.FloatTensor(self.one_hot_encoding(tcrb)))
        lst.append(torch.LongTensor(self.pad_sequence(peptide)))
        lst.append(torch.LongTensor(len_p))
        lst.append(torch.FloatTensor(sign))
        lst.append(torch.FloatTensor(weight))
        return lst

    def cnn_collate(self, batch):
        peptide, len_p, mhc, len_m, sign, weight = zip(*batch)
        lst = []
        lst.append(torch.LongTensor(self.pad_sequence(peptide, max_len=11)))
        lst.append(torch.LongTensor(self.pad_sequence(mhc)))
        lst.append(torch.FloatTensor(sign))
        lst.append(torch.FloatTensor(weight))
        return lst

    def lstm_collate(self, batch):
        transposed = zip(*batch)
        lst = []
        for samples in transposed:
            if isinstance(samples[0], int):
                lst.append(torch.LongTensor(samples))
            elif isinstance(samples[0], float):
                lst.append(torch.FloatTensor(samples))
            elif isinstance(samples[0], collections.Sequence):
                lst.append(torch.LongTensor(self.pad_sequence(samples)))
        return lst
    pass


class SignedPairsDataset(Dataset):
    def __init__(self, samples):
        self.data = samples
        self.amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
        self.atox = {amino: index for index, amino in enumerate(['PAD'] + self.amino_acids + ['X'])}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        def convert(seq):
            if seq == 'UNK':
                seq = [0]
            else:
                seq = [self.atox[aa] for aa in seq]
            return seq
        tcr_data, pep_data, sign = self.data[index]
        tcra, tcrb, v, j = tcr_data
        peptide, mhc, protein = pep_data
        len_a = len(tcra) if tcra != 'UNK' else 0
        len_b = len(tcrb)
        len_p = len(peptide)
        tcra = convert(tcra)
        tcrb = convert(tcrb)
        peptide = convert(peptide)
        if sign == 1:
            weight = 5
        else:
            weight = 1
        sample = (tcra, len_a, tcrb, len_b, peptide, len_p, float(sign), float(weight))
        return sample

    @staticmethod
    def get_max_length(x):
        return len(max(x, key=len))

    def pad_sequence(self, seq):
        def _pad(_it, _max_len):
            return _it + [0] * (_max_len - len(_it))
        return [_pad(it, self.get_max_length(seq)) for it in seq]

    def one_hot_encoding(self, tcr, max_len=28):
        tcr_batch = list(tcr)
        padding = torch.zeros(len(tcr_batch), max_len, 20 + 1)
        # TCR is converted to numbers at this point
        # We need to match the autoencoder atox, therefore -1
        for i in range(len(tcr_batch)):
            # missing alpha
            if tcr_batch[i] == [0]:
                continue
            tcr_batch[i] = tcr_batch[i] + [self.atox['X']]
            for j in range(min(len(tcr_batch[i]), max_len)):
                padding[i, j, tcr_batch[i][j] - 1] = 1
        return padding

    def ae_collate(self, batch):
        tcra, len_a, tcrb, len_b, peptide, len_p, sign, weight = zip(*batch)
        lst = []
        lst.append(torch.FloatTensor(self.one_hot_encoding(tcra, max_len=34)))
        lst.append(torch.FloatTensor(self.one_hot_encoding(tcrb)))
        lst.append(torch.LongTensor(self.pad_sequence(peptide)))
        lst.append(torch.LongTensor(len_p))
        lst.append(torch.FloatTensor(sign))
        lst.append(torch.FloatTensor(weight))
        return lst

    def lstm_collate(self, batch):
        transposed = zip(*batch)
        lst = []
        for samples in transposed:
            if isinstance(samples[0], int):
                lst.append(torch.LongTensor(samples))
            elif isinstance(samples[0], float):
                lst.append(torch.FloatTensor(samples))
            elif isinstance(samples[0], collections.Sequence):
                lst.append(torch.LongTensor(self.pad_sequence(samples)))
        return lst
    pass


class DiabetesDataset(SignedPairsDataset):
    def __init__(self, samples, weight_factor):
        super().__init__(samples)
        self.diabetes_peptides = get_diabetes_peptides('data/McPAS-TCR.csv')
        self.weight_factor = weight_factor

    def __getitem__(self, index):
        sample = list(super().__getitem__(index))
        weight = sample[-1]
        tcr_data, pep_data, sign = self.data[index]
        peptide, mhc, protein = pep_data
        if peptide in self.diabetes_peptides:
            weight *= self.weight_factor
        return tuple(sample[:-1] + [weight])
    pass


class SinglePeptideDataset(SignedPairsDataset):
    def __init__(self, samples, peptide, force_peptide=False):
        super().__init__(samples)
        self.amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
        self.atox = {amino: index for index, amino in enumerate(['PAD'] + self.amino_acids + ['X'])}
        if force_peptide:
            # we do it only for MPS and we have to check that the signs are correct
            pep_data = (peptide, 'mhc', 'protein')
            self.data = [(pair[0], pep_data, pair[-1]) for pair in samples]
        else:
            self.data = [pair for pair in samples if pair[1][0] == peptide]


def check():
    testfile = 'mhc_peptides_test.csv'
    test_dataset = MHCPepDataset(testfile)
    dataloader = DataLoader(test_dataset, batch_size=10, shuffle=True, num_workers=4,
                            collate_fn=test_dataset.cnn_collate)
    for batch in dataloader:
        print(batch)
        # tcr length (including X, 0 for missing)
        # print(torch.sum(batch[0][0]).item())
        peps, hla, sign, weight = batch
        len_p = torch.sum(peps, dim=1)
        len_h = torch.sum(hla, dim=1)
        print(len_p)
        print(len_h)
        missing = (len_p * len_h == 0).nonzero(as_tuple=True)
        full = (len_p * len_h).nonzero(as_tuple=True)
        print(missing)
        # full = len_a.nonzero(as_tuple=True)
        print(full)
        # tcra_batch_ful = (tcra[full],)
        # tcrb_batch_ful = (tcrb[full],)
        # tcrb_batch_mis = (tcrb[missing],)
        # tcr_batch_ful = (tcra_batch_ful, tcrb_batch_ful)
        # tcr_batch_mis = (None, tcrb_batch_mis)
        exit()

# check()
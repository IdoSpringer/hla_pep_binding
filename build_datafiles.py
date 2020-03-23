import pandas as pd
from sklearn.model_selection import train_test_split


def mhc_ligand(filename, outfile):
    data = pd.read_csv(filename, engine='python', header=1, usecols=['Description', 'Name',
                                                                     'Qualitative Measure',
                                                                     'Allele Name'])
    data.to_csv(outfile, index=False)


def data_split(filename, train_file, test_file):
    data = pd.read_csv(filename, engine='python')
    train, test = train_test_split(data, test_size=0.2, random_state=0)
    train.to_csv(train_file, index=False)
    test.to_csv(test_file, index=False)


# mhc_ligand('mhc_ligand_full.csv', 'mhc_peptides.csv')
# data_split('mhc_peptides.csv', 'mhc_peptides_train.csv', 'mhc_peptides_test.csv')


# todo our plan
# we don't need artificial sampler - labels in data     V
# load mhc-peptide dataset                              V
# split to train and test (in files)                    V
# build dataloader                                      V
# build lstm encoder
# build ergo-like model and trainer
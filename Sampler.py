import pandas as pd
import numpy as np
import random
import pickle


def read_data(datafile, file_key):
    amino_acids = [letter for letter in 'ARNDCEQGHILKMFPSTWYV']
    all_pairs = []
    def invalid(seq):
        return pd.isna(seq) or any([aa not in amino_acids for aa in seq])
    if file_key == 'mcpas':
        data = pd.read_csv(datafile, engine='python')
        for index in range(len(data)):
            tcr = data['CDR3.alpha.aa'][index]
            tcrb = data['CDR3.beta.aa'][index]
            v = data['TRBV'][index]
            j = data['TRBJ'][index]
            peptide = data['Epitope.peptide'][index]
            protein = data['Antigen.protein'][index]
            mhc = data['MHC'][index]
            if invalid(tcrb) or invalid(peptide):
                continue
            if invalid(tcra):
                tcra = 'UNK'
            tcr_data = (tcra, tcrb, v, j)
            pep_data = (peptide, mhc, protein)
            all_pairs.append((tcr_data, pep_data))
    elif file_key == 'vdjdb':
        data = pd.read_csv(datafile, engine='python', sep='\t')
        # first read all TRB, then unite with TRA according to sample id
        paired = {}
        for index in range(len(data)):
            id = int(data['complex.id'][index])
            type = data['Gene'][index]
            tcr = data['CDR3'][index]
            if type == 'TRB':
                tcrb = tcr
                v = data['V'][index]
                j = data['J'][index]
                peptide = data['Epitope'][index]
                protein = data['Epitope gene'][index]
                mhc = data['MHC A'][index]
                if invalid(tcrb) or invalid(peptide):
                    continue
                tcr_data = ('UNK', tcrb, v, j)
                pep_data = (peptide, mhc, protein)
                # only TRB
                if id == 0:
                    all_pairs.append((tcr_data, pep_data))
                else:
                    paired[id] = (list(tcr_data), pep_data)
            if type == 'TRA':
                tcra = tcr
                if invalid(tcra):
                    tcra = 'UNK'
                tcr_data, pep_data = paired[id]
                tcr_data[0] = tcra
                paired[id] = (tuple(tcr_data), pep_data)
        all_pairs.extend(list(paired.values()))
    train_pairs, test_pairs = train_test_split(set(all_pairs))
    return all_pairs, train_pairs, test_pairs


def train_test_split(all_pairs):
    '''
    Splitting the TCR-PEP pairs
    '''
    train_pairs = []
    test_pairs = []
    for pair in all_pairs:
        # 80% train, 20% test
        p = np.random.binomial(1, 0.8)
        if p == 1:
            train_pairs.append(pair)
        else:
            test_pairs.append(pair)
    return train_pairs, test_pairs


def positive_examples(pairs):
    examples = []
    for pair in pairs:
        tcr_data, pep_data = pair
        examples.append((tcr_data, pep_data, 1))
    return examples


def is_negative(all_pairs, tcrb, pep):
    for pair in all_pairs:
        tcr_data, pep_data = pair
        if tcr_data[1] == tcrb and pep_data[0] == pep:
            return False
    return True


def negative_examples(pairs, all_pairs, size):
    '''
    Randomly creating intentional negative examples from the same pairs dataset.
    '''
    examples = []
    i = 0
    # Get tcr and peps lists
    tcrs = [tcr_data for (tcr_data, pep_data) in pairs]
    peps = [pep_data for (tcr_data, pep_data) in pairs]
    while i < size:
        # for j in range(5):
        pep_data = random.choice(peps)
        tcr_data = random.choice(tcrs)
        if is_negative(all_pairs, tcr_data[1], pep_data[0]) and \
                (tcr_data, pep_data, 0) not in examples:
                examples.append((tcr_data, pep_data, 0))
                i += 1
    return examples


def get_examples(datafile, file_key):
    all_pairs, train_pairs, test_pairs = read_data(datafile, file_key)
    train_pos = positive_examples(train_pairs)
    test_pos = positive_examples(test_pairs)
    train_neg = negative_examples(train_pairs, all_pairs, 5 * len(train_pos))
    test_neg = negative_examples(test_pairs, all_pairs, 5 * len(test_pos))
    train = train_pos + train_neg
    random.shuffle(train)
    test = test_pos + test_neg
    random.shuffle(test)
    return train, test


def sample_data(datafile, file_key, train_file, test_file):
    train, test = get_examples(datafile, file_key)
    with open(str(train_file) + '.pickle', 'wb') as handle:
        pickle.dump(train, handle)
    with open(str(test_file) + '.pickle', 'wb') as handle:
        pickle.dump(test, handle)


# sample_data('data/McPAS-TCR.csv', 'mcpas', 'mcpas_train_samples', 'mcpas_test_samples')
# sample_data('data/VDJDB_complete.tsv', 'vdjdb', 'vdjdb_train_samples', 'vdjdb_test_samples')

# Notice the different negative sampling - 5 random pairs instead of 5 random TCRs per random peptide


def get_diabetes_peptides(datafile):
    data = pd.read_csv(datafile, engine='python')
    d_peps = set()
    for index in range(len(data)):
        peptide = data['Epitope.peptide'][index]
        if pd.isna(peptide):
            continue
        pathology = data['Pathology'][index]
        if pathology == 'Diabetes Type 1':
            d_peps.add(peptide)
    return d_peps

import logging
import timeit

import joblib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from bayes.LaplacianNB import LaplacianNB


def get_fp(smiles: str) -> list:
    """Function to calculate MorganFingerprint from smiles.
    It returns index of all '1' bits of not-folded fingerprint.

    Args:
        smiles (str): smiles string

    Returns:
        list: return list of index of '1' bits.
    """

    mol = Chem.MolFromSmiles(smiles)

    if not mol:
        return

    fp = AllChem.GetMorganFingerprint(mol, 2)
    if not fp:
        return

    return set(fp.GetNonzeroElements().keys())


def setup_logger():
    """Set up logger to info"""
    # create logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)


setup_logger()


logger = logging.getLogger(__name__)
logger.info('read_csv')

df = pd.read_csv('tests/data/smiles_test.csv', delimiter='\t')
logger.info('calculate fingerprint column')
tic = timeit.default_timer()
df['dicts'] = df['smiles'].apply(
    lambda x: get_fp(x),
)
df.dropna(inplace=True)
X = df['dicts']
Y = df['gene_id']
df.to_pickle('df_processed.pickle')
toc = timeit.default_timer()
result_time = toc - tic
logger.info('Fingerprint calculation took:' + str(result_time))
clf = LaplacianNB()
logger.info('fit data')
tic = timeit.default_timer()
clf.fit(X, Y)
toc = timeit.default_timer()
result_time = toc - tic
logger.info('Fitting took:' + str(result_time))
filename = 'model.sav'

# tic=timeit.default_timer()


joblib.dump(clf, filename)

jll = clf._joint_log_likelihood(X)
jll = clf.predict(X)
jll = clf.predict_log_proba(X)
jll = clf.predict_proba(X)
# toc=timeit.default_timer()
# result_time = toc - tic
# logger.info(f"Prediction took:" + str(result_time))

logger.info('Save to numpy array')

"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001 or FA8702-25-D-B002. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.
© 2025 Massachusetts Institute of Technology.

Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
The software/firmware is provided to you on an As-Is basis
Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

# These constants represent default values used across the Promptodile
# pipeline's functions.
from pathlib import Path


DEFAULT_SEED = 42

# -- Shared -------------------------------------------------------------------
FT_MODEL_DIR = 'promptodile_model'

# -- QGen ---------------------------------------------------------------------
OUTPUT_FILE = 'syn_queries.jsonl'
DEFAULT_SYSTEM = 'You are a high-quality synthetic data generator. Your task is to read a document and generate a relevant query. A query is relevant if the document contains all of the necessary information to answer the query. Use the following examples to guide you. Respond with only the query.'
DEFAULT_USER = 'Document: {}'
DEFAULT_ASSISTANT = ''
QGEN_BATCH = 256

# -- Train --------------------------------------------------------------------
POOLING_MODE = 'mean'
TEST_SIZE = 0.1
MINI_BATCH_SIZE = 16
QUERY_PREFIX = ''
PASSAGE_PREFIX = ''

# -- Index/Filter -------------------------------------------------------------
CORPUS_PYSERINI = 'corpus_pyserini.jsonl'
DELIMITER = '"#~%!"'
MAX_LENGTH = 512
INDEX_BATCH = 16
RUN_NAME = 'run'
RUN_TXT = Path('run.txt')

MILVUS_DB = './promptodile.db'
VECTOR_FIELD_NAME = 'vector'
SEARCH_K = 10
CONSISTENCY_TOP_K = 3

TOPPCT = 0.1

# -- Evaluate -------------------------------------------------------------
K = 100

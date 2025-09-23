"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001 or FA8702-25-D-B002. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.
© 2025 Massachusetts Institute of Technology.

Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
The software/firmware is provided to you on an As-Is basis
Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

from typing import Any
from pydantic import BaseModel
from promptodile.config import constants
from promptodile.config.enums.similarity_function_enum import SimilarityFunctionEnum


class TrainConfig(BaseModel):
    model: str
    similarity_function: SimilarityFunctionEnum
    early_stopping_callback_args: dict[str, int | float]  # See https://huggingface.co/docs/transformers/en/main_classes/callback#transformers.EarlyStoppingCallback for options
    training_args: dict[str, Any]  # Passed directly to SentenceTransformerTrainingArguments see https://www.sbert.net/docs/sentence_transformer/training_overview.html#training-arguments for more details
    test_size: float = constants.TEST_SIZE
    mini_batch_size: int = constants.MINI_BATCH_SIZE
    
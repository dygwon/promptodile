"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001 or FA8702-25-D-B002. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.
© 2025 Massachusetts Institute of Technology.

Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
The software/firmware is provided to you on an As-Is basis
Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

from pathlib import Path
from typing import Optional
from pydantic import BaseModel, field_validator, ValidationError
from promptodile.config import constants


class SharedConfig(BaseModel):
    corpus_jsonl: Optional[Path]
    queries_jsonl: Optional[Path]
    qrels_txt: Optional[Path]
    examples_txt: Optional[Path]
    synq_jsonl: Optional[Path]
    ft_model_dir: Optional[Path]
    index_dir: Optional[Path]
    query_prefix: str = constants.QUERY_PREFIX
    passage_prefix: str = constants.PASSAGE_PREFIX

    @field_validator(
        'corpus_jsonl', 'queries_jsonl', 'qrels_txt', 'examples_txt'
    )
    @classmethod
    def validate_file(cls, v: Optional[Path]):
        if v is None:
            return None
        elif not v.is_file():
            raise ValidationError(f'Not a valid file: {v}')
        return v

"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001 or FA8702-25-D-B002. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.
© 2025 Massachusetts Institute of Technology.

Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
The software/firmware is provided to you on an As-Is basis
Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

import json
import logging
from pathlib import Path
from typing import TypeVar, Type, Iterable

logger = logging.getLogger(__name__)


T = TypeVar('T')


def load_configs_from_json(
    json_path: str,
    config_class: Type[T],
) -> T:
    logger.info('getting configs from %s', json_path)
    logger.info('validating as %s', config_class)
    with open(json_path, mode='r', encoding='utf-8') as f:
        configs_json = json.load(f)

    return config_class(**configs_json)


def beir_corpus_to_trec(beir_jsonl: str, trec_jsonl: str) -> None:
    data: list[dict[str, str]] = []
    with open(beir_jsonl, mode='r', encoding='utf-8') as fin:
        for line in fin:
            line_dict = json.loads(line)
            data.append(
                {
                    'docid': line_dict['_id'],
                    'title': line_dict.get('title', ''),
                    'body': line_dict['text'],
                }
            )

    with open(trec_jsonl, mode='w', encoding='utf-8') as fout:
        for line in data:
            json.dump(line, fout)
            fout.write('\n')


def beir_queries_to_trec(beir_jsonl: str, trec_jsonl: str) -> None:
    data: list[dict[str, str]] = []
    with open(beir_jsonl, mode='r', encoding='utf-8') as fin:
        for line in fin:
            line_dict = json.loads(line)
            data.append(
                {'id': line_dict['_id'], 'narrative': line_dict['text']}
            )

    with open(trec_jsonl, mode='w', encoding='utf-8') as fout:
        for line in data:
            json.dump(line, fout)
            fout.write('\n')

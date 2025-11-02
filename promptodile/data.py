"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001 or FA8702-25-D-B002. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.
© 2025 Massachusetts Institute of Technology.

Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
The software/firmware is provided to you on an As-Is basis
Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

import logging
import json
from typing import TypeAlias, Iterable
from pathlib import Path
from promptodile.config import shared_config

logger = logging.getLogger(__name__)


Corpus: TypeAlias = dict[str, dict[str, str]]
FewShot: TypeAlias = dict[str, dict[str, int]]
Queries: TypeAlias = dict[str, str]
SQuery: TypeAlias = dict[str, str]


class Data:
    def __init__(self, config: shared_config.SharedConfig):
        self._config = config
        logger.info('Corpus: %s', self._config.corpus_jsonl)
        logger.info('Queries: %s', self._config.queries_jsonl)
        logger.info('Few-shot Examples: %s', self._config.examples_txt)
        logger.info('Synthetic Queries: %s', self._config.synq_jsonl)

        self._corpus = None
        self._queries = None
        self._few_shot = None
        self._num_few_shot = 0
        self._synq_flat: list[SQuery] | None = None

        if self._config.corpus_jsonl:
            self._corpus = self._load_corpus()
        if self._config.queries_jsonl:
            self._queries = self._load_queries()
        if self._config.examples_txt:
            self._few_shot = self._load_few_shot()

    @property
    def num_few_shot(self) -> int:
        return self._num_few_shot

    @property
    def corpus(self) -> Corpus | None:
        return self._corpus

    @property
    def queries(self) -> Queries | None:
        return self._queries

    @property
    def few_shot(self) -> FewShot | None:
        return self._few_shot

    @property
    def syn_queries(self) -> list[Queries] | None:
        return self._synq_flat

    def _load_corpus(self) -> Corpus | None:
        logger.info('Loading corpus')

        corpus: Corpus = {}
        corpus_jsonl = self._config.corpus_jsonl
        if corpus_jsonl is None:
            return
        with open(corpus_jsonl, mode='r', encoding='utf-8') as fin:
            for line in fin:
                line_dict: dict[str, str] = json.loads(line)
                corpus[line_dict['docid']] = {
                    'title': line_dict.get('title', ''),
                    'body': line_dict['body'],
                }
        logger.info('Loaded %d documents.', len(corpus))

        return corpus

    def _load_queries(self) -> Queries | None:
        logger.info('Loading queries...')

        queries: Queries = {}
        queries_jsonl = self._config.queries_jsonl
        if queries_jsonl is None:
            return
        with open(queries_jsonl, mode='r', encoding='utf-8') as fin:
            for line in fin:
                line_dict = json.loads(line)
                qid = str(line_dict['id'])
                queries[qid] = line_dict['narrative']
        logger.info('Loaded %d queries.', len(queries))

        return queries

    def _load_few_shot(self) -> FewShot | None:
        logger.info('Loading few-shot examples.')

        few_shot: FewShot = {}
        examples_txt = self._config.examples_txt
        if examples_txt is None:
            return
        with open(examples_txt, mode='r', encoding='utf-8') as fin:
            for line in fin:
                qid, _, docid, score = line.split()
                few_shot.setdefault(qid, {})[docid] = int(score)
                self._num_few_shot += 1

        return few_shot

    @staticmethod
    def name_new_file(file: str | Path, ending: str) -> Path:
        """Create a new file from a given one with the ending.

        E.g.,
            file = /path/to/file.txt
            ending = 'flat' --> /path/to/file_flat.txt
        """
        file = Path(file)
        parent = file.parent
        new_stem = f'{file.stem}_{ending}'
        suffix = file.suffix
        return parent / (new_stem + suffix)

    def flatten_and_process_syn_queries(
        self,
        save_flat_file: bool = True,
        exclude_strs: Iterable[str] | None = None,
    ) -> list[SQuery]:
        """Flattens the syn_queries.jsonl file so that each line contains one
        synthetic query.

        The new synthetic query is a string with a key value of \"query\"

        We remove leading and trailing whitespace from each included query, if
        there if any."""
        if self._config.synq_jsonl is None:
            raise AttributeError(
                'Please provide a synthetic queries jsonl file.'
            )
        synq_jsonl = Path(self._config.synq_jsonl)
        exclude_strs = set(exclude_strs) if exclude_strs else set()

        flattened: list[SQuery] = []

        synq_flat_jsonl = self.name_new_file(synq_jsonl, 'flat')
        # Read and return the file if it already exists.
        if synq_flat_jsonl.exists():
            logger.info('Returning existing flattened file %s', synq_flat_jsonl)
            with open(synq_flat_jsonl, mode='r', encoding='utf-8') as fin:
                for line in fin:
                    flattened.append(json.loads(line))

            self._synq_flat = flattened
            return flattened

        total_queries, num_removed = 0, 0
        with open(synq_jsonl, mode='r', encoding='utf-8') as fin:
            for line in fin:
                line_dict = json.loads(line)
                queries: list[str] = line_dict['queries']
                avg_logprobs: list[float] = line_dict['avg_logprobs']
                total_queries += len(queries)
                query_logprob_iter = zip(queries, avg_logprobs)
                for i, query_logprob in enumerate(query_logprob_iter):
                    query, avg_logprob = query_logprob
                    query = query.strip()
                    # Skip empty strings or ones that we identify as skippable.
                    # If all queries for the document are skipped, the document
                    # is excluded from the flattened file.
                    if not query or query.lower() in exclude_strs:
                        num_removed += 1
                        continue

                    new_dict: SQuery = {}
                    docid = line_dict['docid']
                    new_dict['qid'] = docid + '_' + str(i)
                    new_dict['docid'] = docid
                    new_dict['query'] = query
                    new_dict['avg_logprob'] = avg_logprob
                    flattened.append(new_dict)

        logger.info('Total queries in file: %d', total_queries)
        logger.info('Queries removed: %d', num_removed)

        if save_flat_file:
            logger.info('writing to %s', synq_flat_jsonl)
            with open(synq_flat_jsonl, mode='w', encoding='utf-8') as fout:
                for line_dict in flattened:
                    json.dump(line_dict, fout)
                    fout.write('\n')

        self._synq_flat = flattened
        return flattened

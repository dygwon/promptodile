"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001 or FA8702-25-D-B002. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.
© 2025 Massachusetts Institute of Technology.

Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
The software/firmware is provided to you on an As-Is basis
Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

import json
from typing import TypeAlias
from promptodile.config import shared_config


Corpus: TypeAlias = dict[str, dict[str, str]]
FewShot: TypeAlias = dict[str, dict[str, int]]
Queries: TypeAlias = dict[str, str]

class Data:
    def __init__(self, config: shared_config.SharedConfig):
        self._config = config
        print(f'Corpus: {self._config.corpus_jsonl}')
        print(f'Queries: {self._config.queries_jsonl}')
        print(f'Few-shot Examples: {self._config.examples_txt}')
        
        self._corpus = None
        self._queries = None
        self._few_shot = None
        self._num_few_shot = 0
        
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
    
    def _load_corpus(self) -> Corpus | None:
        print('Loading corpus')
        
        corpus: Corpus = {}
        corpus_jsonl = self._config.corpus_jsonl
        if corpus_jsonl is None:
            return
        with open(corpus_jsonl, mode='r', encoding='utf-8') as fin:
            for line in fin:
                line_dict = json.loads(line)
                corpus[line_dict['docid']] = {
                    'title': line_dict['title'],
                    'body': line_dict['body']
                }
        print(f'Loaded {len(corpus)} documents.')
                
        return corpus
    
    def _load_queries(self) -> Queries | None:
        print('Loading queries...')
        
        queries: Queries = {}
        queries_jsonl = self._config.queries_jsonl
        if queries_jsonl is None:
            return
        with open(queries_jsonl, mode='r', encoding='utf-8') as fin:
            for line in fin:
                line_dict = json.loads(line)
                queries[line_dict['id']] = line_dict['narrative']
        print(f'Loaded {len(queries)} queries.')
        
        return queries
    
    def _load_few_shot(self) -> FewShot | None:
        print('Loading few-shot examples.')
        
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

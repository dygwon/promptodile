"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001 or FA8702-25-D-B002. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.
© 2025 Massachusetts Institute of Technology.

Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
The software/firmware is provided to you on an As-Is basis
Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

import sys
import json
import logging
import subprocess
import torch
from pathlib import Path
from typing import Any
from tqdm import tqdm
from pyserini.search.faiss import FaissSearcher # type: ignore
from promptodile import data
from promptodile.config import index_config, shared_config
from promptodile.config import constants
from pyserini.encode import AutoQueryEncoder # type: ignore


logger = logging.getLogger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Index:
    def __init__(
        self,
        config: index_config.IndexConfig,
        sconfig: shared_config.SharedConfig,
    ):
        logger.info(config)
        logger.info(sconfig)
        
        self._config = config
        self._sconfig = sconfig
        if self._sconfig.index_dir and not self._sconfig.index_dir.exists():
            self._sconfig.index_dir.mkdir(parents=True, exist_ok=True)
        
        self._data = data.Data(self._sconfig)
        self._corpus_pyserini = self._corpus_to_pyserini()
        
    def _corpus_to_pyserini(self) -> Path:
        """Format corpus content to support indexing via Pyserini.
        
        synq_jsonl: synthetic query jsonl file created during Query Generation.
        
        The newly formatted file is saved in the same directory as the synthetic
        queries jsonl file."""
        corpus_jsonl = self._sconfig.corpus_jsonl
        if corpus_jsonl is None or not corpus_jsonl.is_file():
            raise AttributeError('Please provide a valid file %s', str(corpus_jsonl))
        if self._sconfig.index_dir is None:
            raise AttributeError('Please provide an index directory')

        # The pyserini-styled documents will vary since different embedding models
        # use different query and document/passage prefixes
        index_name = self._sconfig.index_dir.name
        pyserini_path = (
                corpus_jsonl.parent
                / f'{corpus_jsonl.stem}_{index_name}_pyserini{corpus_jsonl.suffix}')
        if pyserini_path.exists():
            logging.info('reading existing pyserini corpus file %s', str(pyserini_path))
            return pyserini_path
        else:
            logging.info('saving pyserini corpus file to %s', str(pyserini_path))
        
            data: list[dict[str, str]] = []
            with open(corpus_jsonl, mode='r', encoding='utf-8') as fin:
                for line in fin:
                    jsonl = json.loads(line)
                    contents = jsonl['body']

                    # Pyserini's built-in prefixing doesn't add a colon, whereas Promptodile
                    # does, so we build the document/passage manually.
                    if 'embeddinggemma-300m' in self._config.ft_model_dir:
                        title = jsonl.get('title', 'none')
                        p_pre = self._sconfig.passage_prefix.format(title)
                        contents = f'{p_pre}: {contents}'
                    elif 'title' in jsonl:
                        contents = jsonl['title'] + '\n\n' + contents

                    data.append({'id': jsonl['docid'], 'contents': contents})
            
            with open(pyserini_path, mode='w', encoding='utf-8') as fout:
                for line in data:
                    json.dump(line, fout)
                    fout.write('\n')
        
        return pyserini_path
    
    def generate_run(self) -> None:
        qrels_txt = self._sconfig.qrels_txt
        if qrels_txt is None:
            raise AttributeError('Please provide a valid file %s', qrels_txt)
        
        logger.info('generating run')
        query_encoder = AutoQueryEncoder(
            encoder_dir=str(self._sconfig.ft_model_dir),
            pooling=self._config.pooling.value,
            l2_norm=self._config.l2_norm,
            device=device)
        
        searcher = FaissSearcher(
            str(self._sconfig.index_dir), query_encoder)
        
        queries = self._data.queries
        if queries is None:
            raise AttributeError('Please provide a valid queries file')
        
        self._config.run_txt.parent.mkdir(parents=True, exist_ok=True)
        test_qids: set[str] = set()
        with open(qrels_txt, 'r') as f:
            for line in f:
                qid = line.strip().split()[0]
                test_qids.add(qid)

        with open(self._config.run_txt, mode='w', encoding='utf-8') as fout:
            for qid, qtext in tqdm(queries.items(), total=len(queries)):
                if qid in test_qids: # No need to run for queries not in our test qids (will be faster....)

                    # Pyserini's built-in prefixing doesn't add a colon, whereas Promptodile
                    # does, so we build the query manually.
                    if self._sconfig.query_prefix:
                        query = f'{self._sconfig.query_prefix}: {qtext}'
                    else:
                        query = qtext
                    hits = searcher.search(query, k=constants.K) # type: ignore
                    
                    rows: list[str] = []
                    for i, hit in enumerate(hits): # type: ignore
                        rows.append(f'{qid} Q0 {hit.docid} {i} {hit.score} ' # type: ignore
                                    f'{self._config.run_name}')
                    fout.write('\n'.join(rows))
                    fout.write('\n')
                    
    def index(self) -> None:
        config = self._config
        sconfig = self._sconfig
                
        input_args: list[Any] = [
            'input',
            '--corpus', self._corpus_pyserini,
            '--delimiter', constants.DELIMITER  # Prevents pyserini from splitting on newlines contained in documents.
        ]
        output_args: list[Any] = [
            'output',
            '--embeddings', sconfig.index_dir,
            '--to-faiss'
        ]
        encoder_args: list[Any] = [
            'encoder',
            '--encoder', sconfig.ft_model_dir,
            '--pooling', config.pooling.value,
            '--max-length', str(config.max_length),
            '--dimension', str(config.emb_dim),
            '--batch', str(config.batch),
            '--fp16',
            '--device', device
        ]
        
        if 'embeddinggemma-300m' in self._config.ft_model_dir:
            encoder_args.remove('--fp16')
        if self._config.l2_norm:
            encoder_args.append('--l2-norm')
        
        subp_cmd = [sys.executable, '-m', 'pyserini.encode']
        subp_cmd.extend(input_args)
        subp_cmd.extend(output_args)
        subp_cmd.extend(encoder_args)
        logger.info('Executing: %s' % subp_cmd)
        subprocess.run(subp_cmd)
    
    
if __name__ == '__main__':
    from promptodile import utils
    from promptodile import configure_logging
    
    if len(sys.argv) == 4:
        level = int(sys.argv[3])
        configure_logging(level=level)
    
    config_json = sys.argv[1]
    sconfig_json = sys.argv[2]
    
    config = utils.load_configs_from_json(
        config_json, index_config.IndexConfig)
    sconfig = utils.load_configs_from_json(
        sconfig_json, shared_config.SharedConfig)
    
    qrels_txt = sconfig.qrels_txt
    if qrels_txt is None:
        raise AttributeError('Please provide a valid file %s', qrels_txt)
        
    idx = Index(config, sconfig)
    idx.index()
    idx.generate_run()
    
    subprocess.run([
        sys.executable, '-m', 'pyserini.eval.trec_eval',
        '-c', '-m', 'ndcg_cut.10',
        qrels_txt, config.run_txt])

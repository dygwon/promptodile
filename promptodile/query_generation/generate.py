"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001 or FA8702-25-D-B002. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.
© 2025 Massachusetts Institute of Technology.

Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
The software/firmware is provided to you on an As-Is basis
Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

import logging
import fcntl
import json
import random
import torch
import copy
from tqdm import trange
from promptodile.config import qgen_config, shared_config
from promptodile.query_generation import qgen
from promptodile import data

logger = logging.getLogger(__name__)


class Generate:
    def __init__(
        self,
        config: qgen_config.QGenConfig,
        sconfig: shared_config.SharedConfig
    ):
        logger.info(config)
        logger.info(sconfig)
        self._sconfig = sconfig
        self._config = config
        self._data = data.Data(self._sconfig)
        self._qgen = qgen.QGen(self._config)

    def _write_jsonl_shared(
        self,
        output_data: list[dict[str, str | list[str]]],
    ) -> None:
        """Write to the output jsonl file.
        
        When generating queries for a large number of documents, it can be more
        efficient to split up the documents and process them separately, in
        parallel. We use a file lock to support that use case."""
        synq_jsonl = self._sconfig.synq_jsonl
        if synq_jsonl is None:
            raise AttributeError(f'Please provide a valid file {synq_jsonl}')
        elif not synq_jsonl.exists():
            synq_jsonl.parent.mkdir(parents=True, exist_ok=True)
            
        logger.info(f'writing to shared file {synq_jsonl}')
        with open(synq_jsonl, mode='a', encoding='utf-8') as fout:
            fcntl.flock(fout, fcntl.LOCK_EX)
            for obj in output_data:
                json.dump(obj, fout)
                fout.write('\n')
            fcntl.flock(fout, fcntl.LOCK_UN)
            
    def _save_batch(
        self,
        batch_ids: list[str],
        corpus_dict: data.Corpus,
        output_text: list[list[str]],
        output_logprobs: list[list[float]],
    ) -> None:
        """Reformat and save the generated data.
        
        The formatting of the saved data mirrors that of the corpus.jsonl file,
        with the addition of a 'queries' field that holds an array of synthetic
        queries."""
        save_data: list[dict[str, str | list[str]]] = []
        for cid, queries, avg_logprobs in zip(
            batch_ids, output_text, output_logprobs):
            data: dict[str, str | list[str]] = {
                'docid': cid,
                'title': corpus_dict[cid].get('title', ''),
                'body': corpus_dict[cid]['body'],
                'queries': queries,
                'avg_logprobs': avg_logprobs
            }
            save_data.append(data)
        
        self._write_jsonl_shared(save_data)
    
    def _build_batch_conversations(
        self,
        batch_text: list[str],
        few_shot_prompt: list[qgen.Chat],
    ) -> list[list[qgen.Chat]]:
        """Take all the documents in a batch and create individual converations
        for each.
        
        Each conversation shares the same few-shot prompt."""
        conversations: list[list[qgen.Chat]] = []
        for text in batch_text:
            if self._qgen.user:
                u_content = self._qgen.user.format(text)
            else:
                u_content = text
                
            conversation = copy.deepcopy(few_shot_prompt)
            
            conversation.append(self._qgen.format_chat('user', u_content))
            conversations.append(conversation)
    
        return conversations
    
    def _build_corpus_for_generation(self) -> data.Corpus:
        """Remove few-shot examples and select subset of corpus, if
        specified."""
        corpus_dict = self._data.corpus
        if corpus_dict is None:
            raise AttributeError('Please provide a valid corpus jsonl file.')
        
        logger.info('corpus size: %d', len(corpus_dict))
        
        few_shot = self._data.few_shot
        max_corpus_size = self._qgen.config.max_corpus_size
        
        # Remove few-shot examples from the corpus before sampling a subset.
        # This protects us from ever having fewer than the max_corpus_length.
        few_shot_corpus_ids: set[str] = set()
        if few_shot:
            # Gather all the corpus ids used in the few shot examples.
            for query_id in few_shot:
                for corpus_id in few_shot[query_id]:
                    few_shot_corpus_ids.add(corpus_id)
        
        # Gather all the text in the corpus, excluding those used in the
        # few-shot examples.
        selected_ids = set(corpus_dict.keys()) - few_shot_corpus_ids
        if max_corpus_size and len(selected_ids) > max_corpus_size:
            logger.info(
                'selecting %d documents for corpus of size %d',
                max_corpus_size, len(selected_ids))
            
            selected_ids = random.sample(list(selected_ids), max_corpus_size)
            
        new_corpus_dict: data.Corpus = {}
        for corpus_id in selected_ids:
            new_corpus_dict[corpus_id] = corpus_dict[corpus_id]
        
        return new_corpus_dict
        
    def _build_few_shot_prompt(self) -> list[qgen.Chat]:
        """Build the few-shot prompt.
        
        The building process is highly dependent on the provided prompt templates.
        Please overload this function as needed to match your corpus and desired
        prompt format."""
        few_shot = self._data.few_shot
        queries = self._data.queries
        corpus_dict = self._data.corpus
        if few_shot is None or queries is None or corpus_dict is None:
            raise AttributeError('Please provide valid corpus, queries, and '
                                 'examples files.')
        elif (self._qgen.system is None
              or self._qgen.user is None
              or self._qgen.assistant is None):
            raise AttributeError('Please provide the required prompt '
                                 'templates.')
        
        system = self._qgen.system
        user = self._qgen.user
        assistant = self._qgen.assistant
        format_chat = self._qgen.format_chat
        
        few_shot_prompt: list[qgen.Chat] = []
        if system:
            few_shot_prompt.append(format_chat('system', system))
        
        for qid in few_shot:
            query = queries[qid]
            a_content = assistant.format(query)
            
            for cid in few_shot[qid]:
                corpus_text = corpus_dict[cid]['body']
                u_content = user.format(corpus_text)
                
                few_shot_prompt.append(format_chat('user', u_content))
                few_shot_prompt.append(format_chat('assistant', a_content))
        
        logging.info('Few shot prompt: %s', json.dumps(few_shot_prompt))
    
        return few_shot_prompt    

    def generate(self) -> None:
        sampling_params = self._qgen.get_sampling_params()
        
        if self._data.num_few_shot:
            prompt = self._build_few_shot_prompt()
        else:
            prompt = []
        
        corpus_gen = self._build_corpus_for_generation()
        cids = list(corpus_gen.keys())
        
        batch_size = self._qgen.config.batch_size
        for i in trange(0, len(cids), batch_size):
            batch_ids = cids[i:i+batch_size]
            batch_text: list[str] = []
            for cid in batch_ids:
                # Use the title if we're given one.
                text = corpus_gen[cid].get('title', '')
                if len(text):
                    text += f"\n\n{corpus_gen[cid]['body']}"
                else:
                    text = corpus_gen[cid]['body']
                batch_text.append(text)
            conversations = self._build_batch_conversations(batch_text, prompt)
        
            try:
                output_text, output_logprobs = self._qgen.chat(
                    conversations, sampling_params)
            except torch.OutOfMemoryError as e:
                # We lose the entire batch if there are OOM errors.
                print(f'Batch {i}: {e}', flush=True)
                continue
            
            self._save_batch(
                batch_ids, corpus_gen, output_text, output_logprobs)


if __name__ == '__main__':
    import sys
    from promptodile import utils
    from promptodile import configure_logging
    
    if len(sys.argv) == 4:
        level = int(sys.argv[3])
        configure_logging(level=level)
    
    config_json = sys.argv[1]
    sconfig_json = sys.argv[2]
    
    config = utils.load_configs_from_json(config_json, qgen_config.QGenConfig)
    sconfig = utils.load_configs_from_json(
        sconfig_json, shared_config.SharedConfig)
    
    generate = Generate(config, sconfig)
    generate.generate()

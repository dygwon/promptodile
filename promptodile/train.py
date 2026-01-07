"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001 or FA8702-25-D-B002. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.
© 2025 Massachusetts Institute of Technology.

Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
The software/firmware is provided to you on an As-Is basis
Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

import os
import json
import logging
from datasets import Dataset # type: ignore[import]
from sklearn.model_selection import train_test_split # type: ignore[reportUnknownVariableType]
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerTrainer,
    models,
    losses,
    util
)
from transformers.trainer_callback import EarlyStoppingCallback

from promptodile.config.enums.similarity_function_enum import SimilarityFunctionEnum
from promptodile.config import train_config, shared_config
from promptodile.config import constants
from promptodile import data
from pathlib import Path

logger = logging.getLogger(__name__)


class Train:
    def __init__(
        self,
        config: train_config.TrainConfig,
        sconfig: shared_config.SharedConfig,
    ):
        logger.info(config)
        logger.info(sconfig)
        self._config = config
        self._sconfig = sconfig
        self._training_args = self._config.training_args
        
        if self._sconfig.ft_model_dir:
            logger.info('saving to %s', self._sconfig.ft_model_dir)
            self._training_args['output_dir'] = self._sconfig.ft_model_dir
        else:
            self._training_args['output_dir'] = constants.FT_MODEL_DIR
        
        synq_jsonl = self._sconfig.synq_jsonl
        if synq_jsonl is None:
            raise AttributeError(f'Please provide a valid file {synq_jsonl}')
        filtered_jsonl = data.Data.name_new_file(synq_jsonl, 'filtered')
        if Path(filtered_jsonl).exists():
            logger.info('Filtered data exists')
            synq_jsonl = filtered_jsonl

        logger.info(f'Loading data from {synq_jsonl}')
        with open(synq_jsonl, mode='r', encoding='utf-8') as fin:
            self._data = [json.loads(line) for line in fin]
        
        self._model = self._init_model()
        self._loss = self._init_loss()
    
    def _init_model(self) -> SentenceTransformer:
        logger.info('Initializing model from %s', self._config.model)
        base_model = models.Transformer(self._config.model)
        emb_dim = base_model.get_word_embedding_dimension()
        pooling_model = models.Pooling(
            emb_dim, pooling_mode=constants.POOLING_MODE)
        model = SentenceTransformer(modules=[base_model, pooling_model])
        return model
    
    def _prepare_dataset(self) -> tuple[Dataset, Dataset]:
        logger.info('preparing dataset')
        q_pre = self._sconfig.query_prefix
        p_pre = self._sconfig.passage_prefix
        
        queries: list[str] = []
        documents: list[str] = []
        for line in self._data:
            for syn_q in line['queries']:
                if not syn_q:  # We don't want empty queries.
                    continue
                query = f'{q_pre}: {syn_q}' if q_pre else syn_q

                # The document prefix for google/embeddinggemma-300m takes an optional
                # title.
                if p_pre and 'embeddinggemma-300m' in self._config.model:
                    title = line.get('title', '')
                    title = 'none' if len(title) == 0 else title
                    p_pre = p_pre.format(title)
                
                doc = f'{p_pre}: {line["body"]}' if p_pre else line['body']
                queries.append(query)
                documents.append(doc)
        
        if len(queries) != len(documents):
            raise ValueError('There should be one document for each query.')
        
        logger.info('Example:')
        logger.info('Query: %s', queries[0])
        logger.info('Document: %s', documents[0])
        logger.info('%d query-document pairs', len(queries))
        
        logger.info('generating train/test split...')
        q_train, q_test, d_train, d_test = train_test_split( # type: ignore
            queries,
            documents,
            test_size=self._config.test_size,
            random_state=int(os.getenv('RANDOM_SEED', constants.DEFAULT_SEED)))
        
        train_dataset = Dataset.from_dict({ # type: ignore
            'anchor': q_train,
            'positive': d_train
        })
        eval_dataset = Dataset.from_dict({ # type: ignore
            'anchor': q_test,
            'positive': d_test
        })
        
        return train_dataset, eval_dataset

    def _init_loss(self):
        logger.info('initializing loss')
        
        if self._config.similarity_function == SimilarityFunctionEnum.DOT:
            similarity_fct = util.dot_score # type: ignore
            scale = 1.0  # per https://sbert.net/docs/package_reference/sentence_transformer/losses.html#sentence_transformers.losses.CachedMultipleNegativesRankingLoss
        else:
            similarity_fct = util.cos_sim # type: ignore
            scale = 20.0  # this is the default
        logger.info(
            'similarity function: %s, scale: %f', similarity_fct, scale) # type: ignore
            
        loss = losses.CachedMultipleNegativesRankingLoss(
            self._model,
            scale=scale,
            similarity_fct=similarity_fct,
            mini_batch_size=self._config.mini_batch_size)
        
        return loss

    def train(self):
        train_dataset, eval_dataset = self._prepare_dataset()
        args = SentenceTransformerTrainingArguments(**self._training_args)
        trainer = SentenceTransformerTrainer(
            model=self._model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=self._loss,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        logger.info('Training...')
        trainer.train() # type: ignore
        trainer.save_model(str(self._training_args['output_dir']))


if __name__ == '__main__':
    import sys
    from promptodile import utils
    from promptodile import configure_logging
    
    if len(sys.argv) == 4:
        level = int(sys.argv[3])
        configure_logging(level=level)
    
    config_json = sys.argv[1]
    sconfig_json = sys.argv[2]
    
    config = utils.load_configs_from_json(
        config_json, train_config.TrainConfig)
    sconfig = utils.load_configs_from_json(
        sconfig_json, shared_config.SharedConfig)
    
    train = Train(config, sconfig)
    train.train()
    

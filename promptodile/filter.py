import json
import logging
import numpy as np
from pathlib import Path
from promptodile.config import shared_config, filter_config
from promptodile import data
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from promptodile.config import constants
from tqdm import trange

logger = logging.getLogger(__name__)


type RetrievedResults = list[list[dict[str, int | float | dict[str, str]]]]


class Filter:
    def __init__(
        self,
        config: filter_config.FilterConfig,
        sconfig: shared_config.SharedConfig,
    ):
        logger.info(config)
        logger.info(sconfig)

        self._config = config
        self._sconfig = sconfig

        self._col_name = self._config.collection_params.collection_name
        self._data = data.Data(self._sconfig)

        if self._sconfig.synq_jsonl is None:
            raise AttributeError('Please provide a synthetic queries file')
        self._synq_jsonl = self._sconfig.synq_jsonl
        self._synq_flat = self._data.flatten_and_process_syn_queries(
            exclude_strs=self._sconfig.exclude_strs
        )
        self._ret_jsonl = self._data.name_new_file(
            self._synq_jsonl, 'retrieved'
        )
        self._filtered_jsonl = self._data.name_new_file(
            self._synq_jsonl, 'filtered'
        )

        self._model = self._init_model()

        self._db_path = Path(self._config.db_dir) / constants.MILVUS_DB
        logger.info(f'using milvus lite database: {self._db_path}')
        self._client = MilvusClient(str(self._db_path))

    def _init_model(self) -> SentenceTransformer:
        return SentenceTransformer(self._config.model)

    def _create_collection(self) -> None:
        col_params = self._config.collection_params

        if self._client.has_collection(self._col_name):
            logger.info('dropping existing collection')
            self._client.drop_collection(self._col_name)

        self._client.create_collection(**col_params.model_dump())

    def _insert_batch(self, docids: list[str], texts: list[str]):
        logger.info('embedding and inserting %d documents', len(docids))
        if self._sconfig.passage_prefix:
            ppre = f'{self._sconfig.passage_prefix}: '
        else:
            ppre = ''
        texts_w_prefix = [f'{ppre}{text}'.strip() for text in texts]
        embeddings = self._model.encode(texts_w_prefix)

        insert_data: list[dict[str, str | np.ndarray]] = []
        vector_field_name = self._config.collection_params.vector_field_name
        for docid, embedding, text in zip(docids, embeddings, texts):
            insert_data.append(
                {
                    'docid': docid,
                    vector_field_name: embedding,
                    'text': text,
                }
            )

        self._client.insert(collection_name=self._col_name, data=insert_data)

    def index(self) -> None:
        corpus = self._data.corpus
        if corpus is None:
            raise AttributeError('Please provide a corpus file.')

        logger.info('indexing %d documents', len(corpus))
        docids: list[str] = []
        texts: list[str] = []
        for docid in corpus:
            corpus_dict = corpus[docid]
            title = corpus_dict.get('title', '')
            body = corpus_dict['body']
            text = f'{title}\n\n{body}' if title else body

            docids.append(docid)
            texts.append(text)

            if len(docids) >= self._config.batch_size:
                self._insert_batch(docids, texts)
                docids.clear()
                texts.clear()

        if docids:
            self._insert_batch(docids, texts)
            docids.clear()
            texts.clear()

    def _save_results_batch(
        self,
        qid_batch: list[str],
        source_docids_batch: list[str],
        results: RetrievedResults,
    ) -> None:
        logger.info(
            'saving retrieved results for %d queries to %s',
            len(qid_batch),
            self._ret_jsonl,
        )

        with open(self._ret_jsonl, mode='a', encoding='utf-8') as fout:
            for qid, source_docid, result in zip(
                qid_batch,
                source_docids_batch,
                results,
            ):
                res_dict = {
                    'qid': qid,
                    'source_docid': source_docid,
                    'docids': [
                        ret_doc['entity']['docid'] for ret_doc in result
                    ],
                }
                json.dump(res_dict, fout)
                fout.write('\n')

    def retrieve(self, k: int = constants.SEARCH_K) -> None:
        if self._sconfig.query_prefix:
            qpre = f'{self._sconfig.query_prefix}: '
            logger.info('using prefix "%s"', qpre)
        else:
            qpre = ''
        qids: list[str] = []
        squeries: list[str] = []
        source_docids: list[str] = []
        for line in self._synq_flat:
            qids.append(line['qid'])
            squeries.append(qpre + line['query'])
            source_docids.append(line['docid'])

        batch_size = self._config.batch_size
        metric_type = self._config.collection_params.metric_type
        for i in trange(0, len(qids), batch_size, desc='Retrieving'):
            qid_batch = qids[i : i + batch_size]
            source_docids_batch = source_docids[i : i + batch_size]
            squery_batch = squeries[i : i + batch_size]
            qembs = self._model.encode(squery_batch)

            logger.debug('searching')
            results = self._client.search(
                collection_name=self._col_name,
                data=qembs,
                limit=k,
                output_fields=['docid'],
                search_params={'metric_type': metric_type},
            )

            self._save_results_batch(qid_batch, source_docids_batch, results)

    def _get_keepers(
        self,
        k: int = constants.CONSISTENCY_TOP_K,
    ) -> dict[str, list[str]]:
        """Go through the jsonl of retrieved documents and only keep those where
        the synthetic query's document is found in the top k.

        Resulting data structure is a dictionary mapping source document ids to
        the list of queries. For example,

        {
            DOCID1: [query1, query2, query3, ...],
            DOCID2: [query2, query4, query4, ...],
            ...
        }
        """
        logger.info('reading %s', self._ret_jsonl)
        retrieved: list[dict[str, str | list[str]]] = []
        with open(self._ret_jsonl, mode='r', encoding='utf-8') as fin:
            for line in fin:
                retrieved.append(json.loads(line))

        num_keep = 0
        docids_to_qids: dict[str, list[str]] = {}
        for line in retrieved:
            qid = line['qid']
            source_docid = line['source_docid']
            topk = line['docids'][:k]
            if not isinstance(qid, str):
                logger.error('qid is not a str %s', qid)
                continue
            if not isinstance(topk, list):
                logger.error(
                    'docids from retrieved file is not a list %s',
                    topk,
                )
                continue

            if source_docid in topk:
                docids_to_qids.setdefault(source_docid, []).append(qid)
                num_keep += 1

        logger.info(
            'keep %d/%d (%.1f%%)',
            num_keep,
            len(retrieved),
            round(num_keep / len(retrieved) * 100, 1),
        )

        return docids_to_qids

    def _qids_to_queries(self) -> dict[str, str]:
        """Create a simple mapping of query ids to query texts."""
        qids_to_queries: dict[str, str] = {}
        for line in self._synq_flat:
            qid = line['qid']
            query = line['query']
            if qid in qids_to_queries:
                logger.error('Duplicate query id found: %s', qid)
                logger.error('Keeping the first text encountered.')
                continue
            qids_to_queries[qid] = query

        return qids_to_queries

    def _qids_to_avg_logprobs(self) -> dict[str, float]:
        """Create a simple mapping of query ids to average log probabilities"""
        qids_to_logprobs: dict[str, float] = {}
        for line in self._synq_flat:
            qid = line['qid']
            avg_logprob = line['avg_logprob']
            if qid in qids_to_logprobs:
                logger.error('Duplicate query id found: %s', qid)
                logger.error('Keeping the first text encountered.')
                continue
            qids_to_logprobs[qid] = avg_logprob

        return qids_to_logprobs

    def _save_consistent(self, k: int = constants.CONSISTENCY_TOP_K) -> None:
        logger.info('saving "consistent" queries')
        if self._data.corpus is None:
            raise AttributeError('Please provide a valid corpus file.')

        docids_to_qids = self._get_keepers(k=k)
        qids_to_query = self._qids_to_queries()
        qids_to_logprobs = self._qids_to_avg_logprobs()

        with open(self._filtered_jsonl, mode='w', encoding='utf-8') as fout:
            for docid, qids in docids_to_qids.items():
                output = {
                    'docid': docid,
                    'title': self._data.corpus[docid].get('title', ''),
                    'body': self._data.corpus[docid]['body'],
                    'queries': [qids_to_query[qid] for qid in qids],
                    'avg_logprobs': [qids_to_logprobs[qid] for qid in qids]
                }
                json.dump(output, fout)
                fout.write('\n')

    def consistency_filter(self):
        logger.info('conducting consistency filtering')
        self._create_collection()
        self.index()
        self.retrieve()
        self._save_consistent()

    def toppct_filter(self, toppct: float=constants.TOPPCT):
        logger.info('selecting top %f\%')
        pass


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
        config_json, filter_config.FilterConfig
    )
    sconfig = utils.load_configs_from_json(
        sconfig_json, shared_config.SharedConfig
    )

    filter = Filter(config, sconfig)
    filter.consistency_filter()

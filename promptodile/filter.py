import logging
from promptodile.config import shared_config, filter_config
from promptodile import data
from pymilvus import MilvusClient

logger = logging.getLogger(__name__)


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
        self._data = data.Data(self._sconfig)
        self._synq_flat = self._data.flatten_and_process_syn_queries(
            exclude_strs=self._sconfig.exclude_strs
        )

        logger.info('saving milvus lite database to ./promptodile.db')
        self._client = MilvusClient('./promptodile.db')

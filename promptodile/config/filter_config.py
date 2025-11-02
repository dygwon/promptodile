from pydantic import BaseModel
from promptodile.config.collection_params import CollectionParams


class FilterConfig(BaseModel):
    collection_params: CollectionParams
    batch_size: int
    model: str
    db_dir: str

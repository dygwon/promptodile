from pydantic import BaseModel
from promptodile.config import constants


class CollectionParams(BaseModel):
    collection_name: str
    dimension: int
    metric_type: str
    auto_id: bool = True
    vector_field_name: str = constants.VECTOR_FIELD_NAME

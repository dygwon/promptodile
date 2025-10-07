from pydantic import BaseModel


class CollectionParams(BaseModel):
    collection_name: str
    dimension: int
    metric_type: str

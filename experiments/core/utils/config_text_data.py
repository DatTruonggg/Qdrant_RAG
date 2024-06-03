from datetime import datetime
from typing import Optional
from numpy import ndarray
from pydantic import BaseModel, Field, computed_field, ConfigDict
from uuid import UUID


"""
This class aim to config payload for text data
"""
class TextData(BaseModel):
    id: UUID #id of text
    content: str
    url: Optional[str] = None #TODO: url của pdf file -> có nghĩa chỗ này sẽ bổ trợ tính năng add tài liệu vào system
    language: Optional[str] = None
    txt_vector: Optional[ndarray] = Field(None, exclude=True)
    index_date: datetime
    categories: Optional[list[str]] = []    
    format: Optional[str] = None #maybe for s3 in AWS
    local: Optional[bool] = False

    @computed_field()
    @property
    def payload(self):
        result = self.model_dump(exclude={"id", "index_date"})
        result['index_date'] = self.index_date.isoformat()
        return result       

    @classmethod
    def from_payload(cls, id: str, payload: dict, txt_vector: Optional[ndarray] = None):
        index_date = datetime.fromisoformat(payload['index_date'])
        del payload['index_date']
        return cls(id=UUID(id),
                   index_date=index_date,
                   **payload,
                   text_contain_vector=txt_vector if txt_vector is not None else None)        

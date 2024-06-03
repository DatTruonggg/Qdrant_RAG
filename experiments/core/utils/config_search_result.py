from pydantic import BaseModel
from .config_text_data import TextData

class SearchResult(BaseModel):
    txt_data: TextData
    score: float
import os
from typing import List, Optional
import numpy as np

from grpc.aio import AioRpcError
from httpx import HTTPError

from utils.retry_deco_async import *
from utils.config_text_data import TextData 
from utils.config_search_result import SearchResult
from experiments.setting.setting import rag_setting, ModeSetting, AdvancedSearchModelSetting, AdvanceSearchSetting

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import RecommendStrategy
from qdrant_client.http import models
from dotenv import load_dotenv

import logging
load_dotenv()

class QdrantDB:
    AVAILABLE_POINT_TYPES = models.Record | models.ScoredPoint | models.PointStruct

    def __init__(self):
        match rag_setting.qdrant.mode:
            case ModeSetting.SERVER:
                self._qd = AsyncQdrantClient(host=rag_setting.qdrant.host,
                                             port=rag_setting.qdrant.port,
                                             grpc_port=rag_setting.qdrant.grpc_port,
                                             api_key=rag_setting.qdrant.api_key,
                                             )
                wrap_object(self._qd, retry_async(AioRpcError, HTTPError)) # TODO: tìm cách giải thích chỗ này
                self._local = False
            case ModeSetting.LOCAL:
                self._qd = AsyncQdrantClient(path=rag_setting.qdrant.local_path)
                self._local = True
            case ModeSetting.MEMORY:
                logging.warn("If you use In-Memory, dataa will be lost after application restart")
                self._qd = AsyncQdrantClient(":memory:")
                self._local = True
            case _:
                raise ValueError("Qdrant Mode Error. Please choose a mode!!!")
        self.collection_name = rag_setting.qdrant.collection_name

    @classmethod
    def _get_vector_from_txt(cls, text_data:TextData) -> models.PointVectors:
        return models.PointVectors(id = text_data.id,
                                   vector=text_data.txt_vector.tolist())

    @classmethod
    def _get_point_from_text(cls, text_data: TextData) -> models.PointStruct:
        return models.PointStruct(id = str(text_data.id),
                                  payload = text_data.payload, ##TODO: check xem chỗ này đúng không?
                                  vector = cls._get_vector_from_txt(text_data).vector)

    def _get_text_from_point(self, point: AVAILABLE_POINT_TYPES) -> TextData:
        return (TextData.from_payload(point.id,
                                      point.payload.copy() if self._local else point.payload,
                                      txt_vector=np.array(point.vector.self.txt_vector), dtype = np.float32))

    def _get_text_from_points(self, points: list[AVAILABLE_POINT_TYPES]) -> list[TextData]:
        return [self._get_text_from_point(t) for t in points]

    def _get_search_result_from_scored_point(self, point: models.ScoredPoint) -> SearchResult:
        return SearchResult(text = self._get_text_from_point(point), score=point.score)

    async def onload(self):
        if not await self.check_collection():
            logging.warn("Collection not found. Initializing...")
            await self.initialize_collection()

    async def query_search(self, query_vector,
                          skip: int = 0,
                          top_k: int = 10) -> List[SearchResult]:
        logging.info("Querying Qdrant... top_k = {}", top_k)
        result = await self._qd.search(collection_name=self.collection_name,
                                 query_vector=query_vector, ## TODO: check chỗ query này -> nó phải dạng vector
                                 limit=top_k,
                                 offset=skip,
                                 with_payload=True)
        logging.success("Query completed!")
        return result
    async def query_similarity(self,
                               search_id: Optional[str] = None,
                                positive_vectors: Optional[list[np.ndarray]] = None,
                                negative_vectors: Optional[list[np.ndarray]] = None,
                                mode: Optional[AdvancedSearchModelSetting] = None,
                                with_vector: bool =False,
                                top_k: int = 10,
                                skip: int = 0
                                ) -> List[SearchResult]:
        _positive_vectors = [t.tolist() for t in positive_vectors] if positive_vectors is not None else [search_id]
        _negative_vectors = [t.tolist() for t in negative_vectors] if negative_vectors is not None else None
        _strategy = None if mode is None else (RecommendStrategy.AVERAGE_VECTOR if
                                               mode == AdvanceSearchSetting.average else RecommendStrategy.BEST_SCORE)
        logging.info("Querying Qdrant... top_k = {}", top_k)
        result = await self._qd.recommend(collection_name=self.collection_name,
                                              using = "text",
                                              positive=_positive_vectors,
                                              negative=_negative_vectors,
                                              strategy=_strategy,
                                              with_vectors=with_vector,
                                              limit=top_k,
                                              offset=skip,
                                              with_payload=True)
        logging.success("Query completed!")               
        return [self._get_search_result_from_scored_point(t) for t in result]
    
    async def insertVector(self, vector: list[TextData]):
        logging.info("Inserting {} vectors inrto Qdrant....", len(vector))
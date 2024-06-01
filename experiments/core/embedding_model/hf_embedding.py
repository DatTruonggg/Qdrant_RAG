import os

class Embedding():
    @staticmethod
    def get_hf_model(model_name: str):
        from sentence_transformers import SentenceTransformer
        return model_name and SentenceTransformer(model_name)
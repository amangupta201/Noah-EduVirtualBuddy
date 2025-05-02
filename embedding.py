from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from typing import Union, List

class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def encode(self, texts: list[str]) -> np.ndarray:
        # Filter out any empty strings
        texts = [t for t in texts if t.strip()]
        if not texts:
            return np.array([])

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling

        # Optional: Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy()

class ChromaCompatibleEmbeddingFunction:
    def __init__(self, model: EmbeddingModel):
        self.model = model

    def __call__(self, input: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(input, str):
            input = [input]
        embeddings = self.model.encode(input)
        return embeddings.tolist() if embeddings.size else []

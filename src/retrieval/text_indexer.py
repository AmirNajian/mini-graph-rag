"""
TF-IDF text indexing and retrieval.
"""
from typing import List, Tuple, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class TextIndexer:
    """TF-IDF based text indexing and retrieval."""
    
    def __init__(self, config):
        """
        Initialize text indexer.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words="english",
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=1,  # Minimum document frequency
            max_df=0.95,  # Maximum document frequency
        )
        self.documents: List[str] = []
        self.document_ids: List[str] = []
        self.tfidf_matrix = None
        self._fitted = False
    
    def fit(self, documents: List[Dict[str, str]]):
        """
        Fit the TF-IDF vectorizer on documents.
        
        Args:
            documents: List of dictionaries with 'id' and 'text' keys
        """
        self.documents = [doc["text"] for doc in documents]
        self.document_ids = [doc["id"] for doc in documents]
        
        # Fit and transform
        self.tfidf_matrix = self.vectorizer.fit_transform(self.documents)
        self._fitted = True
    
    def query(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Query the index and return top-k most similar documents.
        
        Args:
            text: Query text
            top_k: Number of top results to return
            
        Returns:
            List of tuples (doc_id, score) sorted by score (descending)
        """
        if not self._fitted:
            raise ValueError("Indexer not fitted. Call fit() first.")
        
        # Transform query
        query_vector = self.vectorizer.transform([text])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Return results
        results = [
            (self.document_ids[idx], float(similarities[idx]))
            for idx in top_indices
            if similarities[idx] > 0.0
        ]
        
        return results
    
    def get_document(self, doc_id: str) -> str:
        """
        Get document text by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document text
        """
        try:
            idx = self.document_ids.index(doc_id)
            return self.documents[idx]
        except ValueError:
            raise ValueError(f"Document ID {doc_id} not found")
    
    def update(self, documents: List[Dict[str, str]]):
        """
        Update the index with new documents.
        
        Args:
            documents: List of new documents
        """
        # For simplicity, refit on all documents
        # In production, could use incremental updates
        existing_docs = [
            {"id": doc_id, "text": text}
            for doc_id, text in zip(self.document_ids, self.documents)
        ]
        all_docs = existing_docs + documents
        self.fit(all_docs)


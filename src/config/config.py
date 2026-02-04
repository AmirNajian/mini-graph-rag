"""
Configuration management for Mini GraphRAG.
"""
import os
from pathlib import Path
import yaml
from pydantic import BaseModel


class Config(BaseModel):
    """Configuration class for Mini GraphRAG service."""
    
    # Entity extraction
    entity_min_length: int = 2
    entity_max_length: int = 50
    entity_confidence_threshold: float = 0.5
    use_spacy_ner: bool = False
    
    # Entity resolution
    edit_distance_threshold: int = 2
    acronym_expansion_confidence: float = 0.8
    merge_confidence_threshold: float = 0.7
    
    # Knowledge graph
    cooccurrence_window: int = 10  # sentences
    graph_edge_weight_decay: float = 0.9
    max_graph_hops: int = 2
    
    # Retrieval
    tfidf_weight: float = 0.6
    graph_weight: float = 0.4
    retrieval_top_k: int = 10
    min_retrieval_score: float = 0.1
    
    # Answer synthesis
    max_answer_length: int = 500
    citation_context_chars: int = 50
    
    # Logging
    log_level: str = "INFO"
    
    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            entity_min_length=int(os.getenv("ENTITY_MIN_LENGTH", "2")),
            entity_max_length=int(os.getenv("ENTITY_MAX_LENGTH", "50")),
            entity_confidence_threshold=float(os.getenv("ENTITY_CONFIDENCE_THRESHOLD", "0.5")),
            use_spacy_ner=os.getenv("USE_SPACY_NER", "false").lower() == "true",
            edit_distance_threshold=int(os.getenv("EDIT_DISTANCE_THRESHOLD", "2")),
            acronym_expansion_confidence=float(os.getenv("ACRONYM_EXPANSION_CONFIDENCE", "0.8")),
            merge_confidence_threshold=float(os.getenv("MERGE_CONFIDENCE_THRESHOLD", "0.7")),
            cooccurrence_window=int(os.getenv("COOCCURRENCE_WINDOW", "10")),
            graph_edge_weight_decay=float(os.getenv("GRAPH_EDGE_WEIGHT_DECAY", "0.9")),
            max_graph_hops=int(os.getenv("MAX_GRAPH_HOPS", "2")),
            tfidf_weight=float(os.getenv("TFIDF_WEIGHT", "0.6")),
            graph_weight=float(os.getenv("GRAPH_WEIGHT", "0.4")),
            retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "10")),
            min_retrieval_score=float(os.getenv("MIN_RETRIEVAL_SCORE", "0.1")),
            max_answer_length=int(os.getenv("MAX_ANSWER_LENGTH", "500")),
            citation_context_chars=int(os.getenv("CITATION_CONTEXT_CHARS", "50")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
    
    @classmethod
    def default(cls) -> "Config":
        """Create default configuration."""
        config_path = Path("config.yaml")
        if config_path.exists():
            return cls.from_yaml(config_path)
        return cls.from_env()


"""
Hybrid retrieval combining lexical and graph-based methods.
"""
from typing import List, Dict, Optional
from dataclasses import dataclass

from typing import Optional as TypingOptional

from entity_extractor import EntityExtractor
from entity_resolver import EntityResolver
from text_indexer import TextIndexer
from knowledge_graph import KnowledgeGraph
from schema.retrieval import RetrievalResult

class Retriever:
    """Hybrid retriever combining TF-IDF and graph-based retrieval."""
    
    def __init__(self, config):
        """
        Initialize retriever.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.entity_extractor = EntityExtractor(config)
        self.entity_resolver: TypingOptional[EntityResolver] = None
        self.text_indexer: Optional[TextIndexer] = None
        self.knowledge_graph: Optional[KnowledgeGraph] = None
    
    def update_index(
        self,
        text_indexer: TextIndexer,
        knowledge_graph: KnowledgeGraph,
        entity_resolver: TypingOptional[EntityResolver] = None,
    ):
        """
        Update the retriever with new index and graph.
        
        Args:
            text_indexer: Text indexer instance
            knowledge_graph: Knowledge graph instance
            entity_resolver: Entity resolver instance (should match the one from ingestion)
        """
        self.text_indexer = text_indexer
        self.knowledge_graph = knowledge_graph
        if entity_resolver is not None:
            self.entity_resolver = entity_resolver
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """
        Retrieve documents using hybrid retrieval.
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of retrieval results sorted by score
        """
        if self.text_indexer is None or self.knowledge_graph is None:
            raise ValueError("Retriever not initialized. Call update_index() first.")
        
        # Stage 1: Lexical retrieval (TF-IDF)
        lexical_results = self.text_indexer.query(query, top_k=top_k * 2)
        lexical_scores = {doc_id: score for doc_id, score in lexical_results}
        
        # Stage 2: Entity linking (extract entities from query)
        query_entities = self.entity_extractor.extract_entities(query)
        query_entity_texts = [e.text for e in query_entities]
        
        # Stage 3: Graph expansion
        graph_scores = self._calculate_graph_scores(query_entity_texts, lexical_scores.keys())
        
        # Stage 4: Blend scores
        final_results = self._blend_scores(lexical_scores, graph_scores, query_entity_texts)
        
        # Sort and return top-k
        sorted_results = sorted(final_results, key=lambda r: r.score, reverse=True)
        return sorted_results[:top_k]
    
    def _calculate_graph_scores(
        self, query_entities: List[str], candidate_doc_ids: List[str]
    ) -> Dict[str, float]:
        """
        Calculate graph-based scores for candidate documents.
        
        Args:
            query_entities: List of entity texts from query
            candidate_doc_ids: List of candidate document IDs
            
        Returns:
            Dictionary mapping doc_id to graph score
        """
        graph_scores = {doc_id: 0.0 for doc_id in candidate_doc_ids}
        
        # Resolve query entities to canonical IDs
        if self.entity_resolver is None:
            # Fallback: create new resolver (won't have ingestion mappings)
            resolver = EntityResolver(self.config)
        else:
            resolver = self.entity_resolver
        
        query_entity_ids = []
        for entity_text in query_entities:
            canonical_id = resolver.resolve(entity_text)
            query_entity_ids.append(canonical_id)
        
        # For each query entity, expand graph neighborhood
        for entity_id in query_entity_ids:
            if entity_id in self.knowledge_graph.graph:
                # Get neighbors within max_hops
                neighbors = self.knowledge_graph.neighbors(
                    entity_id, hops=self.config.max_graph_hops
                )
                
                # Get documents for entity and its neighbors
                entity_docs = self.knowledge_graph.get_documents_for_entity(entity_id)
                for doc_id in entity_docs:
                    if doc_id in graph_scores:
                        # Base score for direct match
                        graph_scores[doc_id] += 1.0
                
                # Boost for documents connected through neighbors
                for neighbor_id, path_length in neighbors.items():
                    neighbor_docs = self.knowledge_graph.get_documents_for_entity(neighbor_id)
                    for doc_id in neighbor_docs:
                        if doc_id in graph_scores:
                            # Decay score by path length
                            boost = self.config.graph_edge_weight_decay ** path_length
                            graph_scores[doc_id] += boost
        
        # Normalize scores
        if graph_scores:
            max_score = max(graph_scores.values())
            if max_score > 0:
                graph_scores = {
                    doc_id: score / max_score for doc_id, score in graph_scores.items()
                }
        
        return graph_scores
    
    def _blend_scores(
        self,
        lexical_scores: Dict[str, float],
        graph_scores: Dict[str, float],
        query_entities: List[str],
    ) -> List[RetrievalResult]:
        """
        Blend lexical and graph scores.
        
        Args:
            lexical_scores: Dictionary of doc_id -> lexical score
            graph_scores: Dictionary of doc_id -> graph score
            query_entities: List of entity texts from query
            
        Returns:
            List of retrieval results
        """
        results = []
        
        # Get all document IDs
        all_doc_ids = set(lexical_scores.keys()) | set(graph_scores.keys())
        
        for doc_id in all_doc_ids:
            lexical_score = lexical_scores.get(doc_id, 0.0)
            graph_score = graph_scores.get(doc_id, 0.0)
            
            # Blend scores
            final_score = (
                self.config.tfidf_weight * lexical_score
                + self.config.graph_weight * graph_score
            )
            
            # Find matched entities for this document
            matched_entities = []
            if self.knowledge_graph:
                doc_entities = self.knowledge_graph.get_entities_for_document(doc_id)
                matched_entities = [
                    entity for entity in query_entities if entity in doc_entities
                ]
            
            if final_score >= self.config.min_retrieval_score:
                results.append(
                    RetrievalResult(
                        doc_id=doc_id,
                        score=final_score,
                        lexical_score=lexical_score,
                        graph_score=graph_score,
                        matched_entities=matched_entities,
                    )
                )
        
        return results


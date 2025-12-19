"""
Answer synthesis and citation extraction.
"""
from typing import List, Dict, Any
import re

from retriever import RetrievalResult
from knowledge_graph import KnowledgeGraph
from text_indexer import TextIndexer


class Synthesizer:
    """Synthesizes answers from retrieved documents."""
    
    def __init__(self, config):
        """
        Initialize synthesizer.
        
        Args:
            config: Configuration object
        """
        self.config = config
    
    def generate(
        self,
        retrieval_results: List[RetrievalResult],
        query: str,
        knowledge_graph: KnowledgeGraph,
        text_indexer: TextIndexer,
    ) -> Dict[str, Any]:
        """
        Generate answer with citations and graph trace.
        
        Args:
            retrieval_results: List of retrieval results
            query: Original query
            knowledge_graph: Knowledge graph instance
            text_indexer: Text indexer instance
            
        Returns:
            Dictionary with 'answer', 'citations', and 'graph_trace'
        """
        if not retrieval_results:
            return self._generate_not_found_response(query, knowledge_graph)
        
        # Extract citations
        citations = self.extract_citations(retrieval_results, query, text_indexer)
        
        # Generate answer text
        answer_text = self.generate_answer(retrieval_results, text_indexer, query)
        
        # Build graph trace
        graph_trace = self.build_trace(
            retrieval_results, query, knowledge_graph, text_indexer
        )
        
        return {
            "answer": answer_text,
            "citations": citations,
            "graph_trace": graph_trace,
        }
    
    def extract_citations(
        self,
        retrieval_results: List[RetrievalResult],
        query: str,
        text_indexer: TextIndexer,
    ) -> List[Dict[str, Any]]:
        """
        Extract citations from retrieved documents.
        
        Args:
            retrieval_results: List of retrieval results
            query: Query string
            text_indexer: Text indexer instance
            
        Returns:
            List of citation dictionaries
        """
        citations = []
        query_terms = set(query.lower().split())
        
        for result in retrieval_results[:3]:  # Top 3 documents
            try:
                doc_text = text_indexer.get_document(result.doc_id)
                
                # Find sentences containing query terms
                sentences = re.split(r"[.!?]+", doc_text)
                
                for sentence in sentences:
                    sentence_lower = sentence.lower()
                    # Count matching query terms
                    matches = sum(1 for term in query_terms if term in sentence_lower)
                    
                    if matches > 0:
                        # Find position in original document
                        start_char = doc_text.find(sentence)
                        end_char = start_char + len(sentence)
                        
                        # Add context
                        context_start = max(0, start_char - self.config.citation_context_chars)
                        context_end = min(
                            len(doc_text), end_char + self.config.citation_context_chars
                        )
                        fragment = doc_text[context_start:context_end].strip()
                        
                        citations.append(
                            {
                                "doc_id": result.doc_id,
                                "fragment": fragment,
                                "start_char": start_char,
                                "end_char": end_char,
                            }
                        )
                        
                        if len(citations) >= 2:  # At least 2 citations
                            break
                
                if len(citations) >= 2:
                    break
                    
            except ValueError:
                continue
        
        return citations
    
    def generate_answer(
        self,
        retrieval_results: List[RetrievalResult],
        text_indexer: TextIndexer,
        query: str,
    ) -> str:
        """
        Generate answer text from retrieved documents.
        
        Args:
            retrieval_results: List of retrieval results
            text_indexer: Text indexer instance
            query: Query string
            
        Returns:
            Answer text
        """
        if not retrieval_results:
            return "I could not find relevant information to answer this query."
        
        # Template-based synthesis (MVP)
        # Extract key sentences from top documents
        answer_parts = []
        
        for result in retrieval_results[:3]:  # Top 3 documents
            try:
                doc_text = text_indexer.get_document(result.doc_id)
                
                # Find first sentence that seems relevant
                sentences = re.split(r"[.!?]+", doc_text)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 20:  # Reasonable sentence length
                        answer_parts.append(sentence)
                        break
            except ValueError:
                continue
        
        if not answer_parts:
            return "I could not find relevant information to answer this query."
        
        # Combine sentences
        answer = " ".join(answer_parts[:2])  # Use first 2 sentences
        
        # Truncate if too long
        if len(answer) > self.config.max_answer_length:
            answer = answer[: self.config.max_answer_length] + "..."
        
        return answer
    
    def build_trace(
        self,
        retrieval_results: List[RetrievalResult],
        query: str,
        knowledge_graph: KnowledgeGraph,
        text_indexer: TextIndexer,
    ) -> Dict[str, Any]:
        """
        Build graph trace explaining the retrieval process.
        
        Args:
            retrieval_results: List of retrieval results
            query: Query string
            knowledge_graph: Knowledge graph instance
            text_indexer: Text indexer instance
            
        Returns:
            Graph trace dictionary
        """
        from entity_extractor import EntityExtractor
        
        entity_extractor = EntityExtractor(self.config)
        query_entities = [e.text for e in entity_extractor.extract_entities(query)]
        
        # Get expanded entities from graph
        expanded_entities = set()
        for result in retrieval_results:
            doc_entities = knowledge_graph.get_entities_for_document(result.doc_id)
            expanded_entities.update(doc_entities)
        
        # Get top documents
        top_docs = [result.doc_id for result in retrieval_results[:5]]
        
        # Build reasoning
        reasoning_parts = []
        if query_entities:
            reasoning_parts.append(f"Query entities: {', '.join(query_entities)}")
        
        if retrieval_results:
            top_result = retrieval_results[0]
            reasoning_parts.append(
                f"Top document '{top_result.doc_id}' scored {top_result.score:.3f} "
                f"(lexical: {top_result.lexical_score:.3f}, graph: {top_result.graph_score:.3f})"
            )
        
        reasoning = ". ".join(reasoning_parts)
        
        return {
            "query_entities": query_entities,
            "expanded_entities": list(expanded_entities)[:10],  # Limit to 10
            "top_docs": top_docs,
            "reasoning": reasoning,
        }
    
    def _generate_not_found_response(
        self, query: str, knowledge_graph: KnowledgeGraph
    ) -> Dict[str, Any]:
        """Generate response when no results found."""
        from entity_extractor import EntityExtractor
        
        entity_extractor = EntityExtractor(self.config)
        query_entities = [e.text for e in entity_extractor.extract_entities(query)]
        
        return {
            "answer": "I could not find relevant information in the corpus to answer this query.",
            "citations": [],
            "graph_trace": {
                "query_entities": query_entities,
                "expanded_entities": [],
                "top_docs": [],
                "reasoning": "No documents matched the query above the minimum score threshold.",
            },
        }


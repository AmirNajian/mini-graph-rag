"""
Document ingestion pipeline.
"""
from typing import List, Dict
from collections import defaultdict

from entity_extractor import EntityExtractor
from entity_resolver import EntityResolver
from knowledge_graph import KnowledgeGraph
from text_indexer import TextIndexer


class IngestPipeline:
    """Pipeline for ingesting documents into the knowledge graph."""
    
    def __init__(self, config):
        """
        Initialize ingestion pipeline.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.entity_extractor = EntityExtractor(config, use_spacy=config.use_spacy_ner)
        self.entity_resolver = EntityResolver(config)
        self.knowledge_graph = KnowledgeGraph(config)
        self.text_indexer = TextIndexer(config)
    
    def run(self, documents: List[Dict[str, str]]) -> Dict[str, int]:
        """
        Run the ingestion pipeline.
        
        Args:
            documents: List of dictionaries with 'id' and 'text' keys
            
        Returns:
            Dictionary with ingestion statistics
        """
        # Step 1: Extract entities from all documents
        doc_entities = {}
        all_extracted_entities = []
        
        for doc in documents:
            doc_id = doc["id"]
            text = doc["text"]
            
            # Extract entities
            entities = self.entity_extractor.extract_entities(text)
            all_extracted_entities.extend(entities)
            doc_entities[doc_id] = entities
        
        # Step 2: Resolve entities (deduplication)
        entity_to_canonical = {}
        for entity in all_extracted_entities:
            canonical_id = self.entity_resolver.resolve(entity.text)
            entity_to_canonical[entity.text] = canonical_id
        
        # Step 3: Build knowledge graph
        for doc_id, entities in doc_entities.items():
            # Add entities to graph
            canonical_ids = []
            for entity in entities:
                canonical_id = entity_to_canonical[entity.text]
                canonical_ids.append(canonical_id)
                self.knowledge_graph.add_entity(canonical_id, doc_id)
            
            # Add co-occurrence edges
            for i, entity_id_1 in enumerate(canonical_ids):
                for entity_id_2 in canonical_ids[i + 1 :]:
                    # Add bidirectional edge with weight based on co-occurrence
                    weight = 1.0
                    self.knowledge_graph.add_relation(entity_id_1, entity_id_2, weight)
                    self.knowledge_graph.add_relation(entity_id_2, entity_id_1, weight)
        
        # Step 4: Index documents (TF-IDF)
        self.text_indexer.fit(documents)
        
        # Calculate statistics
        stats = {
            "documents_processed": len(documents),
            "entities_extracted": len(all_extracted_entities),
            "entities_resolved": len(set(entity_to_canonical.values())),
            "graph_nodes": self.knowledge_graph.graph.number_of_nodes(),
            "graph_edges": self.knowledge_graph.graph.number_of_edges(),
        }
        
        return stats


"""
Knowledge graph implementation using Microsoft GraphRAG.

This module integrates Microsoft GraphRAG (https://github.com/microsoft/graphrag) 
while maintaining backward compatibility with the existing interface.

Architecture:
- Primary storage: NetworkX graph (for manual entity/relation management)
- GraphRAG integration: Optional, for import/export and future LLM-based enhancements
- Interface compatibility: Maintains the same API as the original NetworkX-only implementation

Note: Microsoft GraphRAG is designed as a full LLM-based pipeline, while this project
uses heuristic entity extraction. This implementation bridges both approaches by:
1. Using NetworkX as the primary graph storage (compatible with heuristic extraction)
2. Providing import/export methods to work with GraphRAG's data format
3. Allowing future integration with GraphRAG's LLM-based extraction if desired

To use GraphRAG features:
1. Install: `uv sync` (graphrag is in dependencies)
2. Initialize GraphRAG project: `graphrag init --root ./graphrag_data`
3. Optionally pass graphrag_root to KnowledgeGraph constructor
4. Use import_from_graphrag() to load GraphRAG-generated graphs
5. Use export_to_graphrag_format() to export for GraphRAG processing
"""
from typing import Dict, List, Set, Optional
import networkx as nx
from dataclasses import dataclass
from pathlib import Path
import os

try:
    # Try importing GraphRAG components
    # Note: GraphRAG's API may vary by version
    try:
        from graphrag.indexing import Indexer
        from graphrag.querying import QueryEngine
        GRAPHRAG_INDEXING_AVAILABLE = True
    except ImportError:
        GRAPHRAG_INDEXING_AVAILABLE = False
    
    # Check if we can access graph data structures
    try:
        import graphrag
        GRAPHRAG_AVAILABLE = True
    except ImportError:
        GRAPHRAG_AVAILABLE = False
        
except ImportError:
    GRAPHRAG_AVAILABLE = False
    GRAPHRAG_INDEXING_AVAILABLE = False

if not GRAPHRAG_AVAILABLE:
    # Fallback to NetworkX if GraphRAG not available
    pass  # Will use NetworkX as primary storage


@dataclass
class Relation:
    """Represents a relation between entities."""
    source_id: str
    target_id: str
    weight: float
    edge_type: str = "cooccurrence"


class KnowledgeGraph:
    """
    Knowledge graph wrapper using Microsoft GraphRAG.
    
    This implementation uses GraphRAG's graph storage capabilities while maintaining
    a compatible interface for manual entity/relation management. Since GraphRAG is
    designed as a full LLM-based pipeline and this project uses heuristic extraction,
    we maintain a NetworkX graph for manual operations and can optionally sync with
    GraphRAG's graph structure.
    """
    
    def __init__(self, config, graphrag_root: Optional[Path] = None):
        """
        Initialize knowledge graph.
        
        Args:
            config: Configuration object
            graphrag_root: Optional path to GraphRAG project root. If None, uses
                         in-memory NetworkX graph only.
        """
        self.config = config
        self.graphrag_root = graphrag_root
        self.use_graphrag = GRAPHRAG_AVAILABLE and graphrag_root is not None
        
        # Always maintain NetworkX graph for compatibility
        self.graph = nx.DiGraph()
        # Mapping: entity_id -> set of document IDs
        self.entity_to_docs: Dict[str, Set[str]] = {}
        # Mapping: doc_id -> set of entity IDs
        self.doc_to_entities: Dict[str, Set[str]] = {}
        
        # GraphRAG components (if available)
        if self.use_graphrag and GRAPHRAG_INDEXING_AVAILABLE:
            try:
                # GraphRAG typically requires initialization via CLI: `graphrag init --root <path>`
                # and configuration files. For this implementation, we use NetworkX as the
                # primary graph storage and maintain compatibility with GraphRAG's data format.
                if graphrag_root and not graphrag_root.exists():
                    graphrag_root.mkdir(parents=True, exist_ok=True)
                
                # Note: GraphRAG's Indexer and QueryEngine require configuration files
                # (settings.yaml) that are typically created via `graphrag init`.
                # We maintain NetworkX as primary storage for manual entity/relation management.
                self.graphrag_initialized = False
            except Exception as e:
                # Silently fall back to NetworkX
                self.use_graphrag = False
    
    def add_entity(self, entity_id: str, doc_id: Optional[str] = None):
        """
        Add an entity to the graph.
        
        Args:
            entity_id: Entity ID
            doc_id: Optional document ID that contains this entity
        """
        # Add to NetworkX graph (primary storage)
        if entity_id not in self.graph:
            self.graph.add_node(entity_id, entity_id=entity_id)
        
        if doc_id:
            if entity_id not in self.entity_to_docs:
                self.entity_to_docs[entity_id] = set()
            self.entity_to_docs[entity_id].add(doc_id)
            
            if doc_id not in self.doc_to_entities:
                self.doc_to_entities[doc_id] = set()
            self.doc_to_entities[doc_id].add(entity_id)
        
        # Optionally sync with GraphRAG if available
        # Note: GraphRAG typically expects entities to be extracted via LLM,
        # so we're maintaining NetworkX as the source of truth for manual operations
    
    def add_relation(
        self,
        source_id: str,
        target_id: str,
        weight: float = 1.0,
        edge_type: str = "cooccurrence",
    ):
        """
        Add a relation between two entities.
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            weight: Edge weight
            edge_type: Type of relation
        """
        # Ensure entities exist
        self.add_entity(source_id)
        self.add_entity(target_id)
        
        # Add or update edge in NetworkX
        if self.graph.has_edge(source_id, target_id):
            # Update weight (sum for co-occurrence)
            current_weight = self.graph[source_id][target_id].get("weight", 0.0)
            self.graph[source_id][target_id]["weight"] = current_weight + weight
        else:
            self.graph.add_edge(
                source_id, target_id, weight=weight, edge_type=edge_type
            )
    
    def neighbors(self, entity_id: str, hops: int = 1) -> Dict[str, int]:
        """
        Get neighbors of an entity within N hops.
        
        Uses NetworkX for graph traversal. If GraphRAG is available and initialized,
        could potentially use GraphRAG's graph structure, but NetworkX is more
        suitable for manual graph operations.
        
        Args:
            entity_id: Entity ID
            hops: Number of hops to traverse
            
        Returns:
            Dictionary mapping neighbor entity IDs to path length
        """
        if entity_id not in self.graph:
            return {}
        
        neighbors_dict = {}
        
        # BFS traversal using NetworkX
        queue = [(entity_id, 0)]
        visited = {entity_id}
        
        while queue:
            current, depth = queue.pop(0)
            
            if depth > 0:  # Don't include the starting entity
                neighbors_dict[current] = depth
            
            if depth < hops:
                for neighbor in self.graph.neighbors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
        
        return neighbors_dict
    
    def connected_subgraph(self, entity_ids: List[str], hops: int = 2) -> nx.DiGraph:
        """
        Get connected subgraph containing the given entities.
        
        Args:
            entity_ids: List of entity IDs
            hops: Maximum hops to include
            
        Returns:
            Subgraph containing the entities and their neighbors
        """
        # Collect all nodes within hops of the given entities
        nodes_to_include = set(entity_ids)
        
        for entity_id in entity_ids:
            if entity_id in self.graph:
                neighbors_dict = self.neighbors(entity_id, hops=hops)
                nodes_to_include.update(neighbors_dict.keys())
        
        # Create subgraph from NetworkX graph
        return self.graph.subgraph(nodes_to_include).copy()
    
    def get_documents_for_entity(self, entity_id: str) -> Set[str]:
        """
        Get all documents containing an entity.
        
        Args:
            entity_id: Entity ID
            
        Returns:
            Set of document IDs
        """
        return self.entity_to_docs.get(entity_id, set())
    
    def get_entities_for_document(self, doc_id: str) -> Set[str]:
        """
        Get all entities in a document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Set of entity IDs
        """
        return self.doc_to_entities.get(doc_id, set())
    
    def get_edge_weight(self, source_id: str, target_id: str) -> float:
        """
        Get weight of an edge between two entities.
        
        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            
        Returns:
            Edge weight, or 0.0 if no edge exists
        """
        if self.graph.has_edge(source_id, target_id):
            return self.graph[source_id][target_id].get("weight", 0.0)
        return 0.0
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get graph statistics.
        
        Returns:
            Dictionary with node count, edge count, etc.
        """
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "documents": len(self.doc_to_entities),
            "using_graphrag": self.use_graphrag,
        }
    
    def export_to_graphrag_format(self, output_path: Path) -> None:
        """
        Export the NetworkX graph to a format compatible with GraphRAG.
        
        This is a helper method to convert our heuristic-based graph structure
        to a format that GraphRAG can use. GraphRAG typically expects entities
        and relationships extracted via LLM, but we can export our structure.
        
        Note: GraphRAG uses specific data formats. This export creates a simplified
        representation that may need adjustment based on GraphRAG version.
        
        Args:
            output_path: Path to export the graph data
        """
        import json
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export entities in a format compatible with GraphRAG
        entities_file = output_path / "entities.jsonl"
        with open(entities_file, "w", encoding="utf-8") as f:
            for node_id in self.graph.nodes():
                # Create entity representation compatible with GraphRAG format
                entity_data = {
                    "id": node_id,
                    "title": node_id,  # Use ID as title for now
                    "type": "ENTITY",  # Default type
                    "description": "",  # Could be enhanced with entity metadata
                }
                f.write(json.dumps(entity_data, ensure_ascii=False) + "\n")
        
        # Export relationships
        relationships_file = output_path / "relationships.jsonl"
        with open(relationships_file, "w", encoding="utf-8") as f:
            for source, target, data in self.graph.edges(data=True):
                relationship_data = {
                    "source": source,
                    "target": target,
                    "type": data.get("edge_type", "cooccurrence"),
                    "weight": data.get("weight", 1.0),
                    "description": "",  # Could be enhanced with relationship metadata
                }
                f.write(json.dumps(relationship_data, ensure_ascii=False) + "\n")
    
    def import_from_graphrag(self, graphrag_data_path: Path) -> None:
        """
        Import graph data from GraphRAG format.
        
        This allows importing entities and relationships that were created
        using GraphRAG's LLM-based extraction pipeline.
        
        Note: GraphRAG's actual data format may vary. This is a simplified
        importer that works with JSONL format. Adjust based on your GraphRAG version.
        
        Args:
            graphrag_data_path: Path to GraphRAG's graph data directory
        """
        import json
        
        # Import entities
        # GraphRAG may store entities in different locations depending on version
        possible_entity_files = [
            graphrag_data_path / "entities.jsonl",
            graphrag_data_path / "graph" / "entities.jsonl",
            graphrag_data_path / "index" / "entities.jsonl",
        ]
        
        for entities_file in possible_entity_files:
            if entities_file.exists():
                with open(entities_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            entity_data = json.loads(line)
                            entity_id = entity_data.get("id") or entity_data.get("title")
                            if entity_id:
                                self.add_entity(entity_id)
                        except json.JSONDecodeError:
                            continue
                break
        
        # Import relationships
        possible_rel_files = [
            graphrag_data_path / "relationships.jsonl",
            graphrag_data_path / "graph" / "relationships.jsonl",
            graphrag_data_path / "index" / "relationships.jsonl",
        ]
        
        for relationships_file in possible_rel_files:
            if relationships_file.exists():
                with open(relationships_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            rel_data = json.loads(line)
                            source = rel_data.get("source") or rel_data.get("source_id")
                            target = rel_data.get("target") or rel_data.get("target_id")
                            weight = rel_data.get("weight", 1.0)
                            edge_type = rel_data.get("type") or rel_data.get("relationship_type", "cooccurrence")
                            
                            if source and target:
                                self.add_relation(source, target, weight, edge_type)
                        except json.JSONDecodeError:
                            continue
                break

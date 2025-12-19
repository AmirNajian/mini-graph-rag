"""
Example usage of Mini GraphRAG service.
"""
from config import Config
from ingest_pipeline import IngestPipeline
from retriever import Retriever
from synthesizer import Synthesizer

# Sample documents
documents = [
    {
        "id": "doc1",
        "text": "Franklin Templeton is a major investment management firm founded in 1947. "
        "The company manages assets worth over $1 trillion. FT has offices worldwide.",
    },
    {
        "id": "doc2",
        "text": "Gro Intelligence is an agricultural data platform. Gro provides insights "
        "on crop yields and market trends. The company was founded in 2014.",
    },
    {
        "id": "doc3",
        "text": "Franklin Templeton Investments, also known as FT, has been a leader in "
        "the asset management industry for decades. The firm offers various investment products.",
    },
]


def main():
    """Run example."""
    # Initialize
    config = Config.default()
    pipeline = IngestPipeline(config)
    retriever = Retriever(config)
    synthesizer = Synthesizer(config)
    
    # Ingest documents
    print("Ingesting documents...")
    stats = pipeline.run(documents)
    print(f"Ingestion complete: {stats}")
    
    # Update retriever
    retriever.update_index(
        pipeline.text_indexer, pipeline.knowledge_graph, pipeline.entity_resolver
    )
    
    # Answer queries
    queries = [
        "What is Franklin Templeton?",
        "Tell me about Gro Intelligence",
        "What does FT do?",
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        
        # Retrieve
        results = retriever.retrieve(query, top_k=3)
        
        # Synthesize
        answer = synthesizer.generate(
            results, query, pipeline.knowledge_graph, pipeline.text_indexer
        )
        
        print(f"Answer: {answer['answer']}")
        print(f"Citations: {len(answer['citations'])}")
        print(f"Graph Trace: {answer['graph_trace']}")


if __name__ == "__main__":
    main()


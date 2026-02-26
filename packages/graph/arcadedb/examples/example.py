"""Example usage of the ArcadeDB community adapter for Cognee."""

import asyncio
import os
import pathlib
import pprint

import cognee
from cognee.infrastructure.databases.graph import get_graph_engine

from cognee_community_graph_adapter_arcadedb import register


async def main():
    # Configure cognee to use ArcadeDB
    cognee.config.set_graph_database_provider("arcadedb")
    register()

    # Set up your ArcadeDB connection
    # ArcadeDB exposes the Bolt protocol — default port is 7687
    # Make sure ArcadeDB is running with Bolt enabled
    cognee.config.set_graph_db_config(
        {
            "graph_database_url": "bolt://localhost:7687",
            "graph_database_username": "root",
            "graph_database_password": "arcadedb",
        }
    )

    # Optional: Set custom data and system directories
    system_path = pathlib.Path(__file__).parent
    cognee.config.system_root_directory(os.path.join(system_path, ".cognee_system"))
    cognee.config.data_root_directory(os.path.join(system_path, ".data_storage"))

    # Sample data to add to the knowledge graph
    sample_data = [
        "Artificial intelligence is a branch of computer science that aims to"
        " create intelligent machines.",
        "Machine learning is a subset of AI that focuses on algorithms that can"
        " learn from data.",
        "Deep learning is a subset of machine learning that uses neural networks"
        " with many layers.",
        "Natural language processing enables computers to understand and process"
        " human language.",
        "Computer vision allows machines to interpret and make decisions"
        " based on visual information.",
    ]

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    print("Adding data to Cognee...")
    await cognee.add(sample_data, "ai_knowledge")

    print("Processing data with Cognee...")
    await cognee.cognify(["ai_knowledge"])

    print("Searching for insights...")
    search_results = await cognee.search(
        query_type=cognee.SearchType.GRAPH_COMPLETION,
        query_text="artificial intelligence",
    )

    print(f"Found {len(search_results)} insights:")
    for i, result in enumerate(search_results, 1):
        print(f"{i}. {result}")

    print("\nYou can get the graph data directly, or visualize it in an HTML file like below:")

    # Get graph data directly
    graph_engine = await get_graph_engine()
    graph_data = await graph_engine.get_graph_data()

    print("\nDirect graph data:")
    pprint.pprint(graph_data)

    # Or visualize it in HTML
    print("\nVisualizing the graph...")
    await cognee.visualize_graph(system_path / "graph.html")
    print(f"Graph visualization saved to {system_path / 'graph.html'}")


if __name__ == "__main__":
    asyncio.run(main())

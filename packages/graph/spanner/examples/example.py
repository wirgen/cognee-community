"""Example: use Cognee with Google Cloud Spanner as the graph backend."""

import asyncio
import os
import pathlib
import pprint

import cognee
from cognee.infrastructure.databases.graph import get_graph_engine

from cognee_community_graph_adapter_spanner import register


async def main() -> None:
    # Register the Spanner Graph adapter
    register()
    cognee.config.set_graph_database_provider("spanner")

    # Configure Spanner: use URL or separate project_id, instance_id, database_id
    graph_url = os.getenv(
        "GRAPH_DATABASE_URL",
        os.getenv("SPANNER_DATABASE", "project-id/instance-id/database-id"),
    )
    cognee.config.set_graph_db_config({"graph_database_url": graph_url})

    system_path = pathlib.Path(__file__).parent
    cognee.config.system_root_directory(os.path.join(system_path, ".cognee_system"))
    cognee.config.data_root_directory(os.path.join(system_path, ".data_storage"))

    sample_data = [
        "Artificial intelligence is a branch of computer science that aims to "
        "create intelligent machines.",
        "Machine learning is a subset of AI that focuses on algorithms that "
        "can learn from data.",
        "Deep learning uses neural networks with many layers.",
    ]

    try:
        print("Adding data to Cognee (Spanner Graph backend)...")
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
            print(f"  {i}. {result}")

        graph_engine = await get_graph_engine()
        nodes, edges = await graph_engine.get_graph_data()
        print(f"\nGraph: {len(nodes)} nodes, {len(edges)} edges")
        pprint.pprint((nodes[:3], edges[:3]))
    except Exception as e:
        print(f"Error: {e}")
        print(
            "Ensure Spanner database exists with Cognee schema and "
            "GRAPH_DATABASE_URL (or SPANNER_DATABASE) is set to "
            "project_id/instance_id/database_id."
        )
        raise


if __name__ == "__main__":
    asyncio.run(main())

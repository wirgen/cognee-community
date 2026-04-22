import os
import pathlib

from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).parent.parent / ".env")

import cognee
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.search.operations import get_history
from cognee.modules.search.types import SearchType
from cognee.modules.users.methods import get_default_user
from cognee.shared.logging_utils import get_logger

from cognee_community_vector_adapter_moss import register  # noqa: F401

logger = get_logger()

TEST_DATA_DIR = pathlib.Path(__file__).parent.parent.parent.parent / "test_data"


async def main():
    cognee.config.set_vector_db_config(
        {
            "vector_db_provider": "moss",
            "vector_db_key": os.getenv("MOSS_PROJECT_KEY", ""),
            "vector_db_name": os.getenv("MOSS_PROJECT_ID", ""),
            "vector_dataset_database_handler": "moss",
        }
    )

    data_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_moss")
        ).resolve()
    )
    cognee.config.data_root_directory(data_directory_path)
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_moss")
        ).resolve()
    )
    cognee.config.system_root_directory(cognee_directory_path)

    # Clean slate
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Add both files with node sets, cognify once
    nlp_path = str(TEST_DATA_DIR / "Natural_language_processing.txt")
    quantum_path = str(TEST_DATA_DIR / "Quantum_computers.txt")

    await cognee.add([nlp_path], "natural_language", node_set=["NLP"])
    await cognee.add([quantum_path], "quantum", node_set=["Quantum", "Computers"])
    await cognee.cognify(["quantum", "natural_language"])

    # Test: document retrieval
    from cognee.modules.users.permissions.methods import get_document_ids_for_user

    user = await get_default_user()
    document_ids = await get_document_ids_for_user(user.id, ["natural_language"])
    assert len(document_ids) == 1, f"Expected 1 doc in natural_language, got {len(document_ids)}"

    document_ids = await get_document_ids_for_user(user.id)
    assert len(document_ids) == 2, f"Expected 2 total docs, got {len(document_ids)}"
    print("PASSED: document_retrieval")

    # Test: vector search with query_vector
    vector_engine = get_vector_engine()
    query_text = "Tell me about NLP"
    query_vector = (await vector_engine.embedding_engine.embed_text([query_text]))[0]

    result = await vector_engine.search(
        collection_name="Entity_name", query_vector=query_vector, limit=5
    )
    assert len(result) > 0, "Search returned no results"
    assert all(hasattr(r, "id") and hasattr(r, "score") for r in result)
    print("PASSED: vector_search")

    # Test: vector search with query_text
    result = await vector_engine.search(
        collection_name="Entity_name",
        query_text="Natural language processing",
        limit=5,
        include_payload=True,
    )
    assert len(result) > 0, "Text search returned no results"
    assert result[0].payload is not None, "Payload should be included"
    print("PASSED: text_search")

    # Test: graph completion search
    random_node = (
        await vector_engine.search(
            collection_name="Entity_name", query_text="Quantum computer", include_payload=True
        )
    )[0]
    random_node_name = random_node.payload["text"]

    search_results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION, query_text=random_node_name
    )
    assert len(search_results) != 0, "Graph completion search results are empty"
    print("PASSED: graph_completion_search")

    # Test: chunks search with dataset filter
    search_results = await cognee.search(
        query_type=SearchType.CHUNKS, query_text=random_node_name, datasets=["quantum"]
    )
    assert len(search_results) != 0, "Chunks search results are empty"
    print("PASSED: chunks_search")

    # Test: summaries search
    search_results = await cognee.search(
        query_type=SearchType.SUMMARIES, query_text=random_node_name
    )
    assert len(search_results) != 0, "Summaries search results are empty"
    print("PASSED: summaries_search")

    # Test: search history
    user = await get_default_user()
    history = await get_history(user.id)
    assert len(history) > 0, "Search history is empty"
    print("PASSED: search_history")

    # Test: nodeset filtering (OR) — NLP or Quantum should return results
    result = await vector_engine.search(
        collection_name="DocumentChunk_text",
        query_vector=query_vector,
        include_payload=True,
        limit=10,
        node_name=["NLP", "Quantum"],
        node_name_filter_operator="OR",
    )
    assert len(result) > 0, "OR filter search returned no results"
    print("PASSED: nodeset_filter_or")

    # Test: nodeset filtering (AND) — NLP AND Quantum never overlap, expect empty
    result = await vector_engine.search(
        collection_name="DocumentChunk_text",
        query_vector=query_vector,
        include_payload=True,
        limit=10,
        node_name=["NLP", "Quantum"],
        node_name_filter_operator="AND",
    )
    assert len(result) == 0, f"AND filter for non-overlapping nodesets should return empty, got {len(result)}"
    print("PASSED: nodeset_filter_and")

    # Test: prune cleans up
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    print("PASSED: prune")

    print("\nAll tests passed!")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

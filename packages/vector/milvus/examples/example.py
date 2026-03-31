import asyncio
import os
import pathlib
from os import path

# Please provide an OpenAI API Key
os.environ.setdefault("LLM_API_KEY", "")


async def main() -> None:
    # NOTE: Importing the register module we let cognee know it can use the Milvus vector adapter
    # NOTE: The "noqa: F401" mark is to make sure the linter doesn't flag this as an unused import
    from cognee import SearchType, add, cognify, config, prune, search
    from cognee_community_vector_adapter_milvus import register  # noqa: F401

    system_path = pathlib.Path(__file__).parent
    config.system_root_directory(path.join(system_path, ".cognee_system"))
    config.data_root_directory(path.join(system_path, ".cognee_data"))

    # Please provide your Milvus instance url or local path
    config.set_vector_db_config(
        {
            "vector_db_provider": "milvus",
            "vector_db_url": os.getenv("MILVUS_URL", "http://localhost:19530"),
            "vector_db_key": os.getenv("MILVUS_KEY", ""),
        }
    )

    await prune.prune_data()
    await prune.prune_system(metadata=True)

    await add("""
    Natural language processing (NLP) is an interdisciplinary
    subfield of computer science and information retrieval.
    """)

    await add("""
    Sandwhiches are best served toasted with cheese, ham, mayo,
    lettuce, mustard, and salt & pepper.
    """)

    await cognify()

    query_text = "Tell me about NLP"

    search_results = await search(query_type=SearchType.GRAPH_COMPLETION, query_text=query_text)

    for result_text in search_results:
        print("\nSearch result: \n" + result_text)


if __name__ == "__main__":
    asyncio.run(main())

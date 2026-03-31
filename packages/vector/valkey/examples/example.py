import asyncio
import os
import pathlib
from os import path

from cognee import SearchType, add, cognify, config, prune, search
from cognee_community_vector_adapter_valkey import register  # noqa: F401

# Please provide an OpenAI API Key
# os.environ.setdefault("LLM_API_KEY", "your-api-key")


async def main():
    system_path = pathlib.Path(__file__).parent
    config.system_root_directory(path.join(system_path, ".cognee-system"))
    config.data_root_directory(path.join(system_path, ".cognee-data"))

    # Please provide your Valkey instance url
    config.set_vector_db_config(
        {
            "vector_db_provider": "valkey",
            "vector_db_url": os.getenv("VECTOR_DB_URL", "valkey://localhost:6379"),
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

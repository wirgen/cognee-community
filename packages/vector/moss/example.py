import asyncio
import os
import pathlib

from dotenv import load_dotenv

load_dotenv(pathlib.Path(__file__).parent / ".env")

from cognee_community_vector_adapter_moss import register  # noqa: F401


async def main():
    from cognee import SearchType, add, cognify, config, prune, search

    config.set_vector_db_config(
        {
            "vector_db_provider": "moss",
            "vector_db_key": os.getenv("MOSS_PROJECT_KEY", ""),
            "vector_db_name": os.getenv("MOSS_PROJECT_ID", ""),
            "vector_dataset_database_handler": "moss",
        }
    )

    await prune.prune_data()
    await prune.prune_system(metadata=True)

    text = """
    Natural language processing (NLP) is an interdisciplinary
    subfield of computer science and information retrieval.
    """

    await add(text)

    await cognify()

    query_text = "Tell me about NLP"

    search_results = await search(query_type=SearchType.GRAPH_COMPLETION, query_text=query_text)

    for result_text in search_results:
        print(result_text)


if __name__ == "__main__":
    asyncio.run(main())

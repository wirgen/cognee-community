import asyncio
import os
import pathlib

import cognee
from cognee import SearchType
from cognee_community_pipeline_codify.code_graph_pipeline import run_code_graph_pipeline
from cognee_community_retriever_code import register  # noqa: F401
from cognee_community_retriever_code.code_retriever import CodeSearchType


async def main():
    # Disable permissions feature for this example
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"
    repo_path = str(pathlib.Path(__file__).parent.parent.parent.parent.resolve())
    include_docs = False

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    run_status = False
    async for run_status in run_code_graph_pipeline(repo_path=repo_path, include_docs=include_docs):
        run_status = run_status

    # Test CODE search
    search_results = await cognee.search(
        query_type=SearchType[CodeSearchType.name],
        query_text="Find dependencies and relationships between components",
    )
    assert len(search_results) != 0, "The search results list is empty."
    print("\n\nSearch results are:\n")
    for result in search_results:
        print(f"{result}\n")

    return run_status


if __name__ == "__main__":
    asyncio.run(main())

from typing import Any, List, Optional
from uuid import NAMESPACE_OID, UUID, uuid5

from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.infrastructure.llm import get_max_chunk_tokens
from cognee.modules.cognify.config import get_cognify_config
from cognee.modules.data.methods import create_authorized_dataset
from cognee.modules.observability.get_observe import get_observe
from cognee.modules.pipelines import run_tasks
from cognee.modules.pipelines.tasks.task import Task
from cognee.modules.users.methods import get_default_user
from cognee.modules.users.models import User
from cognee.shared.data_models import KnowledgeGraph
from cognee.shared.logging_utils import get_logger
from cognee.tasks.documents import classify_documents, extract_chunks_from_documents
from cognee.tasks.graph import extract_graph_from_data
from cognee.tasks.ingestion import ingest_data
from cognee.tasks.storage import add_data_points
from cognee.tasks.summarization import summarize_text
from cognee_community_tasks_codify.get_non_code_files import get_non_py_files
from cognee_community_tasks_codify.get_repo_file_dependencies import get_repo_file_dependencies

observe = get_observe()

logger = get_logger("code_graph_pipeline")


class LightweightData(DataPoint):
    """Lightweight DataPoint model for data ingestion only."""

    id: UUID
    repo_path: str


def build_lightweight_data_object(data_list):
    return [
        LightweightData(id=uuid5(NAMESPACE_OID, str(data)), repo_path=data) for data in data_list
    ]


def ingest_repositories(data_list: List[Any]):
    return [data.repo_path for data in data_list]


@observe
async def run_code_graph_pipeline(
    repo_path,
    include_docs=False,
    excluded_paths: Optional[list[str]] = None,
    supported_languages: Optional[list[str]] = None,
    user: User = None,
    detailed_extraction: bool = True,
):
    cognee_config = get_cognify_config()
    if not user:
        from cognee.low_level import setup

        await setup()

        user = await get_default_user()

    tasks = [
        Task(ingest_repositories),
        Task(
            get_repo_file_dependencies,
            detailed_extraction=detailed_extraction,
            supported_languages=supported_languages,
            excluded_paths=excluded_paths,
        ),
        # This task takes a long time to complete
        # Task(summarize_code, task_config={"batch_size": 500}),
        Task(add_data_points, task_config={"batch_size": 30}),
    ]

    if include_docs:
        # These tasks take a long time to complete
        non_code_tasks = [
            Task(get_non_py_files, task_config={"batch_size": 50}),
            Task(ingest_data, dataset_name="repo_docs", user=user),
            Task(classify_documents),
            Task(extract_chunks_from_documents, max_chunk_size=get_max_chunk_tokens()),
            Task(
                extract_graph_from_data,
                graph_model=KnowledgeGraph,
                task_config={"batch_size": 50},
            ),
            Task(
                summarize_text,
                summarization_model=cognee_config.summarization_model,
                task_config={"batch_size": 50},
            ),
        ]

    dataset_name = "codebase"

    # Save dataset to database
    dataset = await create_authorized_dataset(dataset_name, user)

    if include_docs:
        non_code_pipeline_run = run_tasks(
            non_code_tasks, dataset.id, repo_path, user, "cognify_pipeline"
        )
        async for run_status in non_code_pipeline_run:
            yield run_status

    async for run_status in run_tasks(
        tasks,
        dataset.id,
        build_lightweight_data_object(repo_path if isinstance(repo_path, list) else [repo_path]),
        user,
        "cognify_code_pipeline",
        incremental_loading=False,
    ):
        yield run_status

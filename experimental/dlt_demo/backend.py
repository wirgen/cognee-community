from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, inspect


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
GRAPH_FILE_PATH = BASE_DIR / "graph_visualization.html"

# Ensure local cognee package is importable when run from this demo folder.
REPO_ROOT = BASE_DIR.parents[2]
COGNEE_PACKAGE_ROOT = REPO_ROOT / "cognee"
if str(COGNEE_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(COGNEE_PACKAGE_ROOT))

import cognee  # noqa: E402

# DB_SCHEMES = {"postgresql", "postgres", "mysql", "sqlite", "mssql", "oracle"}


class UploadRequest(BaseModel):
    connection_string: str = Field(..., min_length=1)
    dataset_name: str = Field(default="main_dataset", min_length=1)
    ontology_file_path: Optional[str] = None
    query: Optional[str] = None


class SchemaDecisionRequest(BaseModel):
    approved: bool


class SearchRequest(BaseModel):
    question: str = Field(..., min_length=1)
    dataset_name: Optional[str] = None


class WorkflowState:
    def __init__(self) -> None:
        self.pending_dataset_name: Optional[str] = None
        self.pending_ontology_file_path: Optional[str] = None
        self.pending_schema: Optional[dict[str, Any]] = None


state = WorkflowState()

app = FastAPI(title="Cognee DLT Workflow Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_schema(connection_string: str) -> dict[str, Any]:
    """Extract a simple relational schema preview from SQLite/Postgres connection strings."""
    try:
        engine = create_engine(connection_string)
        inspector = inspect(engine)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not connect to database: {exc}") from exc

    try:
        tables: list[dict[str, Any]] = []
        for table_name in inspector.get_table_names():
            columns = inspector.get_columns(table_name)
            foreign_keys = inspector.get_foreign_keys(table_name)
            tables.append(
                {
                    "table_name": table_name,
                    "columns": [
                        {
                            "name": col.get("name"),
                            "type": str(col.get("type")),
                            "nullable": col.get("nullable"),
                        }
                        for col in columns
                    ],
                    "foreign_keys": [
                        {
                            "column": (fk.get("constrained_columns") or [None])[0],
                            "ref_table": fk.get("referred_table"),
                            "ref_column": (fk.get("referred_columns") or [None])[0],
                        }
                        for fk in foreign_keys
                    ],
                }
            )
        return {"tables": tables}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Schema extraction failed: {exc}") from exc
    finally:
        engine.dispose()


# def looks_like_db_connection_string(value: str) -> bool:
#     if "://" not in value:
#         return False
#     scheme = value.split("://", 1)[0].lower()
#     base_scheme = scheme.split("+", 1)[0]
#     return base_scheme in DB_SCHEMES


# def to_dlt_source(connection_string: str, query: Optional[str]):
#     if importlib.util.find_spec("dlt") is None:
#         raise HTTPException(
#             status_code=400,
#             detail=(
#                 "DB ingestion requires dlt. Install it in this environment: "
#                 "`pip install 'dlt[sqlalchemy]>=1.9.0,<2'`"
#             ),
#         )
#
#     from cognee.tasks.ingestion.create_dlt_source import create_dlt_source_from_connection_string
#
#     return create_dlt_source_from_connection_string(connection_string, query=query)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/upload-and-extract")
async def upload_and_extract(request: UploadRequest) -> dict[str, Any]:
    # data_for_add: Any = request.connection_string
    # if looks_like_db_connection_string(request.connection_string):
    #     data_for_add = to_dlt_source(request.connection_string, request.query)

    try:
        add_result = await cognee.add(
            request.connection_string,
            dataset_name=request.dataset_name,
            query=request.query,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"cognee.add failed: {exc}") from exc

    schema = extract_schema(request.connection_string)

    state.pending_dataset_name = request.dataset_name
    state.pending_ontology_file_path = (
        request.ontology_file_path.strip() if request.ontology_file_path else None
    )
    state.pending_schema = schema

    return {
        "message": "Data uploaded with cognee.add(). Schema extracted. Please approve schema.",
        "schema": schema,
        "dataset_name": request.dataset_name,
        "add_result": str(add_result),
    }


@app.post("/api/schema-decision")
async def schema_decision(request: SchemaDecisionRequest) -> dict[str, Any]:
    if state.pending_dataset_name is None:
        raise HTTPException(status_code=400, detail="No pending workflow. Upload data first.")

    if not request.approved:
        return {"message": "Schema rejected. Flow stopped (no cognify run)."}

    kwargs: dict[str, Any] = {}
    if state.pending_ontology_file_path:
        kwargs["ontology_file_path"] = state.pending_ontology_file_path

    try:
        cognify_result = await cognee.cognify(
            datasets=[state.pending_dataset_name],
            **kwargs,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"cognee.cognify failed: {exc}") from exc

    return {
        "message": "Schema approved. cognee.cognify() executed.",
        "dataset_name": state.pending_dataset_name,
        "cognify_result": str(cognify_result),
    }


@app.post("/api/visualize-graph")
async def visualize_graph() -> dict[str, Any]:
    try:
        result = await cognee.visualize_graph(destination_file_path=str(GRAPH_FILE_PATH))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"visualize_graph failed: {exc}") from exc

    return {
        "message": "Graph visualization generated.",
        "graph_url": "/graph/latest",
        "result": str(result),
    }


@app.post("/api/search")
async def search_graph(request: SearchRequest) -> dict[str, Any]:
    dataset_name = request.dataset_name or state.pending_dataset_name
    datasets = [dataset_name] if dataset_name else None

    try:
        results = await cognee.search(
            query_text=request.question.strip(),
            datasets=datasets,
            top_k=50,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"search failed: {exc}") from exc

    return {
        "message": "Search completed.",
        "question": request.question,
        "dataset_name": dataset_name,
        "answer": str(results),
    }


@app.get("/graph/latest")
async def graph_latest() -> FileResponse:
    if not GRAPH_FILE_PATH.exists():
        raise HTTPException(status_code=404, detail="Graph not generated yet. Click Visualize Graph first.")
    return FileResponse(GRAPH_FILE_PATH, media_type="text/html")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

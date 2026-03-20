# DLT Workflow Demo (React + Python)

Simple demo UI for this workflow:

1. Upload relational data source (connection string) and call `cognee.add()`.
2. Extract schema and ask user to approve (`Yes`/`No`).
3. If approved, run `cognee.cognify()`.
4. Optionally pass a custom ontology path.
5. Trigger graph visualization via `visualize_graph()`.
6. Ask a question and search through the generated graph via `search()`.

## Run

From this folder:

```bash
cd /Users/milicevi/cognee_project/cognee-community/experimental/dlt_demo
pip install -r requirements.txt
uvicorn backend:app --reload --port 8010
```

Open:

- http://127.0.0.1:8010

## Notes

- The backend calls:
  - `cognee.add(connection_string, dataset_name=..., query=...)`
  - `cognee.cognify(datasets=[...], ontology_file_path=...)` (if ontology path is set)
  - `cognee.visualize_graph(destination_file_path=...)`
  - `cognee.search(query_text=..., datasets=[...])`
- Schema extraction is done via SQLAlchemy inspector for quick preview.
- For PostgreSQL, ensure the matching SQLAlchemy driver is installed in your environment (for example `psycopg`/`psycopg2`).
- If a DB connection string is being ingested as plain text, it usually means `dlt` was missing.
  This demo now raises a clear error in that case; reinstall with `pip install -r requirements.txt`.

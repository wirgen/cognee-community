const { useState } = React;

function App() {
  const [connectionString, setConnectionString] = useState("sqlite:///sample.db");
  const [datasetName, setDatasetName] = useState("main_dataset");
  const [ontologyPath, setOntologyPath] = useState("");
  const [schema, setSchema] = useState(null);
  const [status, setStatus] = useState("Idle");
  const [busy, setBusy] = useState(false);
  const [graphUrl, setGraphUrl] = useState("");
  const [question, setQuestion] = useState("");
  const [searchAnswer, setSearchAnswer] = useState("");

  async function callApi(path, body) {
    const res = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Request failed");
    }
    return data;
  }

  async function uploadAndExtract() {
    setBusy(true);
    setStatus("Uploading data with cognee.add() and extracting schema...");
    setSchema(null);
    try {
      const data = await callApi("/api/upload-and-extract", {
        connection_string: connectionString,
        dataset_name: datasetName,
        ontology_file_path: ontologyPath || null,
      });
      setSchema(data.schema);
      setStatus(data.message);
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function decideSchema(approved) {
    setBusy(true);
    setStatus(approved ? "Running cognee.cognify()..." : "Schema rejected.");
    try {
      const data = await callApi("/api/schema-decision", { approved });
      setStatus(data.message);
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function visualizeGraph() {
    setBusy(true);
    setStatus("Generating graph visualization...");
    try {
      const data = await callApi("/api/visualize-graph", {});
      setGraphUrl(data.graph_url);
      setStatus(data.message);
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function runSearch() {
    if (!question.trim()) {
      setStatus("Please enter a question first.");
      return;
    }
    setBusy(true);
    setStatus("Searching the generated graph...");
    setSearchAnswer("");
    try {
      const data = await callApi("/api/search", {
        question,
        dataset_name: datasetName || null,
      });
      setSearchAnswer(data.answer || "No answer.");
      setStatus(data.message);
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="wrap">
      <div className="card">
        <h1>Cognee DLT Workflow Demo</h1>
        <p className="muted">
          Upload relational data, review extracted schema, approve to run cognify, optionally use ontology, then visualize graph.
        </p>
      </div>

      <div className="card">
        <div className="grid">
          <label>
            Connection string
            <input
              value={connectionString}
              onChange={(e) => setConnectionString(e.target.value)}
              placeholder="postgresql://user:pass@host:5432/db or sqlite:///path/to.db"
            />
          </label>

          <label>
            Dataset name
            <input
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              placeholder="main_dataset"
            />
          </label>

          <label>
            Custom ontology path (optional)
            <input
              value={ontologyPath}
              onChange={(e) => setOntologyPath(e.target.value)}
              placeholder="/absolute/path/to/ontology.owl"
            />
          </label>
        </div>

        <div className="row" style={{ marginTop: 12 }}>
          <button disabled={busy} onClick={uploadAndExtract}>
            1) Upload Data + Extract Schema
          </button>
        </div>
      </div>

      <div className="card">
        <h3>2) Schema Check</h3>
        <p className="muted">If schema looks good, click Yes to continue with cognify.</p>
        {schema ? (
          <pre>{JSON.stringify(schema, null, 2)}</pre>
        ) : (
          <p className="muted">No schema yet.</p>
        )}
        <div className="row">
          <button className="no" disabled={busy || !schema} onClick={() => decideSchema(false)}>
            No
          </button>
          <button className="yes" disabled={busy || !schema} onClick={() => decideSchema(true)}>
            Yes
          </button>
        </div>
      </div>

      <div className="card">
        <h3>3) Graph Visualization</h3>
        <div className="row">
          <button className="ghost" disabled={busy} onClick={visualizeGraph}>
            Visualize Graph
          </button>
        </div>
        {graphUrl ? (
          <div style={{ marginTop: 12 }}>
            <p>
              Open graph in new tab: <a href={graphUrl} target="_blank" rel="noreferrer">{graphUrl}</a>
            </p>
            <iframe className="graph-frame" title="Graph Visualization" src={graphUrl} />
          </div>
        ) : null}
      </div>

      <div className="card">
        <h3>4) Search Graph</h3>
        <p className="muted">Ask a question about the generated graph.</p>
        <div className="grid" style={{ marginTop: 10 }}>
          <label>
            Question
            <input
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="What are the key relationships in this dataset?"
            />
          </label>
        </div>
        <div className="row" style={{ marginTop: 10 }}>
          <button disabled={busy} onClick={runSearch}>
            Search
          </button>
        </div>
        {searchAnswer ? (
          <div style={{ marginTop: 10 }}>
            <pre>{searchAnswer}</pre>
          </div>
        ) : null}
      </div>

      <div className="card">
        <div className="status">Status: {status}</div>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);

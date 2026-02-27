from cognee.infrastructure.databases.graph import use_graph_adapter

from .turingdb_adapter import TuringDBAdapter

use_graph_adapter("turingdb", TuringDBAdapter)

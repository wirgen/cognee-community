"""Register the Spanner graph adapter with Cognee."""

from cognee.infrastructure.databases.graph import use_graph_adapter

from .spanner_adapter import SpannerGraphAdapter

use_graph_adapter("spanner", SpannerGraphAdapter)

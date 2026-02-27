import asyncio
import json
from textwrap import dedent
from typing import TYPE_CHECKING, Any, List, Optional
from uuid import UUID

from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult
from cognee.infrastructure.engine.utils import parse_id

if TYPE_CHECKING:
    from cognee.infrastructure.databases.graph.graph_db_interface import (
        GraphDBInterface,
    )
    from cognee.infrastructure.databases.vector.vector_db_interface import (
        VectorDBInterface,
    )

from cognee.infrastructure.databases.exceptions import MissingQueryParameterError
from cognee.infrastructure.databases.graph.graph_db_interface import (
    EdgeData,
    Node,
    NodeData,
)
from cognee.infrastructure.databases.vector.embeddings import get_embedding_engine
from cognee.infrastructure.databases.vector.embeddings.EmbeddingEngine import (
    EmbeddingEngine,
)
from cognee.infrastructure.engine import DataPoint

from falkordb.falkordb import FalkorDB
from falkordb.graph import Graph, QueryResult


class IndexSchema(DataPoint):
    """
    Define a schema for indexing that includes text data and associated metadata.

    This class inherits from the DataPoint class. It contains a string attribute 'text' and
    a dictionary 'metadata' that specifies the index fields for this schema.
    """

    text: str

    metadata: dict = {"index_fields": ["text"]}


class FalkorDBAdapter:
    """
    Manage and interact with a graph database using vector embeddings.

    Public methods include:
    - query
    - embed_data
    - stringify_properties
    - create_data_point_query
    - create_edge_query
    - create_collection
    - has_collection
    - create_data_points
    - create_vector_index
    - has_vector_index
    - index_data_points
    - add_node
    - add_nodes
    - add_edge
    - add_edges
    - has_edges
    - retrieve
    - extract_node
    - extract_nodes
    - get_connections
    - search
    - batch_search
    - get_graph_data
    - delete_data_points
    - delete_node
    - delete_nodes
    - delete_graph
    - prune
    - get_node
    - get_nodes
    - get_neighbors
    - get_graph_metrics
    - get_document_subgraph
    - get_degree_one_nodes
    """

    def __init__(
        self,
        graph_database_url: str | None = None,
        graph_database_port: int | None = 6379,
        graph_database_username: str | None = None,
        graph_database_password: str | None = None,
        embedding_engine: EmbeddingEngine | None = None,
        url: str | None = None,
        api_key: str | None = None,
        database_name: str | None = "cognee_graph",
        **kwargs,
    ):
        self.driver = FalkorDB(
            host=url if url else graph_database_url,
            port=graph_database_port if graph_database_port else 6379,
            username=graph_database_username,
            password=graph_database_password,
        )
        self.embedding_engine = get_embedding_engine() if not embedding_engine else embedding_engine
        self.graph_name = database_name if database_name else "cognee_graph"
        self.api_key = api_key

    # TODO: This should return a list of results, not a single result
    def query(self, query: str, params: dict = None) -> QueryResult:
        """
        Execute a query against the graph database.

        Handles exceptions during the query execution by logging errors and re-raising the
        exception.

        The method can be called only if a valid query string and parameters are provided.

        Parameters:
        -----------

            - query (str): The query string to be executed against the graph database.
            - params (dict): A dictionary of parameters to be used in the query. (default {})

        Returns:
        --------

            The result of the query execution, returned by the graph database.
        """
        if params is None:
            params = {}
        graph = self.driver.select_graph(self.graph_name)

        try:
            return graph.query(query, params)
        except Exception as e:
            print(f"Error executing query: {e}")
            raise e

    async def embed_data(self, data: list[str]) -> list[list[float]]:
        """
        Embed a list of text data into vector representations using the embedding engine.

        Parameters:
        -----------

            - data (list[str]): A list of strings that should be embedded into vectors.

        Returns:
        --------

            - list[list[float]]: A list of lists, where each inner list contains float values
              representing the embedded vectors.
        """
        return await self.embedding_engine.embed_text(data)  # type: ignore

    async def stringify_properties(self, properties: dict) -> str:
        """
        Convert properties dictionary to a string format suitable for database queries.

        Parameters:
        -----------

            - properties (dict): A dictionary containing properties to be converted to string
              format.

        Returns:
        --------

            - str: A string representation of the properties in the appropriate format.
        """

        # TODO: Check what types we support
        def parse_value(value: Any) -> str:
            """
            Convert a value to its string representation based on type for database queries.

            Parameters:
            -----------

                - value: The value to parse into a string representation.

            Returns:
            --------

                Returns the string representation of the value in the appropriate format.
            """
            if type(value) is UUID:
                return f"'{str(value)}'"
            if type(value) is int or type(value) is float:
                return str(value)
            if (
                type(value) is list
                and len(value) > 0
                and type(value[0]) is float
                and len(value) == self.embedding_engine.get_vector_size()
            ):
                return f"'vecf32({value})'"
            # if type(value) is datetime:
            #     return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f%z")
            if type(value) is dict:
                return f"'{json.dumps(value).replace(chr(39), chr(34))}'"
            if type(value) is str:
                # Escape single quotes and handle special characters
                escaped_value = (
                    str(value)
                    .replace("'", "\\'")
                    .replace('"', '\\"')
                    .replace("\n", "\\n")
                    .replace("\r", "\\r")
                    .replace("\t", "\\t")
                )
                return f"'{escaped_value}'"
            return f"'{str(value)}'"

        return ",".join([f"{key}:{parse_value(value)}" for key, value in properties.items()])

    async def create_data_point_query(
        self, data_point: DataPoint, vectorized_values: dict
    ) -> tuple[str, dict]:
        """
        Compose a query to create or update a data point in the database.

        Parameters:
        -----------

            - data_point (DataPoint): An instance of DataPoint containing information about the
              entity.
            - vectorized_values (dict): A dictionary of vectorized values related to the data
              point.

        Returns:
        --------

            A tuple containing the query string and parameters dictionary.
        """
        node_label = type(data_point).__name__
        property_names = DataPoint.get_embeddable_property_names(data_point)

        properties = {
            **data_point.model_dump(),
            **(
                {
                    property_names[index]: (
                        vectorized_values[index]
                        if index < len(vectorized_values)
                        else getattr(data_point, property_name, None)
                    )
                    for index, property_name in enumerate(property_names)
                }
            ),
        }

        # Clean the properties - remove None values and handle special types
        clean_properties = {}
        for key, value in properties.items():
            if value is not None:
                if isinstance(value, UUID):
                    clean_properties[key] = str(value)
                elif isinstance(value, dict):
                    clean_properties[key] = json.dumps(value)
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], float):
                    # This is likely a vector - convert to string representation
                    clean_properties[key] = f"vecf32({value})"
                else:
                    clean_properties[key] = value

        query = dedent(
            f"""
            MERGE (node:{node_label} {{id: $node_id}})
            SET node += $properties, node.updated_at = timestamp()
        """
        ).strip()

        params = {"node_id": str(data_point.id), "properties": clean_properties}

        return query, params

    def sanitize_relationship_name(self, relationship_name: str) -> str:
        """
        Sanitize relationship name to be valid for Cypher queries.

        Parameters:
        -----------
            - relationship_name (str): The original relationship name

        Returns:
        --------
            - str: A sanitized relationship name valid for Cypher
        """
        # Replace hyphens, spaces, and other special characters with underscores
        import re

        sanitized = re.sub(r"[^\w]", "_", relationship_name)
        # Remove consecutive underscores
        sanitized = re.sub(r"_+", "_", sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip("_")
        # Ensure it starts with a letter or underscore
        if sanitized and not sanitized[0].isalpha() and sanitized[0] != "_":
            sanitized = "_" + sanitized
        return sanitized or "RELATIONSHIP"

    async def create_edge_query(self, edge: tuple[str, str, str, dict]) -> str:
        """
        Generate a query to create or update an edge between two nodes in the graph.

        Parameters:
        -----------

            - edge (tuple[str, str, str, dict]): A tuple consisting of source and target node
              IDs, edge type, and edge properties.

        Returns:
        --------

            - str: A string containing the query to be executed for creating the edge.
        """
        # Sanitize the relationship name for Cypher compatibility
        sanitized_relationship = self.sanitize_relationship_name(edge[2])

        # Add the original relationship name to properties
        edge_properties = {**edge[3], "relationship_name": edge[2]}
        properties = await self.stringify_properties(edge_properties)
        properties = f"{{{properties}}}"

        return dedent(
            f"""
            MERGE (source {{id:'{edge[0]}'}})
            MERGE (target {{id: '{edge[1]}'}})
            MERGE (source)-[edge:{sanitized_relationship}]->(target)
            ON MATCH SET edge += {properties}, edge.updated_at = timestamp()
            ON CREATE SET edge += {properties}, edge.updated_at = timestamp()
        """
        ).strip()

    # TODO: Check if this is needed, or if collections are created automatically
    async def create_collection(self, collection_name: str) -> None:
        """
        Create a collection in the graph database with the specified name.

        Parameters:
        -----------

            - collection_name (str): The name of the collection to be created.
        """
        pass

    async def has_collection(self, collection_name: str) -> bool:
        """
        Check if a collection with the specified name exists in the graph database.

        Parameters:
        -----------

            - collection_name (str): The name of the collection to check for existence.

        Returns:
        --------

            - bool: Returns true if the collection exists, otherwise false.
        """
        collections = self.driver.list_graphs()

        return collection_name in collections

    async def create_data_points(self, data_points: list[DataPoint]) -> None:
        """
        Add a list of data points to the graph database via batching.

        Can raise exceptions if there are issues during the database operations.

        Parameters:
        -----------

            - data_points (list[DataPoint]): A list of DataPoint instances to be inserted into
              the database.
        """
        embeddable_values: list[Any] = []
        vector_map: dict[str, dict[str, int | None]] = {}

        for data_point in data_points:
            property_names: list[str] = DataPoint.get_embeddable_property_names(data_point)
            key = str(data_point.id)
            vector_map[key] = {}

            for property_name in property_names:
                property_value = getattr(data_point, property_name, None)

                if property_value is not None:
                    vector_map[key][property_name] = len(embeddable_values)
                    embeddable_values.append(property_value)
                else:
                    vector_map[key][property_name] = None

        vectorized_values = await self.embed_data(embeddable_values)

        for data_point in data_points:
            vectorized_data: dict[str, list[float] | None] = {}
            for property_name in DataPoint.get_embeddable_property_names(data_point):
                vector_index = vector_map[str(data_point.id)][property_name]
                if vector_index is not None:
                    vectorized_data[property_name] = vectorized_values[vector_index]
                else:
                    vectorized_data[property_name] = None

            query, params = await self.create_data_point_query(data_point, vectorized_data)
            self.query(query, params)

    async def create_vector_index(self, index_name: str, index_property_name: str) -> None:
        """
        Create a vector index in the specified graph for a given property if it does not already
        exist.

        Parameters:
        -----------

            - index_name (str): The name of the vector index to be created.
            - index_property_name (str): The name of the property on which the vector index will
              be created.
        """
        graph = self.driver.select_graph(self.graph_name)

        if not self.has_vector_index(graph, index_name, f"{index_property_name}_vector"):
            graph.create_node_vector_index(
                index_name,
                f"{index_property_name}_vector",
                dim=self.embedding_engine.get_vector_size(),
                similarity_function="cosine",
            )

    def has_vector_index(self, graph: Graph, index_name: str, index_property_name: str) -> bool:
        """
        Determine if a vector index exists on the specified property of the given graph.

        Parameters:
        -----------

            - graph: The graph instance to check for the vector index.
            - index_name (str): The name of the index to check for existence.
            - index_property_name (str): The property name associated with the index.

        Returns:
        --------

            - bool: Returns true if the vector index exists, otherwise false.
        """
        indices = graph.list_indices()
        for index in indices.result_set:
            if index[0] == index_name:
                continue

        return any(
            [
                (index[0] == index_name and index_property_name in index[1])
                for index in indices.result_set
            ]
        )

    # TODO: Check if this is needed, or if data points are indexed automatically
    async def index_data_points(
        self, index_name: str, index_property_name: str, data_points: list[DataPoint]
    ) -> None:
        """
        Index a list of data points in the specified graph database based on properties.

        To be implemented: does not yet have a defined behavior.

        Parameters:
        -----------

            - index_name (str): The name of the index to be created for the data points.
            - index_property_name (str): The property name on which to index the data points.
            - data_points (list[DataPoint]): A list of DataPoint instances to be indexed.
        """
        pass

    async def add_node(self, node_id: str, properties: dict[str, Any]) -> None:
        """
        Add a single node with specified properties to the graph.

        Parameters:
        -----------

            - node_id (str): Unique identifier for the node being added.
            - properties (Dict[str, Any]): A dictionary of properties associated with the node.
        """
        # Clean the properties - remove None values and handle special types
        clean_properties = {"id": node_id}
        for key, value in properties.items():
            if value is not None:
                if isinstance(value, UUID):
                    clean_properties[key] = str(value)
                elif isinstance(value, dict):
                    clean_properties[key] = json.dumps(value)
                else:
                    clean_properties[key] = value
        query = dedent(f"""
            MERGE (node:{properties["type"]} {{id: $node_id}})
            SET node += $properties, node.updated_at = timestamp()
            """).strip()
        for field in properties["metadata"]["index_fields"]:
            query = query + f", node.{field}_vector = vecf32({properties[f'{field}_vector']})"

        params = {"node_id": node_id, "properties": clean_properties}

        self.query(query, params)

    # Helper methods for DataPoint compatibility
    async def add_data_point_node(self, node: DataPoint) -> None:
        """
        Add a single data point as a node in the graph.

        Parameters:
        -----------

            - node (DataPoint): An instance of DataPoint to be added to the graph.
        """
        await self.create_data_points([node])

    async def add_data_point_nodes(self, nodes: list[DataPoint]) -> None:
        """
        Add multiple data points as nodes in the graph.

        Parameters:
        -----------

            - nodes (list[DataPoint]): A list of DataPoint instances to be added to the graph.
        """
        await self.create_data_points(nodes)

    async def add_nodes(self, nodes: list[Node] | list[DataPoint]) -> None:
        """
        Add multiple nodes to the graph in a single operation.

        Parameters:
        -----------

            - nodes (Union[List[Node], List[DataPoint]]): A list of Node tuples
                                or DataPoint objects to be added to the graph.
        """
        for node in nodes:
            if isinstance(node, tuple) and len(node) == 2:
                # Node is in (node_id, properties) format
                node_id, properties = node
                await self.add_node(node_id, properties)
            elif hasattr(node, "id") and hasattr(node, "model_dump"):
                # Node is a DataPoint object
                # TODO: Figure out how to get this data if node is of type Node, not DataPoint
                embeddable_values = []
                property_names = DataPoint.get_embeddable_property_names(node)  # type: ignore
                vector_map = {}
                for property_name in property_names:
                    property_value = getattr(node, property_name, None)
                    if property_value is not None:
                        vector_map[property_name] = len(embeddable_values)
                        embeddable_values.append(property_value)
                vectorized_values = await self.embed_data(embeddable_values)

                properties = {
                    **node.model_dump(),
                    **(
                        {
                            f"{property_name}_vector": (
                                vectorized_values[vector_map[property_name]]
                                if property_name in vector_map
                                else []
                            )
                            for property_name in property_names
                        }
                    ),
                }

                await self.add_node(str(node.id), properties)
            else:
                raise ValueError(
                    f"Invalid node format: {node}. Expected tuple (node_id, properties)"
                    f"or DataPoint object."
                )

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        relationship_name: str,
        properties: dict[str, Any] | None = None,
    ) -> None:
        """
        Create a new edge between two nodes in the graph.

        Parameters:
        -----------

            - source_id (str): The unique identifier of the source node.
            - target_id (str): The unique identifier of the target node.
            - relationship_name (str): The name of the relationship to be established by the
              edge.
            - properties (Optional[Dict[str, Any]]): Optional dictionary of properties
              associated with the edge. (default None)
        """
        if properties is None:
            properties = {}

        edge_tuple = (source_id, target_id, relationship_name, properties)
        query = await self.create_edge_query(edge_tuple)
        self.query(query)

    async def add_edges(self, edges: list[EdgeData]) -> None:
        """
        Add multiple edges to the graph in a single operation.

        Parameters:
        -----------

            - edges (List[EdgeData]): A list of EdgeData objects representing edges to be added.
        """
        for edge in edges:
            if isinstance(edge, tuple) and len(edge) == 4:
                # Edge is in (source_id, target_id, relationship_name, properties) format
                source_id, target_id, relationship_name, properties = edge
                await self.add_edge(source_id, target_id, relationship_name, properties)
            else:
                raise ValueError(
                    f"Invalid edge format: {edge}. Expected tuple (source_id, target_id,"
                    f"relationship_name, properties)."
                )

    async def has_edges(self, edges: list[EdgeData]) -> list[EdgeData]:
        """
        Check if the specified edges exist in the graph based on their attributes.

        Parameters:
        -----------

            - edges: A list of edges to check for existence in the graph.

        Returns:
        --------

            Returns a list of edge tuples that exist in the graph.
        """
        existing_edges = []
        for edge in edges:
            exists = await self.has_edge(str(edge[0]), str(edge[1]), edge[2])
            if exists:
                existing_edges.append(edge)
        return existing_edges

    async def retrieve(self, data_point_ids: list[UUID]) -> list:
        """
        Retrieve data points from the graph based on their IDs.

        Parameters:
        -----------

            - data_point_ids (list[UUID]): A list of UUIDs representing the data points to
              retrieve.

        Returns:
        --------

            Returns the result set containing the retrieved nodes or an empty list if not found.
        """
        result = self.query(
            "MATCH (node) WHERE node.name IN $node_ids RETURN node",
            {
                "node_ids": [str(data_point) for data_point in data_point_ids],
            },
        )
        return result.result_set  # type: ignore

    async def extract_node(self, data_point_id: UUID) -> NodeData:
        """
        Extract the properties of a single node identified by its data point ID.

        Parameters:
        -----------

            - data_point_id (UUID): The UUID of the data point to extract.

        Returns:
        --------

            Returns the properties of the node if found, otherwise None.
        """
        result = await self.retrieve([data_point_id])
        if not result[0]:
            return None

        node = result[0][0]
        if not node:
            return None

        return node.properties

    async def extract_nodes(self, data_point_ids: list[UUID]) -> list[NodeData]:
        """
        Extract properties of multiple nodes identified by their data point IDs.

        Parameters:
        -----------

            - data_point_ids (list[UUID]): A list of UUIDs representing the data points to
              extract.

        Returns:
        --------

            Returns the properties of the nodes in a list.
        """
        return await self.retrieve(data_point_ids)

    async def get_connections(self, node_id: UUID) -> list:
        """
        Retrieve connection details (predecessors and successors) for a given node ID.

        Parameters:
        -----------

            - node_id (UUID): The UUID of the node whose connections are to be retrieved.

        Returns:
        --------

            - list: Returns a list of tuples representing the connections of the node.
        """
        predecessors_query = """
        MATCH (node)<-[relation]-(neighbour)
        WHERE node.id = $node_id
        RETURN neighbour, relation, node
        """
        successors_query = """
        MATCH (node)-[relation]->(neighbour)
        WHERE node.id = $node_id
        RETURN node, relation, neighbour
        """

        predecessors = self.query(predecessors_query, dict(node_id=node_id))
        successors = self.query(successors_query, dict(node_id=node_id))

        connections = []

        for neighbour in predecessors.result_set:
            connections.append(
                (
                    neighbour[0].properties,
                    {"relationship_name": neighbour[1].properties},
                    neighbour[2].properties,
                )
            )

        for neighbour in successors.result_set:
            connections.append(
                (
                    neighbour[0].properties,
                    {"relationship_name": neighbour[1].properties["relationship_name"]},
                    neighbour[2].properties,
                )
            )

        return connections

    async def search(
        self,
        collection_name: str,
        query_text: str | None = None,
        query_vector: list[float] | None = None,
        limit: int | None = None,
        with_vector: bool = False,
        include_payload: bool = False,
        node_name: Optional[List[str]] = None,
    ) -> list:
        """
        Search for nodes in a collection based on text or vector query, with optional limitation
        on results.

        Parameters:
        -----------

            - collection_name (str): The name of the collection in which to search.
            - query_text (str): The text to search for (if using text-based query). (default
              None)
            - query_vector (list[float]): The vector representation of the query if using
              vector-based search. (default None)
            - limit (int): Maximum number of results to return from the search. (default 10)
            - with_vector (bool): Flag indicating whether to return vectors with the search
              results. (default False)
            - include_payload (bool): Flag indicating whether to include payload (default False)

        Returns:
        --------

            Returns the search results as a result set from the graph database.
        """
        if query_text is None and query_vector is None:
            raise MissingQueryParameterError()

        if query_text and not query_vector:
            query_vector = (await self.embed_data([query_text]))[0]

        if "_" in collection_name:
            label, _, attribute_name = collection_name.partition("_")
        else:
            # If no dot, treat the whole thing as a property search
            label = ""
            attribute_name = collection_name

        graph = self.driver.select_graph(self.graph_name)
        if not self.has_vector_index(graph, label, f"{attribute_name}_vector"):
            raise CollectionNotFoundError(f"No vector index found for collection {collection_name}")

        if limit is None:
            query = f"MATCH (n:{label}) RETURN COUNT(n)"
            result = self.query(query)
            limit = result.result_set[0][0]

        if limit == 0:
            return []

        if include_payload:
            result_properties = ["node"]
        else:
            result_properties = ["node.id"]
            if with_vector:
                result_properties.append(f"node.{attribute_name}_vector")

        query = dedent(f"""
        CALL db.idx.vector.queryNodes(
            '{label}',
            '{attribute_name}_vector',
            {limit},
            vecf32({query_vector}))
        YIELD node, score
        RETURN {", ".join(result_properties)}, score
        """).strip()

        search_results = self.query(query)

        # Convert results to ScoredResult objects
        scored_results = []
        for result in search_results.result_set:
            payload_data = result[0].properties if include_payload else {}
            if "name" in payload_data:
                payload_data["text"] = payload_data["name"]

            if not include_payload:
                res_id = result[0]
                vector = result[1] if with_vector else None
                score = result[2] if with_vector else result[1]
            else:
                res_id = payload_data["id"]
                vector = payload_data[f"{attribute_name}_vector"] if with_vector else None
                score = result[1]

            scored_result = ScoredResult(
                id=parse_id(res_id),
                score=score,
                payload=payload_data,
                vector=vector,
            )
            scored_results.append(scored_result)

        return scored_results

    async def batch_search(
        self,
        collection_name: str,
        query_texts: list[str],
        limit: int | None = None,
        with_vectors: bool = False,
        include_payload: bool = False,
        node_name: Optional[List[str]] = None,
    ) -> list:
        """
        Perform batch search across multiple queries based on text inputs and return results
        asynchronously.

        Parameters:
        -----------

            - collection_name (str): The name of the collection in which to perform the
              searches.
            - query_texts (list[str]): A list of text queries to search for.
            - limit (int): Optional limit for the search results for each query. (default None)
            - with_vectors (bool): Flag indicating whether to return vectors with the results.
              (default False)
            - include_payload (bool): Flag indicating whether to include payload for each query.
              (default False)

        Returns:
        --------

            Returns a list of results for each search query executed in parallel.
        """

        query_vectors = await self.embedding_engine.embed_text(query_texts)

        results: list[list] = await asyncio.gather(
            *[
                self.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    with_vector=with_vectors,
                    include_payload=include_payload,
                )
                for query_vector in query_vectors
            ]
        )
        return results

    async def get_graph_data(self) -> tuple:
        """
        Retrieve all nodes and edges from the graph along with their properties.

        Returns:
        --------

            Returns a tuple containing lists of nodes and edges data retrieved from the graph.
        """
        query = "MATCH (n) RETURN ID(n) AS id, labels(n) AS labels, properties(n) AS properties"

        result = self.query(query)

        nodes = [
            (
                record[2]["id"],
                record[2],
            )
            for record in result.result_set
        ]

        query = """
        MATCH (n)-[r]->(m)
        RETURN ID(n) AS source, ID(m) AS target, TYPE(r) AS type, properties(r) AS properties
        """
        result = self.query(query)
        edges = [
            (
                record[3]["source_node_id"],
                record[3]["target_node_id"],
                record[2],
                record[3],
            )
            for record in result.result_set
        ]

        return (nodes, edges)

    async def delete_data_points(
        self, collection_name: str, data_point_ids: list[UUID]
    ) -> QueryResult:
        """
        Remove specified data points from the graph database based on their IDs.

        Parameters:
        -----------

            - collection_name (str): The name of the collection from which to delete the data
              points.
            - data_point_ids (list[UUID]): A list of UUIDs representing the data points to
              delete.

        Returns:
        --------

            Returns the result of the deletion operation from the database.
        """
        return self.query(
            "MATCH (node) WHERE node.id IN $node_ids DETACH DELETE node",
            {
                "node_ids": [str(data_point) for data_point in data_point_ids],
            },
        )

    async def delete_node(self, node_id: str) -> None:
        """
        Delete a specified node from the graph by its ID.

        Parameters:
        -----------

            - node_id (str): Unique identifier for the node to delete.
        """
        query = f"MATCH (node {{id: '{node_id}'}}) DETACH DELETE node"
        self.query(query)

    async def delete_nodes(self, node_ids: list[str]) -> None:
        """
        Delete multiple nodes from the graph by their identifiers.

        Parameters:
        -----------

            - node_ids (List[str]): A list of unique identifiers for the nodes to delete.
        """
        for node_id in node_ids:
            await self.delete_node(node_id)

    async def delete_graph(self) -> None:
        """
        Delete the entire graph along with all its indices and nodes.
        """
        if self.graph_name not in self.driver.list_graphs():
            return

        try:
            graph = self.driver.select_graph(self.graph_name)

            indices = graph.list_indices()
            for index in indices.result_set:
                for field in index[1]:
                    graph.drop_node_vector_index(index[0], field)

            graph.delete()
        except Exception as e:
            print(f"Error deleting graph: {e}")

    async def get_node(self, node_id: str) -> NodeData | None:
        """
        Retrieve a single node from the graph using its ID.

        Parameters:
        -----------

            - node_id (str): Unique identifier of the node to retrieve.
        """
        result = self.query(
            "MATCH (node) WHERE node.id = $node_id RETURN node",
            {"node_id": node_id},
        )

        if result.result_set and len(result.result_set) > 0:
            # FalkorDB returns node objects as first element in the result list
            return result.result_set[0][0].properties
        return None

    async def get_nodes(self, node_ids: list[str]) -> list[NodeData]:
        """
        Retrieve multiple nodes from the graph using their IDs.

        Parameters:
        -----------

            - node_ids (List[str]): A list of unique identifiers for the nodes to retrieve.
        """
        result = self.query(
            "MATCH (node) WHERE node.id IN $node_ids RETURN node",
            {"node_ids": node_ids},
        )

        nodes = []
        if result.result_set:
            for record in result.result_set:
                # FalkorDB returns node objects as first element in each record
                nodes.append(record[0].properties)
        return nodes

    async def get_neighbors(self, node_id: str) -> list[NodeData]:
        """
        Get all neighboring nodes connected to the specified node.

        Parameters:
        -----------

            - node_id (str): Unique identifier of the node for which to retrieve neighbors.
        """
        result = self.query(
            "MATCH (node)-[]-(neighbor) WHERE node.id = $node_id RETURN DISTINCT neighbor",
            {"node_id": node_id},
        )

        neighbors = []
        if result.result_set:
            for record in result.result_set:
                # FalkorDB returns neighbor objects as first element in each record
                neighbors.append(record[0].properties)
        return neighbors

    async def get_edges(self, node_id: str) -> list[EdgeData]:
        """
        Retrieve all edges that are connected to the specified node.

        Parameters:
        -----------

            - node_id (str): Unique identifier of the node whose edges are to be retrieved.
        """
        result = self.query(
            """
            MATCH (n)-[r]-(m)
            WHERE n.id = $node_id
            RETURN n.id AS source_id, m.id AS target_id, type(r) AS relationship_name,
            properties(r) AS properties
            """,
            {"node_id": node_id},
        )

        edges = []
        if result.result_set:
            for record in result.result_set:
                # FalkorDB returns values by index: source_id, target_id,
                # relationship_name, properties
                edges.append(
                    (
                        record[0],  # source_id
                        record[1],  # target_id
                        record[2],  # relationship_name
                        record[3],  # properties
                    )
                )
        return edges

    async def has_edge(self, source_id: str, target_id: str, relationship_name: str) -> bool:
        """
        Verify if an edge exists between two specified nodes.

        Parameters:
        -----------

            - source_id (str): Unique identifier of the source node.
            - target_id (str): Unique identifier of the target node.
            - relationship_name (str): Name of the relationship to verify.
        """
        # Check both the sanitized relationship type and the original name in properties
        sanitized_relationship = self.sanitize_relationship_name(relationship_name)

        result = self.query(
            f"""
            MATCH (source)-[r:{sanitized_relationship}]->(target)
            WHERE source.id = $source_id AND target.id = $target_id
            AND (r.relationship_name = $relationship_name OR NOT EXISTS(r.relationship_name))
            RETURN COUNT(r) > 0 AS edge_exists
            """,
            {
                "source_id": source_id,
                "target_id": target_id,
                "relationship_name": relationship_name,
            },
        )

        if result.result_set and len(result.result_set) > 0:
            # FalkorDB returns scalar results as a list, access by index instead of key
            return result.result_set[0][0]  # type: ignore
        return False

    async def get_graph_metrics(self, include_optional: bool = False) -> dict[str, Any]:
        """
        Fetch metrics and statistics of the graph, possibly including optional details.

        Parameters:
        -----------

            - include_optional (bool): Flag indicating whether to include optional metrics or
              not. (default False)
        """
        # Get basic node and edge counts
        node_result = self.query("MATCH (n) RETURN count(n) AS node_count")
        edge_result = self.query("MATCH ()-[r]->() RETURN count(r) AS edge_count")

        # FalkorDB returns scalar results as a list, access by index instead of key
        num_nodes = node_result.result_set[0][0] if node_result.result_set else 0
        num_edges = edge_result.result_set[0][0] if edge_result.result_set else 0

        metrics = {
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "mean_degree": (2 * num_edges) / num_nodes if num_nodes > 0 else 0,
            "edge_density": num_edges / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0,
            "num_connected_components": 1,  # Simplified for now
            "sizes_of_connected_components": [num_nodes] if num_nodes > 0 else [],
        }

        if include_optional:
            # Add optional metrics - simplified implementation
            metrics.update(
                {
                    "num_selfloops": 0,  # Simplified
                    "diameter": -1,  # Not implemented
                    "avg_shortest_path_length": -1,  # Not implemented
                    "avg_clustering": -1,  # Not implemented
                }
            )
        else:
            metrics.update(
                {
                    "num_selfloops": -1,
                    "diameter": -1,
                    "avg_shortest_path_length": -1,
                    "avg_clustering": -1,
                }
            )

        return metrics

    async def get_document_subgraph(self, content_hash: str) -> dict:
        """
        Get a subgraph related to a specific document by content hash.

        Parameters:
        -----------

            - content_hash (str): The content hash of the document to find.
        """
        query = """
        MATCH (d) WHERE d.id CONTAINS $content_hash
        OPTIONAL MATCH (d)<-[:CHUNK_OF]-(c)
        OPTIONAL MATCH (c)-[:HAS_ENTITY]->(e)
        OPTIONAL MATCH (e)-[:IS_INSTANCE_OF]->(et)
        RETURN d AS document,
               COLLECT(DISTINCT c) AS chunks,
               COLLECT(DISTINCT e) AS orphan_entities,
               COLLECT(DISTINCT c) AS made_from_nodes,
               COLLECT(DISTINCT et) AS orphan_types
        """

        result = self.query(query, {"content_hash": f"text_{content_hash}"})

        if not result.result_set or not result.result_set[0]:
            return {}

        # Convert result to dictionary format
        # FalkorDB returns values by index: document, chunks, orphan_entities,
        # made_from_nodes, orphan_types
        record = result.result_set[0]
        return {
            "document": record[0],
            "chunks": record[1],
            "orphan_entities": record[2],
            "made_from_nodes": record[3],
            "orphan_types": record[4],
        }

    async def get_degree_one_nodes(self, node_type: str) -> list:
        """
        Get all nodes that have only one connection.

        Parameters:
        -----------

            - node_type (str): The type of nodes to filter by, must be 'Entity' or 'EntityType'.
        """
        if not node_type or node_type not in ["Entity", "EntityType"]:
            raise ValueError("node_type must be either 'Entity' or 'EntityType'")

        result = self.query(
            f"""
            MATCH (n:{node_type})
            WITH n, COUNT {{ MATCH (n)--() }} as degree
            WHERE degree = 1
            RETURN n
            """
        )

        # FalkorDB returns node objects as first element in each record
        return [record[0] for record in result.result_set] if result.result_set else []

    async def get_nodeset_subgraph(
        self, node_type: type[Any], node_name: list[str]
    ) -> tuple[list[tuple[int, dict]], list[tuple[int, int, str, dict]]]:
        """
        Fetch a subgraph consisting of a specific set of nodes and their relationships.

        Parameters:
        -----------

            - node_type (Type[Any]): The type of nodes to include in the subgraph.
            - node_name (List[str]): A list of names of the nodes to include in the subgraph.
        """
        label = node_type.__name__

        # Find primary nodes of the specified type and names
        primary_query = f"""
        UNWIND $names AS wantedName
        MATCH (n:{label})
        WHERE n.name = wantedName
        RETURN DISTINCT n.id, properties(n) AS properties
        """

        primary_result = self.query(primary_query, {"names": node_name})
        if not primary_result.result_set:
            return [], []

        # FalkorDB returns values by index: id, properties
        primary_ids = [record[0] for record in primary_result.result_set]

        # Find neighbors of primary nodes
        neighbor_query = """
        MATCH (n)-[]-(neighbor)
        WHERE n.id IN $ids
        RETURN DISTINCT neighbor.id, properties(neighbor) AS properties
        """

        neighbor_result = self.query(neighbor_query, {"ids": primary_ids})
        # FalkorDB returns values by index: id, properties
        neighbor_ids = (
            [record[0] for record in neighbor_result.result_set]
            if neighbor_result.result_set
            else []
        )

        all_ids = list(set(primary_ids + neighbor_ids))

        # Get all nodes in the subgraph
        nodes_query = """
        MATCH (n)
        WHERE n.id IN $ids
        RETURN n.id, properties(n) AS properties
        """

        nodes_result = self.query(nodes_query, {"ids": all_ids})
        nodes = []
        if nodes_result.result_set:
            for record in nodes_result.result_set:
                # FalkorDB returns values by index: id, properties
                nodes.append((record[0], record[1]))

        # Get edges between these nodes
        edges_query = """
        MATCH (a)-[r]->(b)
        WHERE a.id IN $ids AND b.id IN $ids
        RETURN a.id AS source_id, b.id AS target_id, type(r) AS relationship_name,
        properties(r) AS properties
        """

        edges_result = self.query(edges_query, {"ids": all_ids})
        edges = []
        if edges_result.result_set:
            for record in edges_result.result_set:
                # FalkorDB returns values by index: source_id, target_id,
                # relationship_name, properties
                edges.append(
                    (
                        record[0],  # source_id
                        record[1],  # target_id
                        record[2],  # relationship_name
                        record[3],  # properties
                    )
                )

        return nodes, edges

    async def prune(self) -> None:
        """
        Prune the graph by deleting the entire graph structure.
        """
        await self.delete_graph()

    async def is_empty(self) -> bool:
        query = "MATCH (n) RETURN true LIMIT 1;"
        result = self.query(query)
        return not result.result_set


if TYPE_CHECKING:
    _a: GraphDBInterface = FalkorDBAdapter("", 0, None)
    _b: VectorDBInterface = FalkorDBAdapter("", 0, None)

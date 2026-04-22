from cognee.infrastructure.databases.dataset_database_handler import use_dataset_database_handler
from cognee.infrastructure.databases.vector import use_vector_adapter

from .moss_adapter import MossAdapter
from .MossDatasetDatabaseHandler import MossDatasetDatabaseHandler

use_vector_adapter("moss", MossAdapter)
use_dataset_database_handler("moss", MossDatasetDatabaseHandler, "moss")

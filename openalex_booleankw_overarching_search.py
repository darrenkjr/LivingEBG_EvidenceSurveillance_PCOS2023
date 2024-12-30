import pandas as pd 
from datetime import datetime, timedelta
from libraries.openalex_client import OpenAlexClient
import pyarrow as pa
import pyarrow.parquet as pq
from libraries.eval import search_evaluation
from pathlib import Path
from libraries.logging_config import LoggerConfig
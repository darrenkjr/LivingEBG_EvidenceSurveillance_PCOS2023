import pandas as pd 
import pyarrow as pa 
from pathlib import Path
from libraries.logging_config import LoggerConfig
from libraries.eval import search_evaluation

def ris_eval_pipeline(database, search_type): 

    '''
    Takes in search result path containing ris files, evaluates results and saves metrics

    Args: 
        path: Path object to search results 
        search_type: overarching vs topic specific search
        database: embase vs medline 
    '''
    logger = LoggerConfig.setup_logger(logger_name = f'{database}_{search_type}')
    search_eval_cls = search_evaluation(database = database, search_type = search_type, logger = logger)
    search_eval_cls.run_eval_pipeline()


database_list = ['medline']
for database in database_list: 
    # ris_eval_pipeline(database, search_type = 'overarching')
    ris_eval_pipeline(database, search_type = 'topic_specific')




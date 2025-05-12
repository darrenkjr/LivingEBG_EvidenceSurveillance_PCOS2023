import pandas as pd 
import pyarrow as pa 
from pathlib import Path
from libraries.logging_config import LoggerConfig
from libraries.eval import search_evaluation


class embmedline_ris_pipeline: 

    def __init__(self,database, search_type): 
        self.database = database
        self.search_type = search_type
        self.logger = LoggerConfig.setup_logger(logger_name= f'{database}_{search_type}')
        search_strategy_path = Path(__file__).parent / 'dataset' / 'ovid_kw_searches.xlsx'
        self.overarching_search_strats = pd.read_excel(search_strategy_path, sheet_name = 'overarching')
        topic_specific_embase = pd.read_excel(search_strategy_path, sheet_name = 'medline-topicspecific')
        topic_specific_medline = pd.read_excel(search_strategy_path, sheet_name = 'embase-topicspecific')
        self.topic_specific_strats = pd.concat([topic_specific_embase, topic_specific_medline], ignore_index=True)

    def ris_eval_pipeline(self): 

            '''
            Takes in search result path containing ris files, evaluates results and saves metrics

            Args: 
                path: Path object to search results 
                search_type: overarching vs topic specific search
                database: embase vs medline 
            '''
            search_eval_cls = search_evaluation(database = self.database, search_type = self.search_type, logger = self.logger)
            evalmetrics_df = search_eval_cls.run_eval_pipeline()
            return evalmetrics_df
    
if __name__ == '__main__': 
    embmedline_ris_pipeline_cls = embmedline_ris_pipeline(database = 'embase', search_type = 'topic_specific')
    evalmetrics_df = embmedline_ris_pipeline_cls.ris_eval_pipeline()







import pandas as pd 
from pathlib import Path
import asyncio
from libraries.openalex_client import OpenAlexClient
from libraries.logging_config import LoggerConfig
from libraries.eval import search_evaluation
import pyarrow as pa
import pyarrow.parquet as pq

class oa_keywordsearch_dev: 

    def __init__(self, logger = None): 
        self.logger = LoggerConfig.setup_logger(logger_name='oa_kw_search')
        self.oa_client = OpenAlexClient(logger = self.logger)
        self.eval_cls = search_evaluation(
            database = 'oa', 
            search_type = 'overarching', 
            strategy_type = 'boolkw_search', 
            logger = self.logger)
        
        current_dir = Path(__file__).parent

        self.results_path = current_dir / 'search_results' / 'openalex' / 'overarching' / 'boolkw_search'
        self.results_path.mkdir(parents=True, exist_ok=True)

    async def oa_kw_search(self):

        query_kw_list = ['PCOS', 
                         'PCOD', 
                         'polycystic ovarian syndrome', 'polycystic ovarian disease', 
                         'polycystic ovary syndrome', 'polycystic ovary disease'] 
        try: 
            #examine existing files if topics have already been retrieved 
            existing_files = [f.name for f in self.results_path.iterdir() if f.is_file()]
            #search kw retrieved  
            existing_kwsearch_resutls = [f.split('_')[2].replace('.parquet', '') for f in existing_files]
            #check if kw search results are in existing_kwsearch_resutls 
            kwsearch_results_retrieved = [query for query in query_kw_list if query in existing_kwsearch_resutls]
            query_kw_list_remaining = [query for query in query_kw_list if query not in kwsearch_results_retrieved]

            if len(query_kw_list_remaining) == 0: 
                self.logger.info('All keyword search results have already been retrieved, proceeding to evaluation')

            elif len(query_kw_list_remaining) > 0: 

                async with self.oa_client as client:
                    for query in query_kw_list_remaining:
                        try: 
                            self.logger.info(f'Retrieving OpenAlex keyword search results for {query}')
                            results_df = await client.retrieve_oa_kwsearch_data(query)
                            result_table = pa.Table.from_pandas(results_df)
                            pq.write_table(result_table, self.results_path / f'boolkw_results_{query}.parquet')

                        except Exception as e: 
                            self.logger.error(f'Error retrieving OpenAlex keyword search results for {query}: {e}')
                            continue
        
        except Exception as e: 
            self.logger.error(f'Error retrieving OpenAlex keyword search results: {e}')
            raise

                

    async def oa_kw_search_eval_pipeline(self): 
        try: 
            await self.oa_kw_search()
            self.logger.info('OpenAlex keyword search results retrieved, proceeding to evaluation')
            self.eval_cls.run_eval_pipeline()
        except Exception as e: 
            self.logger.error(f'Error running OpenAlex keyword search evaluation pipeline: {e}')
            raise

if __name__ == '__main__': 
    oa_keywordsearch_cls = oa_keywordsearch_dev()
    asyncio.run(oa_keywordsearch_cls.oa_kw_search_eval_pipeline())

        


        





    



        


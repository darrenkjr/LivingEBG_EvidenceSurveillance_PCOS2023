import pandas as pd 
from datetime import datetime, timedelta
import asyncio 
from libraries.openalex_client import OpenAlexClient
import pyarrow as pa
import pyarrow.parquet as pq
from convenience_scripts.openalex_topicsearch_builder import oa_topicsearch_builder
from convenience_scripts.eval import search_evaluation
from pathlib import Path
from libraries.logging_config import LoggerConfig


async def oa_overarching_search(topic_id_list : list, logger = None): 

    async with OpenAlexClient(logger=logger) as oa_client: 
        #generate topic ids 
        logger.info(f'Number of topics: {len(topic_id_list)}')
        #results path 
        current_dir = Path(__file__).parent
        results_path = current_dir / 'search_results' / 'openalex' / 'overarching' / 'topic_search'
        #create results path if it doesn't exist 
        results_path.mkdir(parents=True, exist_ok=True)

        #examine results path and check if topics have already been retrieved 
        existing_files = [f.name for f in results_path.iterdir() if f.is_file()]
        oa_prefix = 'https://openalex.org/'
        existing_topics = [oa_prefix + f.split('_')[2].replace('.parquet', '') for f in existing_files]
        #check if existing topics are in topic_id_list 
        topics_retrieved = [topic for topic in topic_id_list if topic in existing_topics]
        topic_id_list_remaining = [topic for topic in topic_id_list if topic not in topics_retrieved]
        
        if len(topic_id_list_remaining) == 0: 
            logger.info('All topics have already been retrieved, proceeding to evaluation')

        elif len(topic_id_list_remaining) > 0: 
            for topic_id in topic_id_list_remaining: 
                results_df = await oa_client.get_overarching_paginated_search_results(topic_id)
                result_table = pa.Table.from_pandas(results_df)
                topic_name = topic_id.split('/')[-1]
                logger.info(f'saving results for topic: {topic_name}')
                pq.write_table(result_table, results_path / f'topic_searchresults_{topic_name}.parquet')


async def oa_overarching_search_eval_pipeline():  
    logger = LoggerConfig.setup_logger(logger_name='oa_overarching_search')
    oa_topicsearch_builder_cls = oa_topicsearch_builder(logger=logger)
    await oa_topicsearch_builder_cls.retrieve_oa_topics()
    topic_id_list = oa_topicsearch_builder_cls.generate_openalex_topicsearch_ids()
    await oa_overarching_search(topic_id_list, logger=logger) 
    search_eval_cls = search_evaluation('oa', 'overarching', vector_search = False, logger=logger, strategy_type='topic_search')
    search_eval_cls.run_eval_pipeline()

if __name__ == '__main__': 
    asyncio.run(oa_overarching_search_eval_pipeline())




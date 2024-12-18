import pandas as pd 
from datetime import datetime, timedelta
import asyncio 
import aiohttp
from libraries.openalex_client import OpenAlexClient
import pyarrow as pa
import pyarrow.parquet as pq
from convenience_scripts.openalex_topicsearch_builder import oa_topicsearch_builder
from convenience_scripts.eval import search_evaluation
from pathlib import Path


async def oa_overarching_search(topic_id_list : list): 
    async with OpenAlexClient() as oa_client: 
        #generate topic ids 
        print(f'Number of topics: {len(topic_id_list)}')
        #results path 
        current_dir = Path(__file__).parent
        results_path = current_dir / 'search_results' / 'openalex' / 'overarching'

        for topic_id in topic_id_list: 
            results_df = await oa_client.get_overarching_paginated_search_results(topic_id)
            result_table = pa.Table.from_pandas(results_df)
            topic_name = topic_id.split('/')[-1]
            print('saving results for topic: ', topic_name)
            pq.write_table(result_table, results_path / f'topic_searchresults_{topic_name}.parquet')


async def oa_overarching_search_eval_pipeline():  
    oa_topicsearch_builder_cls = oa_topicsearch_builder()
    await oa_topicsearch_builder_cls.retrieve_oa_topics()
    topic_id_list = oa_topicsearch_builder_cls.generate_openalex_topicsearch_ids()
    await oa_overarching_search(topic_id_list) 
    search_eval_cls = search_evaluation('oa', 'overarching', vector_search = False)
    search_eval_cls.run_eval_pipeline()

if __name__ == '__main__': 
    asyncio.run(oa_overarching_search_eval_pipeline())




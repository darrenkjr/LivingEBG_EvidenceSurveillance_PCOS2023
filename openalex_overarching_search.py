import pandas as pd 
from datetime import datetime, timedelta
import asyncio 
import aiohttp
from libraries.openalex_client import OpenAlexClient
import pyarrow as pa
import pyarrow.parquet as pq
from convenience_scripts.openalex_topicsearch_builder import generate_openalex_topicsearch_id
from pathlib import Path


async def main(): 
    async with OpenAlexClient() as oa_client: 
        #generate topic ids 
        topic_id_list = generate_openalex_topicsearch_id(topic_id_list)

        #results path 
        current_dir = Path(__file__).parent
        results_path = current_dir / 'search_results' / 'openalex'

        for topic_id in topic_id_list: 
            results_df = await oa_client.get_paginated_results(topic_id)
            result_table = pa.Table.from_pandas(results_df)
        
            topic_name = topic_id.split('/')[-1]
            print('saving results for topic: ', topic_name)
            pq.write_table(result_table, results_path / f'topic_searchresults_{topic_name}.parquet')

if __name__ == '__main__': 
    asyncio.run(main())





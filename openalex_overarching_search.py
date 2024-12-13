import pandas as pd 
from datetime import datetime, timedelta
import asyncio 
import aiohttp
from libraries.openalex_client import OpenAlexClient
import pyarrow as pa
import pyarrow.parquet as pq



topic_id_list = ['https://openalex.org/T10390', 'https://openalex.org/T10290', 'https://openalex.org/T10351', 'https://openalex.org/T11144', 'https://openalex.org/T10196', 'https://openalex.org/T10263', 'https://openalex.org/T10459', 'https://openalex.org/T10499', 'https://openalex.org/T10560', 'https://openalex.org/T11888', 'https://openalex.org/T13430']
#test topic id 
async def main(): 
    async with OpenAlexClient() as oa_client: 
        for topic_id in topic_id_list: 
            results_df = await oa_client.get_paginated_results(topic_id)
            result_table = pa.Table.from_pandas(results_df)
        
            topic_name = topic_id.split('/')[-1]
            print('saving results for topic: ', topic_name)
            pq.write_table(result_table, f'search_results/openalex/topic_searchresults_{topic_name}.parquet')

if __name__ == '__main__': 
    asyncio.run(main())





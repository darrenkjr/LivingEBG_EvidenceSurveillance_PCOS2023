import polars as pl
from datetime import datetime, timedelta
import asyncio 
import aiohttp
from libraries.openalex_client import OpenAlexClient



topic_id_list = ['https://openalex.org/T10390', 'https://openalex.org/T10290', 'https://openalex.org/T10351', 'https://openalex.org/T11144', 'https://openalex.org/T10196', 'https://openalex.org/T10263', 'https://openalex.org/T10459', 'https://openalex.org/T10499', 'https://openalex.org/T10560', 'https://openalex.org/T11888', 'https://openalex.org/T13430']
#test topic id 
topic_id = topic_id_list[0]

async def main(): 
    async with OpenAlexClient() as oa_client: 
        results = await oa_client.get_paginated_results(topic_id)
        print('saving results')
        results.write_csv('ovearching_search_results.csv')

if __name__ == '__main__': 
    asyncio.run(main())





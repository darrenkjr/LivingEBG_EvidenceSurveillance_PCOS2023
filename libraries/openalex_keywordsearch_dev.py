import pandas as pd 
from pathlib import Path
import ast 
import sys 
import asyncio
from openalex_client import OpenAlexClient

class oa_keywordsearch_dev: 

    def __init__(self, logger = None): 
        self.oa_client = OpenAlexClient()
        goldset_path_oa = Path(__file__).parent.parent / 'dataset' / 'combined_goldset_oatopics.parquet'
        #goldset to test if the kewyord search is working - then test on rest 
        self.goldset_df = pd.read_parquet(goldset_path_oa)
        self.logger = logger

    async def oa_kw_search(self):

        query_kw_list = ['PCOS', 'PCOD', '(PCOS OR PCOD)'] 

        #test combinations 
        async with self.oa_client as client:
            retrieved_goldset_topic_df = await client.retrieve_oa_kwsearch_data(query_kw_list)



    



        


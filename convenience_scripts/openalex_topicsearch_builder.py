import pandas as pd 
from pathlib import Path
import ast 
import sys 
import asyncio
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from libraries.openalex_client import OpenAlexClient

class oa_topicsearch_builder: 

    def __init__(self, logger = None): 
        self.oa_client = OpenAlexClient(logger=logger)
        self.goldset_path_pq = Path(__file__).parent.parent / 'dataset' / 'combined_goldset.parquet'
        self.goldset_path_xlsx = Path(__file__).parent.parent / 'dataset' / 'combined_goldset.xlsx'
        self.goldset_path_pq_out = Path(__file__).parent.parent / 'dataset' / 'combined_goldset_oatopics.parquet'
        self.goldset_path_xlsx_out = Path(__file__).parent.parent / 'dataset' / 'combined_goldset_oatopics.xlsx'
        self.goldset_df = pd.read_parquet(self.goldset_path_pq)
        self.logger = logger
        

    async def retrieve_oa_topics(self): 
        #retrieve topics for each oa id in the oa gold set df 
        goldset_df_valid_id = self.goldset_df[self.goldset_df['retrieved_oa_id'].notna()].copy()
        async with self.oa_client as client:
            #drop duplicate oa_ids in the first instnnce 
            goldset_df_nonduplicate = goldset_df_valid_id.drop_duplicates(subset='retrieved_oa_id')
            goldset_df_nonduplicate['retrieved_oa_id'] = goldset_df_nonduplicate['retrieved_oa_id'].apply(self._clean_oa_id)
            self.logger.info(f'Retrieving OA topics for goldset, total number of unique oa ids: {len(goldset_df_nonduplicate["retrieved_oa_id"])}')
            retrieved_goldset_topic_df = await client.retrieve_oa_data(goldset_df_nonduplicate['retrieved_oa_id'])
        # Clean up IDs for comparison
        retrieved_goldset_topic_df['id'] = retrieved_goldset_topic_df['id'].apply(self._clean_oa_id)
        #drop duplicates 
        retrieved_goldset_topic_df = retrieved_goldset_topic_df.drop_duplicates(subset='id')

        # Create a mapping dictionary from retrieved data
        topic_mapping = retrieved_goldset_topic_df.set_index('id')[['primary_topic']].to_dict('index')
        
        # Apply the mapping to all rows in original DataFrame
        goldset_df_valid_id['retrieved_oa_id'] = goldset_df_valid_id['retrieved_oa_id'].apply(self._clean_oa_id)
        goldset_df_valid_id['primary_topic'] = goldset_df_valid_id['retrieved_oa_id'].map(
            lambda x: topic_mapping.get(x, {}).get('primary_topic') if x in topic_mapping else None
        )

        
        # Find missing IDs
        missing_ids = set(goldset_df_valid_id['retrieved_oa_id']) - set(retrieved_goldset_topic_df['id'])
        self.logger.info(f'Number of ids not retrieved: {len(missing_ids)}')

        if missing_ids: 
            missing_ids_df_rerun = await self.rerun_missing_ids(missing_ids)
            missing_ids_df_rerun['id'] = missing_ids_df_rerun['id'].apply(self._clean_oa_id)
            missing_id_mapping = missing_ids_df_rerun.set_index('id')[['primary_topic']].to_dict('index')
            goldset_df_valid_id['primary_topic'] = goldset_df_valid_id['retrieved_oa_id'].map(
                lambda x: missing_id_mapping.get(x, {}).get('primary_topic') if x in missing_id_mapping else None
            )

            #check number of rows with missing topics 
            if goldset_df_valid_id['primary_topic'].isna().sum() > 0: 
                self.logger.warning(f'Number of rows with missing topics after rerun: {str(goldset_df_valid_id['primary_topic'].isna().sum())}')

        #merge valid id back with full goldset 
        complete_topic_mapping = dict(zip(
            goldset_df_valid_id['retrieved_oa_id'],
            goldset_df_valid_id['primary_topic']
        ))
        #normazlie oa_id before mapping 
        self.goldset_df['retrieved_oa_id'] = self.goldset_df['retrieved_oa_id'].apply(lambda x : self._clean_oa_id(x) if x is not None else None)
        self.goldset_df['primary_topic'] = self.goldset_df['retrieved_oa_id'].map(complete_topic_mapping)
        self.logger.info(f'Number of rows with missing topics with valid OA ids after gold set topic update: {
            self.goldset_df[
                (self.goldset_df["retrieved_oa_id"].notna()) & 
                (self.goldset_df["primary_topic"].isna())
            ].shape[0]
        }')
        #update goldset files 
        self.goldset_df.to_parquet(self.goldset_path_pq_out)
        self.goldset_df.to_excel(self.goldset_path_xlsx_out)

    @staticmethod
    def _clean_oa_id(oa_id): 
        return oa_id.replace('https://openalex.org/', '').lower()
    
    async def rerun_missing_ids(self, missing_ids): 
        #rerun the missing ids 
        async with self.oa_client as client: 
            retrieved_goldset_topic_df = await client.retrieve_oa_data(list(missing_ids))

        return retrieved_goldset_topic_df

    def generate_openalex_topicsearch_ids(self): 
        
        self.goldset_df['topic_id'] = self.goldset_df['primary_topic'].apply(lambda x: x['id'] if pd.notna(x) else None)
        selected_topic_ids = list(set(self.goldset_df['topic_id'].dropna()))
        self.logger.info(f'Number of topic ids generated: {len(selected_topic_ids)}')
        return selected_topic_ids

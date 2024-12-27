import pandas as pd 
from pathlib import Path
import ast 
import sys 
import asyncio
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from libraries.openalex_client import OpenAlexClient

class oa_topicsearch_builder: 

    def __init__(self): 
        self.oa_client = OpenAlexClient()
        goldset_path = Path(__file__).parent.parent / 'dataset' / 'combined_goldset.parquet'
        self.goldset_df = pd.read_parquet(goldset_path)
        self.goldset_df_valid_id = self.goldset_df[self.goldset_df['retrieved_oa_id'].notna()]

    async def retrieve_oa_topics(self): 
        #retrieve topics for each oa id in the oa gold set df 
        
        async with self.oa_client as client:
            #drop duplicate oa_ids in the first instnnce 
            goldset_df_nonduplicate = self.goldset_df_valid_id.drop_duplicates(subset='retrieved_oa_id')
            goldset_df_nonduplicate['retrieved_oa_id'] = goldset_df_nonduplicate['retrieved_oa_id'].str.lower()
            print('Retrieving topics for goldset, total number of unique oa ids: ', len(goldset_df_nonduplicate['retrieved_oa_id']))
            retrieved_goldset_topic_df = await client.retrieve_oa_data(goldset_df_nonduplicate['retrieved_oa_id'])
        # Clean up IDs for comparison
        retrieved_goldset_topic_df['id'] = retrieved_goldset_topic_df['id'].str.replace('https://openalex.org/', '')
        retrieved_goldset_topic_df['id'] = retrieved_goldset_topic_df['id'].str.lower()
        #drop duplicates 
        retrieved_goldset_topic_df = retrieved_goldset_topic_df.drop_duplicates(subset='id')

        # Create a mapping dictionary from retrieved data
        topic_mapping = retrieved_goldset_topic_df.set_index('id')[['primary_topic']].to_dict('index')
        
        # Apply the mapping to all rows in original DataFrame
        self.goldset_df_valid_id['retrieved_oa_id'].str.lower()
        self.goldset_df_valid_id['primary_topic'] = self.goldset_df_valid_id['retrieved_oa_id'].map(
            lambda x: topic_mapping.get(x, {}).get('primary_topic') if x in topic_mapping else None
        )

        # Verify results
        print("\nResults verification:")
        print(f"Original rows: {len(self.goldset_df_valid_id)}")
        
        # Find missing IDs
        self.missing_ids = set(self.goldset_df_valid_id['retrieved_oa_id']) - set(retrieved_goldset_topic_df['id'])
        print('Number of ids not retrieved: ', len(self.missing_ids))
        if self.missing_ids: 
            missing_ids_df_rerun = await self.rerun_missing_ids()


        return self.goldset_df_valid_id
    
    async def rerun_missing_ids(self): 
        #rerun the missing ids 
        async with self.oa_client as client: 
            retrieved_goldset_topic_df = await client.retrieve_oa_topics(list(self.missing_ids))

        return retrieved_goldset_topic_df

    def generate_openalex_topicsearch_id(self): 

        # Parse topic objects safely
        def parse_topics(x):
            if pd.isna(x):
                return None
            try:
                if isinstance(x, str):
                    return ast.literal_eval(x)
                elif isinstance(x, pd.Series):
                    return ast.literal_eval(x.iloc[0])
                return x
            except:
                return None

        self.goldset_df['topic_obj'] = self.goldset_df['topics'].apply(parse_topics)
        self.goldset_df['topic_ids'] = self.goldset_df['topic_obj'].apply(
            lambda x: [topic['id'] for topic in x] if isinstance(x, list) else None
        )

        # Flatten topic IDs more safely
        flattened_data = []
        for idx, sublist in self.goldset_df['topic_ids'].items():
            if isinstance(sublist, list):
                for item in sublist:
                    flattened_data.append({
                        'original_index': idx,
                        'topic_id': item
                    })

        flattened_df = pd.DataFrame(flattened_data)
        # Group by topic_id and count unique indices
        topic_id_coverage = flattened_df.groupby('topic_id')['original_index'].nunique().reset_index()
        topic_id_coverage.columns = ['topic_id', 'unique_index_count']

        # Calculate total unique indices from flattened_df instead of index
        total_unique_indices = len(flattened_df['original_index'].unique())

        # Calculate percentage coverage
        topic_id_coverage['percentage_coverage'] = (topic_id_coverage['unique_index_count'] / total_unique_indices) * 100
        topic_id_coverage_sorted = topic_id_coverage.sort_values(by='percentage_coverage', ascending=False)

        # Group the DataFrame by topic_id and collect the indices for each topic_id
        topic_id_to_indices = flattened_df.groupby('topic_id')['original_index'].apply(set).to_dict()

        # Initialize a set of all observation indices
        all_observations = set(flattened_df['original_index'].unique())

        # Initialize an empty set for covered observations and a list to store the selected topic_ids
        covered_observations = set()
        selected_topic_ids = []

        while covered_observations != all_observations:
            # Check if there are any topic_ids left to consider
            if not topic_id_to_indices:
                print("No more topic_ids left to select from.")
                break
            
            # Find the topic_id that covers the most uncovered observations
            best_topic_id = max(topic_id_to_indices, key=lambda k: len(topic_id_to_indices[k] - covered_observations))
            
            # Add this topic_id to the selected list
            selected_topic_ids.append(best_topic_id)
            
            # Update the covered observations
            covered_observations.update(topic_id_to_indices[best_topic_id])
            
            # Remove the selected topic_id from consideration
            del topic_id_to_indices[best_topic_id]

        if covered_observations == all_observations:
            print("Selected topic IDs to cover all observations:", selected_topic_ids)
        else:
            print("Not all observations could be covered. Selected topic IDs so far:", selected_topic_ids)
        
        return selected_topic_ids



oa_topicsearch_builder = oa_topicsearch_builder()
goldset_df_topics, missed_ids = asyncio.run(oa_topicsearch_builder.retrieve_oa_topics())
print(missed_ids)
goldset_df_topics.to_excel(Path(__file__).parent.parent / 'dataset' / 'oa_topicsearch_goldset.xlsx')
goldset_df_topics.to_parquet(Path(__file__).parent.parent / 'dataset' / 'oa_topicsearch_goldset.parquet')

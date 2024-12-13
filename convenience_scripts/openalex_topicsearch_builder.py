import pandas as pd 
from pathlib import Path
import ast 

def generate_openalex_topicsearch_id(): 
    current_dir = Path(__file__).parent
    dataset_path = current_dir.parent / 'dataset' / 'combined_goldset.parquet'
    goldset_df = pd.read_parquet(dataset_path)
    oa_data = pd.read_excel(current_dir.parent / 'dataset' / '_superseded' / 'api_retrieved_final.xlsx', sheet_name = 'api_results_oa', dtype={'included_article_id': str})

    #add topic obj to data 

    oa_data.set_index('included_article_id', inplace = True)
    goldset_df.set_index('included_article_id', inplace = True)

    #add topics to goldset 
    goldset_df = goldset_df.join(
        oa_data['topics'],  # Select topics column as DataFrame
        how='left'  # Use left join to keep all goldset records
    )

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

    goldset_df['topic_obj'] = goldset_df['topics'].apply(parse_topics)
    goldset_df['topic_ids'] = goldset_df['topic_obj'].apply(
        lambda x: [topic['id'] for topic in x] if isinstance(x, list) else None
    )

    # Flatten topic IDs more safely
    flattened_data = []
    for idx, sublist in goldset_df['topic_ids'].items():
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

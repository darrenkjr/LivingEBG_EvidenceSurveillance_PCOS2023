import pandas as pd 
from pathlib import Path
from cleanhtmltags import clean_html_tags


def groundtruth_setup():    
    #defining file and dataset directories 
    file_dir = Path(__file__).parent
    dataset_dir = file_dir.parent / 'dataset' / '_superseded'
    output_dir = file_dir.parent / 'dataset' 

    rq_dataset_path = dataset_dir / 'PCOS_Guideline_Dataset_checked.xlsm'
    apiretrieved_groundtruth_path = dataset_dir / 'api_retrieved_final.xlsx'
    unsucessful_rob_path = dataset_dir / 'all_unsuccessful_nodupe_rob_FIXED.csv'

    #read in PCOS dataset, and extract valid rqs 
    _df = pd.read_excel(rq_dataset_path, sheet_name='rq_evidence_review', engine='openpyxl', dtype={'question_id': str})
    pcosrq_valid_df = _df.query('evidence_review_type == "SR" and included_num >= 5').copy()[['GDG', 'question_id', 'Topic', 'Question', 'sr_update', 'included_num', 'searchstrat_year_start', 'searchstrat_year_end']]

    #extract full ground truth dataset 
    fullgroundtruth_full_df = pd.read_excel(rq_dataset_path, sheet_name='included_articles', engine='openpyxl', dtype={'question_id': str, 'included_article_id': str, 'retrieved_oa_id': str, 'retrieved_embase_id': str, 'retrieved_pubmed_id': str}).copy()
    #read in ROB (from collection of uncessfully retrieved articles) 
    unsucessful_rob_df = pd.read_csv(unsucessful_rob_path, dtype={'included_article_id': str})
    fullgroundtruth_full_df['assessed_rob'] = fullgroundtruth_full_df.join(
        unsucessful_rob_df.set_index('included_article_id')['rob'],
        on = 'included_article_id'
    )['rob']
    fullgroundtruth_full_df[['sr_update', 'searchstrat_year_start', 'searchstrat_year_end']] = pd.merge(fullgroundtruth_full_df, pcosrq_valid_df, on='question_id', how='left')[
        ['sr_update', 'searchstrat_year_start', 'searchstrat_year_end']
        ]

    #define required columns 
    required_cols = ['GDG', 'question_id', 'sr_update', 'searchstrat_year_start', 'searchstrat_year_end',
                    'included_article_id', 'included_reference', 'author_year_format', 'year_pub_extract', 'assessed_rob']

    # Filter fullgroundtruth_full_df to only include rows where question_id matches those in pcosrq_valid_df (mmeet inclusion criteria)
    fullgroundtruth_valid_df = fullgroundtruth_full_df.query('question_id in @pcosrq_valid_df["question_id"]')[required_cols]

    #consolidate ground truth datasets
    #OpenAlex 
    oa_groundtruth_df = pd.read_excel(apiretrieved_groundtruth_path, 
                                    sheet_name="api_results_oa", 
                                    engine='openpyxl', 
                                    dtype={'included_article_id': str})

    # Embase dataset
    print("\nEMBASE Dataset:")
    embase_groundtruth_df = pd.read_excel(apiretrieved_groundtruth_path, 
                                        sheet_name="api_results_embase", 
                                        engine='openpyxl', 
                                        dtype={'included_article_id': str, 'api_id_retrieved' : str})
    print("Missing values before fillna:")
    print(embase_groundtruth_df[['primary_title', 'notes_abstract']].isna().sum())
    embase_groundtruth_df['primary_title'] = embase_groundtruth_df['primary_title'].fillna(embase_groundtruth_df['title_2ndsearch'])
    embase_groundtruth_df['notes_abstract'] = embase_groundtruth_df['notes_abstract'].fillna(embase_groundtruth_df['abstract_2ndsearch'])

    # PubMed dataset
    print("\nPubMed Dataset:")
    pmed_groundtruth_df = pd.read_excel(apiretrieved_groundtruth_path, 
                                    sheet_name="api_results_pubmed", 
                                    engine='openpyxl', 
                                    dtype={'included_article_id': str, 'api_id_retrieved' : str})

    # Fix: Fill each column separately
    pmed_groundtruth_df['title'] = pmed_groundtruth_df['title'].fillna(pmed_groundtruth_df['title_2ndsearch'])
    pmed_groundtruth_df['abstract'] = pmed_groundtruth_df['abstract'].fillna(pmed_groundtruth_df['abstract_2ndsearch'])

    #merge api results with with ground truth dataset 
    fullgroundtruth_valid_apimerge_df = fullgroundtruth_valid_df.copy()

    #citation network size and OA_id 
    fullgroundtruth_valid_apimerge_df[['retrieved_oa_id','citation_network_size']] = fullgroundtruth_valid_df.join(
        oa_groundtruth_df.set_index('included_article_id')[['api_id_retrieved', 'citation_network_size']], 
        on='included_article_id'
    )[['api_id_retrieved','citation_network_size']]
    #embase id 
    fullgroundtruth_valid_apimerge_df['retrieved_embase_id'] = fullgroundtruth_valid_df.join(
        embase_groundtruth_df.set_index('included_article_id')['api_id_retrieved'],
        on='included_article_id'
    )['api_id_retrieved'].apply(lambda x: str(x) if pd.notna(x) else pd.NA) #make sure this is a string
    #pubmed id 
    fullgroundtruth_valid_apimerge_df['retrieved_pubmed_id'] = fullgroundtruth_valid_df.join(
        pmed_groundtruth_df.set_index('included_article_id')['api_id_retrieved'],
        on='included_article_id'
    )['api_id_retrieved'].apply(lambda x: str(x) if pd.notna(x) else pd.NA)

    #fill in title and abstract 
    # Initialize title and abstract columns as empty
    fullgroundtruth_valid_apimerge_df.set_index('included_article_id', inplace=True)
    fullgroundtruth_valid_apimerge_df[['title', 'abstract']] = pd.NA
    # Create a dictionary mapping source dataframes to their column names
    source_mappings = [
        (oa_groundtruth_df, {'title': 'title', 'abstract': 'abstract'}),
        (embase_groundtruth_df, {'title': 'primary_title', 'abstract': 'notes_abstract'}),
        (pmed_groundtruth_df, {'title': 'title', 'abstract': 'abstract'})
    ]

    # Fill data from all sources in one pass
    for source_df, columns in source_mappings:
        source_data = source_df.set_index('included_article_id')[list(columns.values())]
        source_data.columns = ['title', 'abstract']  # Standardize column names
        fullgroundtruth_valid_apimerge_df.update(source_data)

    missing_titles = fullgroundtruth_valid_apimerge_df['title'].isna().sum()
    missing_abstracts = fullgroundtruth_valid_apimerge_df['abstract'].isna().sum()
    print(f"Rows with missing titles: {missing_titles}")
    print(f"Rows with missing abstracts: {missing_abstracts}")
    print(f"Total rows with either missing: {fullgroundtruth_valid_apimerge_df[['title', 'abstract']].isna().any(axis=1).sum()}")

    #clean up html tags and other weird artefacts 
    fullgroundtruth_valid_apimerge_df['title'] = fullgroundtruth_valid_apimerge_df['title'].apply(lambda x: clean_html_tags(x) if pd.notna(x) else x)
    fullgroundtruth_valid_apimerge_df['abstract'] = fullgroundtruth_valid_apimerge_df['abstract'].apply(lambda x: clean_html_tags(x) if pd.notna(x) else x)

    #save for later use (evaluation)
    fullgroundtruth_valid_apimerge_df.to_excel(output_dir / 'fullgroundtruth_valid_apimerge_df.xlsx', index=False)
    #also save as parquet to preserve data types 
    fullgroundtruth_valid_apimerge_df.to_parquet(output_dir / 'fullgroundtruth_valid_apimerge_df.parquet')
    return fullgroundtruth_valid_apimerge_df

groundtruth_setup()
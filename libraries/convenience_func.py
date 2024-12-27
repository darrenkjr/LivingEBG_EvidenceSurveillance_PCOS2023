import pandas as pd 
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import html
from pathlib import Path

class ConvenienceFunc: 

    def __init__(self, logger = None): 
        self.logger = logger

    @staticmethod
    def date_range_generator(start_date = '1950-01-01', end_date = '2022-12-31', interval_years = 2):
        '''
        Generate a list of non-overlapping date ranges, one year apart

        Args:
            start_date (str): Start date in the format 'YYYY-MM-DD'
            end_date (str): End date in the format 'YYYY-MM-DD'

        Returns:
            list: List of dictionaries with keys 'start_date' and 'end_date' for each range 
        '''
        date_ranges = []
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current_date < end:
            # Calculate the end of this range (either next year - 1 day, or the final end date)
            range_end = min(
                datetime(current_date.year + interval_years, current_date.month, current_date.day) - timedelta(days=1),
                end
            )
            
            date_ranges.append({
                'start': current_date.strftime('%Y-%m-%d'),
                'end': range_end.strftime('%Y-%m-%d')
            })
            
            # Move to start of next range
            current_date = range_end + timedelta(days=1)
            
        return date_ranges

    @staticmethod 
    def clean_html_tags(text):

        text = html.unescape(text)

        soup = BeautifulSoup(text, 'html.parser')
        clean_text = soup.get_text(separator = ' ' ,strip = True)
        clean_text = ' '.join(clean_text.split())
        return clean_text 

    def groundtruth_setup(self):    
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
        
        fullgroundtruth_valid_apimerge_df['title'] = fullgroundtruth_valid_apimerge_df['title'].apply(lambda x: self.clean_html_tags(x) if pd.notna(x) else x)
        fullgroundtruth_valid_apimerge_df['abstract'] = fullgroundtruth_valid_apimerge_df['abstract'].apply(lambda x: self.clean_html_tags(x) if pd.notna(x) else x)

        #save for later use (evaluation)
        fullgroundtruth_valid_apimerge_df.to_excel(output_dir / 'fullgroundtruth_valid_apimerge_df.xlsx', index=False)
        #also save as parquet to preserve data types 
        fullgroundtruth_valid_apimerge_df.to_parquet(output_dir / 'fullgroundtruth_valid_apimerge_df.parquet')
        return fullgroundtruth_valid_apimerge_df  
    
    def goldset_setup(self):
        current_dir = Path(__file__).parent
        dataset_dir = current_dir.parent / 'dataset'
        groundtruth_path = dataset_dir / 'fullgroundtruth_valid_apimerge_df.parquet'
        #setup gold set articles to inform vector search and generating topics for openalex search 
        fullgroundtruth_valid_apimerge_df = pd.read_parquet(groundtruth_path)
        fullgroundtruth_valid_apimerge_df['year_pub_extract'] = pd.to_numeric(
            fullgroundtruth_valid_apimerge_df['year_pub_extract'], 
            errors='coerce'
        ).astype('Int64')  

        fullgroundtruth_valid_apimerge_df.reset_index(inplace = True)

        #firstly for sr updates
        _srupdate = fullgroundtruth_valid_apimerge_df.query('sr_update == "Y" & title.notna() & abstract.notna()').copy()
        _newsr = fullgroundtruth_valid_apimerge_df.query('sr_update != "Y" & title.notna() & abstract.notna()').copy()
        srupdate_goldset = _srupdate.query('year_pub_extract <= searchstrat_year_start').copy()

        #now, new SRs - choose random as long sa title and abstracts are not empty
        newsr_goldset = _newsr.groupby('question_id').apply(
            lambda x: x.sample(
                n=min(3, len(x)),  # x is already filtered for non-empty title/abstract
                replace=False,
                random_state=42
            )
        ).reset_index(drop=True)

        combined_goldset = pd.concat([srupdate_goldset, newsr_goldset], ignore_index = True)
        #export 
        combined_goldset.to_parquet(dataset_dir / 'combined_goldset.parquet')
        combined_goldset.to_excel(dataset_dir / 'combined_goldset.xlsx', index=False)
        return combined_goldset

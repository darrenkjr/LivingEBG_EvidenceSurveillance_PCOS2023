import pandas as pd 
import os 

#import PCOS dataset, and extract appropriate questions 
rq_dataset_path = '../dataset/_superseded/PCOS_Guideline_Dataset.xlsm'
apiretrieved_groundtruth_path = '../dataset/_superseded/goldset_api_retrieved_final.xlsx'
unsucessful_rob_path = '../dataset/_superseded/all_unsuccessful_nodupe_rob.csv'

#appropriate rqs that fit inclusion criteria  
_df = pd.read_excel(os.path.join(os.path.dirname(__file__), rq_dataset_path), sheet_name='rq_evidence_review', engine='openpyxl', dtype={'question_id': str})
#extract valid rqs 
pcosrq_valid_df = _df.query('evidence_review_type == "SR" and included_num >= 5').copy()[['GDG', 'question_id', 'Topic', 'Question', 'sr_update', 'included_num', 'searchstrat_year_start', 'searchstrat_year_end']]
#extract full ground truth dataset 

fullgroundtruth_full_df = pd.read_excel(os.path.join(os.path.dirname(__file__), rq_dataset_path), sheet_name='included_articles', engine='openpyxl', dtype={'question_id': str, 'included_article_id': str}).copy()
#read in ROB (from collection of uncessfully retrieved articles) 
unsucessful_rob_df = pd.read_csv(os.path.join(os.path.dirname(__file__), unsucessful_rob_path), dtype={'included_article_id': str})
fullgroundtruth_full_df['assessed_rob'] = fullgroundtruth_full_df.join(
    unsucessful_rob_df.set_index('included_article_id')['rob'],
    on = 'included_article_id'
)['rob']
# Filter fullgroundtruth_full_df to only include rows where question_id matches those in pcosrq_valid_df
fullgroundtruth_valid_df = fullgroundtruth_full_df.query('question_id in @pcosrq_valid_df["question_id"]')[['GDG', 'question_id', 'included_article_id', 'included_reference', 'year_pub_extract', 'author_year_format', 'assessed_rob']]

#ground truth dataset (not linked to API results )
oa_groundtruth_df = pd.read_excel(os.path.join(os.path.dirname(__file__), apiretrieved_groundtruth_path), sheet_name = "api_results_oa", engine='openpyxl', dtype={'included_article_id': str})
embase_groundtruth_df = pd.read_excel(os.path.join(os.path.dirname(__file__), apiretrieved_groundtruth_path), sheet_name = "api_results_embase", engine='openpyxl', dtype={'included_article_id': str})
pmed_groundtruth_df = pd.read_excel(os.path.join(os.path.dirname(__file__), apiretrieved_groundtruth_path), sheet_name = "api_results_pubmed", engine='openpyxl', dtype={'included_article_id': str})

 
#merge with api result ids 
fullgroundtruth_valid_apimerge_df = fullgroundtruth_valid_df.copy()
fullgroundtruth_valid_apimerge_df['retrieved_oa_id'] = fullgroundtruth_valid_df.join(
    oa_groundtruth_df.set_index('included_article_id')['api_id_retrieved'], 
    on='included_article_id'
)[['api_id_retrieved','citations','references']]
fullgroundtruth_valid_apimerge_df['retrieved_embase_id'] = fullgroundtruth_valid_df.join(
    embase_groundtruth_df.set_index('included_article_id')['api_id_retrieved'],
    on='included_article_id'
)['api_id_retrieved']
fullgroundtruth_valid_apimerge_df['retrieved_pubmed_id'] = fullgroundtruth_valid_df.join(
    pmed_groundtruth_df.set_index('included_article_id')['api_id_retrieved'],
    on='included_article_id'
)['api_id_retrieved']

#check recall of each api 
oa_max_recall = len(fullgroundtruth_valid_apimerge_df[fullgroundtruth_valid_apimerge_df['retrieved_oa_id'].notna()])/len(fullgroundtruth_valid_df)*100
embase_max_recall = len(fullgroundtruth_valid_apimerge_df[fullgroundtruth_valid_apimerge_df['retrieved_embase_id'].notna()])/len(fullgroundtruth_valid_df)*100
pmed_max_recall = len(fullgroundtruth_valid_apimerge_df[fullgroundtruth_valid_apimerge_df['retrieved_pubmed_id'].notna()])/len(fullgroundtruth_valid_df)*100
print(f"OA Max recall: {oa_max_recall:.1f}%, {len(fullgroundtruth_valid_apimerge_df[fullgroundtruth_valid_apimerge_df['retrieved_oa_id'].notna()])}/{len(fullgroundtruth_valid_df)}")
print(f"Embase Max recall: {embase_max_recall:.1f}%, {len(fullgroundtruth_valid_apimerge_df[fullgroundtruth_valid_apimerge_df['retrieved_embase_id'].notna()])}/{len(fullgroundtruth_valid_df)}")
print(f"PubMed Max recall: {pmed_max_recall:.1f}%, {len(fullgroundtruth_valid_apimerge_df[fullgroundtruth_valid_apimerge_df['retrieved_pubmed_id'].notna()])}/{len(fullgroundtruth_valid_df)}")

#save for later use (evaluation)
fullgroundtruth_valid_apimerge_df.to_excel(os.path.join(os.path.dirname(__file__), '../dataset/fullgroundtruth_valid_apimerge_df.xlsx'), index=False)


#setup gold set articles to inform vector search and genreating topics for openalex search 
goldset_articles_df = oa_groundtruth_df.
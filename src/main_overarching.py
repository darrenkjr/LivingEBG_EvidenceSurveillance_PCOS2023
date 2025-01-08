from embmedline_ris_pipeline import embmedline_ris_pipeline
from openalex_overarching_search import oa_overarching_search
from openalex_keywordsearch import oa_keywordsearch_dev
from pubmed_overarching_search import pubmed_overarching_search
import asyncio
import pandas as pd
import dotenv
dotenv.load_dotenv()
import os
from pathlib import Path



def main(): 
    #parse ris files for overarhcing
    total_evalmetrics_df = pd.DataFrame()
    eval_results_path = Path(__file__).parent / 'evaluation_results' / 'overall' / 'overarching'
    eval_results_path.mkdir(parents=True, exist_ok=True)

    pubmed_overarching_search_cls = pubmed_overarching_search()
    pubmed_query = os.getenv('pubmed_query')
    evalmetrics_df = asyncio.run(pubmed_overarching_search_cls.pubmed_ovarching_search_eval_pipeline(pubmed_query))
    total_evalmetrics_df = pd.concat([total_evalmetrics_df, evalmetrics_df], ignore_index=True)
    
    #openalex overarching, topic 
    openalex_overarching_search_cls = oa_overarching_search()
    evalmetrics_df = asyncio.run(openalex_overarching_search_cls.oa_overarching_search_eval_pipeline())
    total_evalmetrics_df = pd.concat([total_evalmetrics_df, evalmetrics_df], ignore_index=True)

    
    #openalex keyword search 
    oa_keywordsearch_cls = oa_keywordsearch_dev()
    evalmetrics_df = asyncio.run(oa_keywordsearch_cls.oa_kw_search_eval_pipeline())
    total_evalmetrics_df = pd.concat([total_evalmetrics_df, evalmetrics_df], ignore_index=True)
    
    #pubmed overarching 



    #parse ris files
    ris_database = ['embase', 'medline']
    for database in ris_database: 
        embmedline_ris_pipeline_cls = embmedline_ris_pipeline(database, search_type = 'overarching')
        evalmetrics_df = embmedline_ris_pipeline_cls.ris_eval_pipeline()
        total_evalmetrics_df = pd.concat([total_evalmetrics_df, evalmetrics_df], ignore_index=True)

    #save total evalmetrics_df
    total_evalmetrics_df.to_csv(f'{eval_results_path}/overall_evalmetrics_df.csv', index=False)


if __name__ == '__main__': 
    main()



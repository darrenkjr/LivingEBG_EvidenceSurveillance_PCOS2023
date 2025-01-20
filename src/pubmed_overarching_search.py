import pandas as pd 
import asyncio 
from libraries.pubmed_client import PubMedClient
import pyarrow as pa
import pyarrow.parquet as pq
from libraries.eval import search_evaluation
from pathlib import Path
from libraries.logging_config import LoggerConfig

class pubmed_overarching_search: 

    def __init__(self): 

        self.logger = LoggerConfig.setup_logger(logger_name='pubmed_overarching_search')
        self.results_path = Path(__file__).parent / 'search_results' / 'pubmed' / 'overarching' 
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.pmid_path = self.results_path / f'pubmed_booleankw_ovearching_search_results_pmid.parquet'
        self.article_details_path = self.results_path / f'pubmed_booleankw_ovearching_search_results.parquet'
        self.search_eval_cls = search_evaluation('pubmed', 'overarching', vector_search = False,logger=self.logger)
        self.consolidated_results_path = self.search_eval_cls.consolidated_results_path

    async def pubmed_overarching_search(self, query: str): 
        pmed_client = PubMedClient(logger = self.logger)
        self.logger.info(f'Running boolean keyword overarching search over pubmed.. with query : {query}')
        #perform check for pmid 
        if self.pmid_path.exists(): 
            self.logger.info(f'PMID list already exists, skipping initial search, and retrieving article details')
            results_pmid = pd.read_parquet(self.pmid_path)
        else: 
            results_pmid = await pmed_client.get_pubmed_search_results_batching(query)
            #save interim pmid list 
            results_pmid.to_parquet(self.results_path / f'pubmed_booleankw_ovearching_search_results_pmid.parquet')
        results_pmid_list = results_pmid['pmid'].to_list()
        #perform check for article details 
        if self.article_details_path.exists() and self.article_details_path.stat().st_size > 0: 
            self.logger.info(f'Article details already exist, skipping retrieval')
            results_df = pd.read_parquet(self.article_details_path)
        else: 
            self.logger.info('Retrieving article details for PMID list')
            results_df = await pmed_client.get_pubmed_article_details(results_pmid_list)
            results_df['publication_year'] = pd.to_numeric(results_df['publication_year'], errors='coerce')
            results_df['publication_year'] = results_df['publication_year'].astype('Int64')
            results_df.to_parquet(self.results_path / f'pubmed_booleankw_ovearching_search_results.parquet')


    async def pubmed_ovarching_search_eval_pipeline(self, query: str): 
        
        if self.consolidated_results_path.exists(): 
            self.logger.info(f'Consolidated results already exist')
            evalmetrics_df = self.search_eval_cls.run_eval_pipeline()
        elif not self.consolidated_results_path.exists(): 
            await self.pubmed_overarching_search(query)
            evalmetrics_df = self.search_eval_cls.run_eval_pipeline()
        return evalmetrics_df

if __name__ == '__main__': 
    pubmed_overarching_search_cls = pubmed_overarching_search()
    query = '(Polycystic Ovary Syndrome[mh] OR "polycystic ovar*"[tiab] OR "poly-cystic ovar*"[tiab] OR PCOS[tiab] OR PCOD[tiab] OR leventhal[tiab] OR Anovulation[mh] OR anovulat*[tiab] OR oligo-ovulat*[tiab] OR oligoovulat*[tiab] OR (ovar*[tiab] AND (sclerocystic[tiab] OR polycystic[tiab] OR poly-cystic[tiab] OR degenerate*[tiab] OR hyperandrogen*[tiab] OR hyperandrogen*[tiab]))) NOT (Animals[mh] NOT Humans[mh])'
    evalmetrics_df = asyncio.run(pubmed_overarching_search_cls.pubmed_ovarching_search_eval_pipeline(query))

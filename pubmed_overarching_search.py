import pandas as pd 
from datetime import datetime, timedelta
import asyncio 
from libraries.pubmed_client import PubMedClient
import pyarrow as pa
import pyarrow.parquet as pq
from convenience_scripts.openalex_topicsearch_builder import oa_topicsearch_builder
from convenience_scripts.eval import search_evaluation
from pathlib import Path
from libraries.logging_config import LoggerConfig

async def pubmed_overarching_search(query: str, logger = None): 
        pmed_client = PubMedClient(logger = logger)
        logger.info(f'Running boolean keyword overarching search over pubmed.. with query : {query}')
        current_dir = Path(__file__).parent
        results_path = current_dir / 'search_results' / 'pubmed' / 'overarching' 
        #create results path if it doesn't exist 
        results_path.mkdir(parents=True, exist_ok=True)
        results_df = await pmed_client.get_pubmed_search_results_batching(query)
        results_df.to_parquet(results_path / f'pubmed_booleankw_ovearching_search_results.parquet')

async def pubmed_ovarching_search_eval_pipeline(query: str): 
    logger = LoggerConfig.setup_logger(logger_name='pubmed_overarching_search')
    await pubmed_overarching_search(query, logger = logger)
    search_eval_cls = search_evaluation('pubmed', 'overarching', vector_search = False,logger=logger)
    search_eval_cls.run_eval_pipeline()

if __name__ == '__main__': 
    query = '(Polycystic Ovary Syndrome[mh] OR "polycystic ovar*"[tiab] OR "poly-cystic ovar*"[tiab] OR PCOS[tiab] OR PCOD[tiab] OR leventhal[tiab] OR Anovulation[mh] OR anovulat*[tiab] OR oligo-ovulat*[tiab] OR oligoovulat*[tiab] OR (ovar*[tiab] AND (sclerocystic[tiab] OR polycystic[tiab] OR poly-cystic[tiab] OR degenerate*[tiab] OR hyperandrogen*[tiab] OR hyperandrogen*[tiab]))) NOT (Animals[mh] NOT Humans[mh])'
    asyncio.run(pubmed_ovarching_search_eval_pipeline(query))

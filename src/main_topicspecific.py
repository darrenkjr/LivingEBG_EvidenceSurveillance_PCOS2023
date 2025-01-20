from embmedline_ris_pipeline import embmedline_ris_pipeline
from libraries.vector_search import vector_search_implementation
import asyncio
import pandas as pd
import dotenv
dotenv.load_dotenv()
import os
from pathlib import Path

def main():
    total_evalmetrics_df = pd.DataFrame()
    eval_results_path = Path(__file__).parent / 'evaluation_results' / 'overall' 
    eval_results_path.mkdir(parents=True, exist_ok=True) 

    ris_database = ['embase', 'medline']
    for database in ris_database: 
        embmedline_ris_pipeline_cls = embmedline_ris_pipeline(database, search_type = 'topic_specific')
        evalmetrics_df = embmedline_ris_pipeline_cls.ris_eval_pipeline()
        total_evalmetrics_df = pd.concat([total_evalmetrics_df, evalmetrics_df], ignore_index=True)


    include_header = True
    total_evalmetrics_df.to_csv(
        eval_results_path / 'overall_evalmetrics_df.csv',  # path is the first positional argument
        mode='a',  # append mode
        header=not (eval_results_path / 'overall_evalmetrics_df.csv').exists() if include_header else False,
        index=False
    )


if __name__ == '__main__': 
    main()


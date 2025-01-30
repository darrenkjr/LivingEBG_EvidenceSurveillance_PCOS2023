from embmedline_ris_pipeline import embmedline_ris_pipeline
from openalex_overarching_search import oa_overarching_search
from openalex_keywordsearch import oa_keywordsearch_dev
from pubmed_overarching_search import pubmed_overarching_search
from openpyxl import load_workbook
import asyncio
import pandas as pd
import dotenv
dotenv.load_dotenv()
import os
from pathlib import Path
from pandas import ExcelWriter


def main(): 
    #parse ris files for overarhcing
    total_evalmetrics_df = pd.DataFrame()
    eval_results_path = Path(__file__).parent / 'evaluation_results' / 'overall' 
    eval_results_path.mkdir(parents=True, exist_ok=True)

    pubmed_overarching_search_cls = pubmed_overarching_search()
    pubmed_query = '(Polycystic Ovary Syndrome[mh] OR "polycystic ovar*"[tiab] OR "poly-cystic ovar*"[tiab] OR PCOS[tiab] OR PCOD[tiab] OR leventhal[tiab] OR Anovulation[mh] OR anovulat*[tiab] OR oligo-ovulat*[tiab] OR oligoovulat*[tiab] OR (ovar*[tiab] AND (sclerocystic[tiab] OR polycystic[tiab] OR poly-cystic[tiab] OR degenerate*[tiab] OR hyperandrogen*[tiab] OR hyperandrogen*[tiab]))) NOT (Animals[mh] NOT Humans[mh])'
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
    
    #parse ris files
    ris_database = ['embase', 'medline']
    for database in ris_database: 
        embmedline_ris_pipeline_cls = embmedline_ris_pipeline(database, search_type = 'overarching')
        evalmetrics_df = embmedline_ris_pipeline_cls.ris_eval_pipeline()
        total_evalmetrics_df = pd.concat([total_evalmetrics_df, evalmetrics_df], ignore_index=True)

    sheet_name = 'ovearching_no_vs'
    excel_path = eval_results_path / 'overall_evalmetrics_df.xlsx'
    if excel_path.exists():
        # Load existing workbook
        book = load_workbook(excel_path)
        
        with ExcelWriter(excel_path, mode='a', engine='openpyxl') as writer:
            if sheet_name in book.sheetnames:
                # Get the last row in existing sheet
                sheet = book[sheet_name]
                start_row = sheet.max_row
                
                total_evalmetrics_df.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    startrow=start_row,  # Start after last row
                    index=False,
                    header=False  # Don't write headers again
                )
            else:
                # New sheet, include headers
                total_evalmetrics_df.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False
                )
    else:
        # New file
        total_evalmetrics_df.to_excel(
            excel_path,
            sheet_name=sheet_name,
            index=False
        )

if __name__ == '__main__': 
    main()



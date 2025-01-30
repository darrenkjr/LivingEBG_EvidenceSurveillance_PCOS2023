from embmedline_ris_pipeline import embmedline_ris_pipeline
# from libraries.vector_search import vector_search_implementation
import asyncio
import pandas as pd
import dotenv
dotenv.load_dotenv()
import os
from pathlib import Path
from pandas import ExcelWriter
from openpyxl import load_workbook

def main():
    total_evalmetrics_df = pd.DataFrame()
    eval_results_path = Path(__file__).parent / 'evaluation_results' / 'overall' 
    eval_results_path.mkdir(parents=True, exist_ok=True) 

    ris_database = ['embase', 'medline']
    for database in ris_database: 
        embmedline_ris_pipeline_cls = embmedline_ris_pipeline(database, search_type = 'topic_specific')
        evalmetrics_df = embmedline_ris_pipeline_cls.ris_eval_pipeline()
        total_evalmetrics_df = pd.concat([total_evalmetrics_df, evalmetrics_df], ignore_index=True)


    sheet_name = 'topic_specific_no_vs'
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


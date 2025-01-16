import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from pathlib import Path

load_dotenv()

class sql_data_migration:

    def __init__(self, db_name, db_user, db_pwd):
        try:
            # Create a SQLAlchemy engine
            self.engine = create_engine(f'postgresql://{db_user}:{db_pwd}@localhost:5432/{db_name}')
            self.gdg_data_path = Path(__file__).parent.parent / 'dataset' / '_superseded' / 'PCOS_Guideline_Dataset_checked.xlsm'
            self.ground_truth_data_path = Path(__file__).parent.parent / 'dataset' / 'fullgroundtruth_valid_apimerge_df.parquet'
            
            self.database_mapping = {
                'database_id' : [1, 2, 3, 4], 
                'database_name' : ['pubmed', 'medline', 'embase', 'openalex']
            }

            self.search_type_mapping = {
                'search_type_id' : [1, 2], 
                'name' : ['overarching', 'topic-specific']
            }

            self.search_strategy_type_mapping = {
                'search_strategy_type_id' : [1, 2], 
                'name' : ['boolean-kw', 'openalex-topic-search']
            }
        except Exception as e:
            print(f"Error connecting to database: {e}")

        # Check tables
        self._check_tables()

    def _check_tables(self):
        with self.engine.connect() as conn:
            result = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';")
            tables = result.fetchall()
        return tables

    def fill_reference_tables(self):

        databases_df = pd.DataFrame(self.database_mapping)
        search_types_df = pd.DataFrame(self.search_type_mapping)
        search_strategy_types_df = pd.DataFrame(self.search_strategy_type_mapping)

        tabledf_dct = {
            'databases': databases_df,
            'search_types': search_types_df,
            'search_strategy_types': search_strategy_types_df
        }

        for table_name, df in tabledf_dct.items():
            df.to_sql(table_name, self.engine, if_exists='replace', index=False)

    def _load_gdg_data(self):
        _df = pd.read_excel(self.gdg_data_path, sheet_name='rq_evidence_review')
        gdg_extract = _df[['GDG', 'Topic']].copy()
        evidence_review_extract = _df[['question_id', 'Question', 'evidence_review_type', 'sr_update', 'sr_new', 'included_num']].copy()
        # edit evidence review types

        er_map = {
            'Y': 1,
            'N': 2,
        }

        evidence_review_extract['evidence_review_type_sql_insert'] = evidence_review_extract['evidence_review_type'].apply(lambda x: er_map[x] if pd.notnull(x) else 3)

    def _load_ground_truth_data(self):
        _df = pd.read_parquet(self.ground_truth_data_path)
import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy import text
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from libraries.logging_config import LoggerConfig

load_dotenv()

class sql_data_migration:

    def __init__(self, db_name, db_user, db_pwd, logger = None):
        try:
            # Create a SQLAlchemy engine
            self.logger = logger
            self.engine = create_engine(f'postgresql://{db_user}:{db_pwd}@localhost:5432/{db_name}')
            self.logger.info(f"Connecting to database {db_name}")
            with self.engine.connect() as conn:
                    #check connections 
                try: 
                    conn.execute(text("SELECT 1"))  
                    self.logger.info(f"Connection successful")
                except Exception as e:
                    self.logger.error(f"Error connecting to database: {e}")

            self.gdg_data_path = Path(__file__).parent.parent / 'dataset' / '_superseded' / 'PCOS_Guideline_Dataset_checked.xlsm'
            self.ground_truth_data_path = Path(__file__).parent.parent / 'dataset' / 'fullgroundtruth_valid_apimerge_df.parquet'
            
            self.database_mapping = {
                'database_id' : [1, 2, 3, 4], 
                'database_name' : ['pubmed', 'medline', 'embase', 'openalex'],
                'free_api_available' : [True, False, True, True]
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
            self.logger.info(f"Checking existing tables")
            result = conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public';"))
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

        self.logger.info(f"Filling reference tables")
        for table_name, df in tabledf_dct.items():
            with self.engine.connect() as conn:
                #check 
                id_col = df.columns[0]
                _ = conn.execute(text(f"SELECT {id_col} from {table_name}"))
                existing_ids = [row[0] for row in _.fetchall()]
            #conduct check for new ids 
            insert_df = df[~df[id_col].isin(existing_ids)]
            if not insert_df.empty: 
                self.logger.info(f"New IDs found for {table_name}, inserting into database") 
                #insert new ids 
                insert_df.to_sql(table_name, self.engine, if_exists='append', index=False)
            else:
                self.logger.info(f"No new IDs found for {table_name}, moving on")


    def migrate_gdg_data(self):
        _df = pd.read_excel(self.gdg_data_path, sheet_name='rq_evidence_review')
        gdg_extract = _df[['GDG', 'Topic']].copy()
        gdg_extract.rename(columns = {'GDG' : 'gdg_id', 'Topic' : 'topic'}, inplace=True)
        gdg_extract.drop_duplicates(subset = 'gdg_id', inplace=True)
        self.logger.info(f"migrateing GDG data into database")

        #retrieve table column from sql and check columns 
        if self._df_sqltable_column_check(gdg_extract, 'gdgs') and self._prior_id_check(gdg_extract, 'gdg_id', 'gdgs'): 
            gdg_extract.to_sql('gdgs', self.engine, if_exists='append', index=False)

        evidence_review_extract = _df[['GDG', 'question_id', 'Question', 'evidence_review_type', 'sr_update', 'sr_new', 'included_num']].copy()
        # edit evidence review types

        _map = {
            'Y': 1,
            'N': 2,
        }

        _ermap = {
            1 : 'systematic review update', 
            2 : 'new systematic review',
            3 : 'narrative reivew'
        }

        evidence_review_extract['evidence_review_type_sql_insert'] = evidence_review_extract['sr_update'].apply(lambda x: _map[x] if pd.notnull(x) else 3)
        evidence_review_extract.drop(columns = ['evidence_review_type'], inplace=True)
        evidence_review_extract['evidence_review_type'] = evidence_review_extract['evidence_review_type_sql_insert'].apply(lambda x: _ermap[x])
        evidence_review_extract.drop(columns = ['sr_update', 'sr_new', 'evidence_review_type_sql_insert'], inplace=True)
        evidence_review_extract['question_id'] = evidence_review_extract['question_id'].astype(str).str.strip()
        evidence_review_extract.rename(columns = {'GDG' : 'gdg_id', 'Question' : 'question', 'question_id' : 'evidence_review_id'}, inplace=True)
        
        if self._df_sqltable_column_check(evidence_review_extract, 'evidence_reviews') and self._prior_id_check(evidence_review_extract, 'evidence_review_id', 'evidence_reviews'): 
            evidence_review_extract.to_sql('evidence_reviews', self.engine, if_exists='append', index=False)

    def migrate_ground_truth_data(self):
        _ = pd.read_parquet(self.ground_truth_data_path)
        _df = _.copy().reset_index()
        

        ground_truth_df = _df[['included_article_id', 'question_id', 'included_reference', 'author_year_format', 'year_pub_extract', 'assessed_rob', 'retrieved_oa_id', 'retrieved_embase_id', 'retrieved_pubmed_id', 'title', 'abstract']].copy()
        ground_truth_df.rename(columns = {
            'question_id' : 'evidence_review_id', 
            'included_article_id' : 'ground_truth_article_id', 
            'year_pub_extract' : 'extracted_publication_year', 
            }, inplace=True)
        
        if self._df_sqltable_column_check(ground_truth_df, 'ground_truth_articles') and self._prior_id_check(ground_truth_df, 'ground_truth_article_id', 'ground_truth_articles'): 
            ground_truth_df.to_sql('ground_truth_articles', self.engine, if_exists='append', index=False)
        else: 
            self.logger.error(f"Error inserting ground truth articles into database, check column names")



    def _df_sqltable_column_check(self, df, table_name):
        df_columns = set(df.columns.tolist())
        with self.engine.connect() as conn:
            result = conn.execute(text(f"SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table_name}'"))
            existing_columns = {row[0] for row in result.fetchall()}

        if df_columns == existing_columns:
            return True
        else:
            missing_in_df = existing_columns - df_columns
            extra_in_df = df_columns - existing_columns

            if missing_in_df:
                self.logger.error(f"Missing columns in DataFrame for table '{table_name}': {missing_in_df}")
            if extra_in_df:
                self.logger.error(f"Extra columns in DataFrame for table '{table_name}': {extra_in_df}")

            return False
    


    def _prior_id_check(self, df, idcol, table_name):
        with self.engine.connect() as conn:
            result = conn.execute(text(f"SELECT {idcol} from {table_name}"))
            existing_ids = [row[0] for row in result.fetchall()]

        if df[idcol].isin(existing_ids).any():
            self.logger.info(f"Some prior IDs found for {table_name}, moving on")
            return False
        else:
            self.logger.info(f"No prior IDs found for {table_name}, inserting new IDs")
            return True
        
    def migrate_search_result_data(self, table_name): 

        pass 

    def migrate_search_strategies(self, table_name): 
        
        pass 

    

if __name__ == '__main__':
    db_name = os.getenv('DB_NAME')
    db_user = os.getenv('DB_USER')
    db_pwd = os.getenv('DB_PWD')
    logger = LoggerConfig.setup_logger(logger_name = 'sql_data_migration')
    sql_instance = sql_data_migration(db_name, db_user, db_pwd, logger)
    sql_instance.fill_reference_tables()
    sql_instance.migrate_gdg_data()
    sql_instance.migrate_ground_truth_data()

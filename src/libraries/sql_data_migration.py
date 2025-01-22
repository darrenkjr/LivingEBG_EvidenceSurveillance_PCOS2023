import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()
from sqlalchemy import create_engine
from sqlalchemy import text
from pathlib import Path
from sql_tables_generation import create_database, create_basic_tables
import sys
sys.path.append(str(Path(__file__).parent.parent))
from libraries.logging_config import LoggerConfig


class sql_data_migration:

    def __init__(self, db_name, db_user, db_pwd, db_host, logger = None):
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

            self.database_id_mapping = {
                'database' : ['pubmed', 'medline', 'embase', 'openalex'], 
                'idcol' : ['pmid', 'id', 'accession_number', 'id']
            }

            self.consolidated_results_path = Path(__file__).parent.parent / 'consolidated_results'

            self.current_article_id  = 1

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
        #add overall gdg 
        gdg_extract.loc[len(gdg_extract)] = {'gdg_id': 6, 'topic': 'overall'}
        self.logger.info(f"migrateing GDG data into database")

        #retrieve table column from sql and check columns 
        if self._df_sqltable_column_check(gdg_extract, 'gdgs'): 
            filered_df = self._prior_id_check(gdg_extract, 'gdg_id', 'gdgs') 
            with self.engine.begin() as conn:
                filered_df.to_sql('gdgs', conn, if_exists='append', index=False)

        evidence_review_extract = _df[['GDG', 'question_id', 'Question', 'evidence_review_type', 'sr_update', 'sr_new', 'included_num', 'searchstrat_year_start', 'searchstrat_year_end']].copy()
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
        #add extra evidence review which sifnigies overall 
        overall_dct = {
            'evidence_review_id' : 'overall', 
            'gdg_id' : 6, 
            'question' : 'overall', 
            'evidence_review_type' : 'new systematic review'
        }
        evidence_review_extract.loc[len(evidence_review_extract)] = overall_dct
        self.er_extract = evidence_review_extract.copy()
        evidence_review_extract.drop(columns = ['searchstrat_year_start', 'searchstrat_year_end'], inplace=True)
        if self._df_sqltable_column_check(evidence_review_extract, 'evidence_reviews'): 
            evidence_review_extract_filtered = self._prior_id_check(evidence_review_extract, 'evidence_review_id', 'evidence_reviews')
            if not evidence_review_extract_filtered.empty: 
                with self.engine.begin() as conn:
                    self.logger.info(f"Inserting new evidence reviews into database, number of entries detected: {len(evidence_review_extract_filtered)}")
                    evidence_review_extract_filtered.to_sql('evidence_reviews', conn, if_exists='append', index=False)
            else: 
                self.logger.info(f"No new evidence reviews to insert")



    def migrate_ground_truth_data(self):
        _ = pd.read_parquet(self.ground_truth_data_path)
        _df = _.copy().reset_index()
        

        ground_truth_df = _df[['included_article_id', 'question_id', 'included_reference', 'author_year_format', 'year_pub_extract', 'assessed_rob', 'retrieved_oa_id', 'retrieved_embase_id', 'retrieved_pubmed_id', 'title', 'abstract']].copy()
        ground_truth_df.rename(columns = {
            'question_id' : 'evidence_review_id', 
            'included_article_id' : 'ground_truth_article_id', 
            'year_pub_extract' : 'extracted_publication_year', 
            }, inplace=True)
        
        if self._df_sqltable_column_check(ground_truth_df, 'ground_truth_articles'): 
            filtered_ground_truth_df = self._prior_id_check(ground_truth_df, 'ground_truth_article_id', 'ground_truth_articles') 
            if not filtered_ground_truth_df.empty: 
                with self.engine.begin() as conn:
                    self.logger.info(f"Inserting new ground truth articles into database, number of entries detected: {len(filtered_ground_truth_df)}")
                    filtered_ground_truth_df.to_sql('ground_truth_articles', conn, if_exists='append', index=False, method='multi')
            else: 
                self.logger.info(f"No new ground truth articles to insert")
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
                raise ValueError(f"Missing columns in DataFrame for table '{table_name}': {missing_in_df}")
            if extra_in_df:
                self.logger.error(f"Extra columns in DataFrame for table '{table_name}': {extra_in_df}")
                raise ValueError(f"Extra columns in DataFrame for table '{table_name}': {extra_in_df}")

            return False
    

    
    def _prior_id_check(self, df, idcol, table_name):
        with self.engine.connect() as conn:
            # Get existing IDs
            result = conn.execute(text(f"SELECT {idcol} FROM {table_name}"))
            existing_ids = [row[0] for row in result.fetchall()]
            
            if not existing_ids:  # If table is empty
                self.logger.info(f"No existing records in {table_name}")
                return df
                
            # Convert IDs to same type as DataFrame for comparison
            existing_ids = [type(df[idcol].iloc[0])(id) for id in existing_ids]
            
            # Filter out existing IDs
            filtered_df = df[~df[idcol].isin(existing_ids)].copy()
            
            self.logger.info(f"Found {len(filtered_df)} new records for {table_name}")
            return filtered_df
        


    def migrate_search_strategies(self): 

        table_name = 'search_strategies'
        database_insert = []
        search_types_insert = []
        search_strategy_types_insert = []
        searchstrat_year_start_insert = []
        searchstrat_year_end_insert = []
        evidence_review_id_insert = []
        search_strategy_id_insert = []


        self.filename_mapping = {}

        search_strategy_id_count = 1 
        for file in self.consolidated_results_path.iterdir():
            
            if str(file).endswith('.parquet'): 
                #split filename 
                self.filename_mapping[file.name] = {
                    'file_path': file,
                    'search_strategy_id': search_strategy_id_count
                }
                file_name = file.name.split('_')
                self.database_name = file_name[0]
                if self.database_name == 'oa': 
                    self.database_name = 'openalex'
                search_type_name = file_name[1]
                if search_type_name == 'topic': 
                    search_type = 'topic-specific'
                elif search_type_name == 'overarching': 
                    search_type = 'overarching'
                else: 
                    self.logger.error(f"Error: search type {search_type_name} not found")
                
                search_strategy_type_name = '_'.join([file_name[3],file_name[4]])
                if search_strategy_type_name == 'boolkw_search' or 'consolidated_boolkw': 
                    search_strategy_type = 'boolean-kw'
                elif search_strategy_type_name == 'topic_search': 
                    search_strategy_type = 'openalex-topic-search'

                if search_type == 'overarching': 
                    current_id = search_strategy_id_count
                    searchstrat_year_start_insert.append(1990)
                    searchstrat_year_end_insert.append(2020)
                    evidence_review_id_insert.append('overall')
                    search_strategy_id_insert.append(current_id )
                    database_insert.append(self.database_name)
                    search_types_insert.append(search_type)
                    search_strategy_types_insert.append(search_strategy_type)
                    search_strategy_id_count += 1
                    if file.name == 'oa_overarching_consolidated_topic_search_results.parquet': 
                        self.logger.warning(f"Skipping {file.name} as its too large")
                    else: 
                        search_result_df = pd.read_parquet(file)
                        self.migrate_search_result_articles(search_strategy_id_count, search_result_df)

                elif search_type == 'topic-specific': 
                    _df = pd.read_parquet(file)
                    _df.rename(columns = {
                        'question_id' : 'evidence_review_id'
                    }, inplace=True)
                    #fix evidence review id 
                    input_evidence_review_id_mapping = {
                        '4.2.4.3.combined' : '4.2/4.3', 
                        '1.4' : '1.4.1/1.4.2', 
                        '1.5' : '1.5.1/1.5.2', 
                        '2.1' : '2.1.1/2.1.2', 
                        '1.9.1.embase' : '1.9.1', 
                    }
                    _df['evidence_review_id'] = _df['evidence_review_id'].map(lambda x : input_evidence_review_id_mapping[x] if x in input_evidence_review_id_mapping else x)
                    evidence_review_id = pd.Series(_df['evidence_review_id'].unique())
                    #check that evidence review id is in the evidence review table 
                    with self.engine.connect() as conn: 
                        result = conn.execute(text("SELECT evidence_review_id FROM evidence_reviews"))
                        existing_evidence_review_ids = [row[0] for row in result.fetchall()]
                    if not set(evidence_review_id).issubset(set(existing_evidence_review_ids)): 
                        #extract evidence review ids that are not in the evidence review table 
                        missing_evidence_review_ids = set(evidence_review_id) - set(existing_evidence_review_ids)
                        self.logger.error(f"Error: Input search strategy assiociated with evidence review id: {missing_evidence_review_ids} not found in evidence review table")
                        raise ValueError(f"Error: Input search strategy assiociated with evidence review id: {missing_evidence_review_ids} not found in evidence review table")

                    # Create a temporary dataframe with ordered IDs
                    temp_df = pd.DataFrame({'evidence_review_id': evidence_review_id})

                    # Merge while preserving order
                    _er_extract = temp_df.merge(self.er_extract[['evidence_review_id', 'searchstrat_year_start', 'searchstrat_year_end']], 
                                            on='evidence_review_id', 
                                            how='left')
                    
                    n_entries = len(evidence_review_id)
                    search_strategy_id_insert.extend(range(search_strategy_id_count, search_strategy_id_count + n_entries))
                    search_strategy_id_count += n_entries
                    database_insert.extend([self.database_name] * n_entries)
                    search_types_insert.extend([search_type] * n_entries)
                    search_strategy_types_insert.extend([search_strategy_type] * n_entries)
                    searchstrat_year_start_insert.extend(_er_extract['searchstrat_year_start'].tolist())
                    searchstrat_year_end_insert.extend(_er_extract['searchstrat_year_end'].tolist())
                    evidence_review_id_insert.extend(_er_extract['evidence_review_id'].tolist())
                    #update search strategy id count 
                    search_strategy_id_count += n_entries
                    self._handle_topic_specific_search(search_strategy_id_insert, evidence_review_id_insert, file)

            #map database to ids 
        database_id = [
            self.database_mapping['database_id'][
                self.database_mapping['database_name'].index(db)
            ] for db in database_insert
        ]
        
        search_strategy_type_id = [
            self.search_strategy_type_mapping['search_strategy_type_id'][
                self.search_strategy_type_mapping['name'].index(st)
            ] for st in search_strategy_types_insert
        ]
        
        search_type_id = [
            self.search_type_mapping['search_type_id'][
                self.search_type_mapping['name'].index(st)
            ] for st in search_types_insert
        ]
        

        search_strategies_df = pd.DataFrame({
            'search_strategy_id' : search_strategy_id_insert,
            'evidence_review_id' : evidence_review_id_insert,
            'database_id' : database_id, 
            'search_type_id' : search_type_id, 
            'search_strategy_type_id' : search_strategy_type_id, 
            'searchstrat_year_start' : searchstrat_year_start_insert, 
            'searchstrat_year_end' : searchstrat_year_end_insert, 

        })

        search_strategies_df['searchdetail_file_path'] = 'placeholder'

        #check columns 
        if self._df_sqltable_column_check(search_strategies_df, table_name): 
            filtered_search_strategies_df = self._prior_id_check(search_strategies_df, 'search_strategy_id', table_name)
            if not filtered_search_strategies_df.empty: 
                with self.engine.begin() as conn:
                    self.logger.info(f"Inserting new search strategies into database, number of entries detected: {len(filtered_search_strategies_df)}")
                    filtered_search_strategies_df.to_sql(table_name, conn, if_exists='append', index=False, method = 'multi')
            else: 
                self.logger.info(f"No new search strategies to insert")

    def _handle_topic_specific_search(self, search_strategy_id_list, evidence_review_id_list, file): 

        for search_strategy_id, evidence_review_id in zip(search_strategy_id_list, evidence_review_id_list):
            df = pd.read_parquet(file)
            group_df = df.query(f"question_id == '{evidence_review_id}'")
            self.migrate_search_result_articles(search_strategy_id, group_df)

    def _grab_latest_search_result_article_id(self): 
                # Get  current max ID from database if exists
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT COALESCE(MAX(search_result_article_id), 0) FROM search_result_articles"))
            self.current_article_id = result.scalar() + 1
    
    def migrate_search_result_articles(self, search_strategy_id_count, search_result_article_df): 
        
        table_name = 'search_result_articles'

        db_id_mappings = {
            db: col for db, col in zip(
                self.database_id_mapping['database'],
                self.database_id_mapping['idcol']
            )
        }

        idcol = db_id_mappings.get(self.database_name)
        if not idcol:
            self.logger.error(f"Unknown database: {self.database_name}")
            raise ValueError(f"Unknown database: {self.database_name}")
        
        if self.database_name == 'embase' or self.database_name == 'medline': 
            search_result_article_df.rename(columns = {
                'primary_title' : 'title', 
                'notes_abstract' : 'abstract',
            }, inplace=True)

        _extract = search_result_article_df[[idcol, 'title', 'abstract', 'publication_year']].copy()
        _extract.rename(columns = {idcol : 'original_id'}, inplace=True)
        _extract['search_strategy_id'] = search_strategy_id_count

        #grab and update search result article id 
        self._grab_latest_search_result_article_id()
        _extract['search_result_article_id'] = range(self.current_article_id, self.current_article_id + len(_extract), 1)
        
        try:
            _extract = self._clean_publication_year(_extract)
        except Exception as e:
            self.logger.error(f"Failed to clean publication years: {e}")
            raise

        if self._df_sqltable_column_check(_extract, table_name): 
            filtered_df = self._prior_id_check(_extract, 'search_result_article_id', table_name)
            if not filtered_df.empty: 
                try: 
                    with self.engine.begin() as conn:
                        filtered_df.to_sql(table_name, conn, if_exists='append', index=False, method='multi')
                except Exception as e: 
                    self.logger.error(f"Error inserting search result articles into database: {e}")
                    raise e 
            else: 
                self.logger.info(f"No new search result articles to insert")

    def _clean_publication_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize publication year column, handling various edge cases"""
        df = df.copy()
        
        try:
            # Step 1: Convert to string and clean
            df['publication_year'] = (df['publication_year']
                .astype(str)
                .replace(['', 'nan', 'None', '//', 'NaT'], pd.NA)
                .str.strip()
            )
            
            # Step 2: Convert to numeric, coercing errors to NaN
            df['publication_year'] = pd.to_numeric(df['publication_year'], errors='coerce')
            
            # Step 3: Convert to nullable integer
            df['publication_year'] = df['publication_year'].astype('Int64')
            
            # Log results
            total = len(df)
            valid = df['publication_year'].notna().sum()
            self.logger.info(f"Publication year cleaning results - Total: {total}, Valid: {valid}, Missing: {total-valid}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error cleaning publication years: {e}")
            # Show sample of problematic values
            problem_values = df[df['publication_year'].notna()]['publication_year'].head()
            self.logger.error(f"Sample of problematic values: {problem_values}")
            raise


if __name__ == '__main__':
    db_name = os.getenv('DB_NAME')
    db_user = os.getenv('DB_USER')
    db_pwd = os.getenv('DB_PWD')
    db_host = os.getenv('DB_HOST')
    print(db_name, db_user, db_pwd, db_host)
    logger = LoggerConfig.setup_logger(logger_name = 'sql_data_migration')
    sql_instance = sql_data_migration(db_name, db_user, db_pwd, db_host, logger)
    create_basic_tables(db_name, db_user, db_pwd, db_host)
    create_database(db_name, db_user, db_pwd, db_host)
    sql_instance.fill_reference_tables()
    sql_instance.migrate_gdg_data()
    sql_instance.migrate_ground_truth_data()
    sql_instance.migrate_search_strategies()
    sql_instance.migrate_search_result_articles()
    # sql_instance.migrate_search_result_data()
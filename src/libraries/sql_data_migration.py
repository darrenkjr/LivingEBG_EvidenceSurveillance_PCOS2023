import pandas as pd
import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()
from sqlalchemy import create_engine, text
from sqlalchemy.types import Integer, VARCHAR
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from libraries.logging_config import LoggerConfig
import psycopg2

class sql_data_migration:

    def __init__(self, db_name, db_user, db_pwd, db_host, db_port, logger = None):

            # Create a SQLAlchemy engine

        self.logger = logger
        self.db_name = db_name
        self.db_user = db_user
        self.db_pwd = db_pwd
        self.db_host = db_host
        self.db_port = db_port

        self.engine  = self._create_database() 
        
        self.logger.info(f"Connecting to database {self.db_name}")
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
            'search_strategy_type_id' : [1, 2, 3], 
            'name' : ['boolean-kw', 'openalex-topic-search', 'boolean-kw-vectorsearch']
        }

        self.database_id_mapping = {
            'database' : ['pubmed', 'medline', 'embase', 'openalex'], 
            'idcol' : ['pmid', 'id', 'accession_number', 'id']
        }

        self.consolidated_results_path = Path(__file__).parent.parent / 'consolidated_results'

        self.current_article_id  = 1
        self.topicsearch_pubyear_clean_flag = False
        self.table_list = ['gdgs', 'databases', 'search_types', 'search_strategy_types', 'evidence_reviews', 'ground_truth_articles', 'search_strategies','search_result_articles', 'evaluation_results']

        #expected table length after initialisation and migration
        self.expected_table_length = {
            'gdgs' : 6, 
            'databases' : 4, 
            'search_types' : 2, 
            'searchstrategy_types' : 3, 
            'evidence_reviews' : 55, 
            'ground_truth_articles' : 1246, 
            'search_strategies' : 82, 
            'search_result_articles' : 294611, 
            'evaluation_results' : 0
        }

    def _create_database(self):
        """Create database if it doesn't exist using pure SQLAlchemy"""
        try:
            # Connect to default postgres database
            default_engine = create_engine(
                f'postgresql://{self.db_user}:{self.db_pwd}@{self.db_host}:{self.db_port}/postgres'
            )
            
            with default_engine.connect() as conn:
                # Set isolation level to AUTOCOMMIT
                conn.execution_options(isolation_level="AUTOCOMMIT")
                
                # Check if database exists
                result = conn.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
                    {"db_name": self.db_name}
                )
                
                if not result.scalar():
                    # Create database if it doesn't exist
                    conn.execute(text(f"CREATE DATABASE {self.db_name}"))
                    self.logger.info(f"Database {self.db_name} created")
                else:
                    self.logger.info(f"Database {self.db_name} already exists")
                    
        except Exception as e:
            self.logger.error(f"Error creating database: {e}")
            raise
        finally:
            # Ensure connection is closed
            if 'default_engine' in locals():
                default_engine.dispose()

            #create self engine 
            return create_engine(f'postgresql://{self.db_user}:{self.db_pwd}@{self.db_host}:{self.db_port}/{self.db_name}')

    def _create_tables(self):
        """Create database tables using SQLAlchemy if they don't exist"""
        try:
            with self.engine.begin() as conn:
                # Check existing tables
                result = conn.execute(text(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
                ))
                #check table names 
                tables = [row[0] for row in result]
                if set(tables) == set(self.table_list): 
                    self.logger.info(f"Tables already exist, table names: {tables}")
                    return 

                self.logger.info('No tables found, creating basic tables')
                # Read SQL file
                sql_code_path = Path(__file__).parent / 'psql_tablesetup.sql'
                with open(sql_code_path, 'r') as file:
                    sql_code = file.read()
                

                # Split and execute multiple statements
                statements = sql_code.split(';')
                for statement in statements:
                    if statement.strip():
                        conn.execute(text(statement))
                
                # Verify created tables
                result = conn.execute(text(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
                ))
                created_tables = [row[0] for row in result]
                self.logger.info(f'Basic tables created, table names: {created_tables}')


        except Exception as e:
            self.logger.error(f'Error creating tables: {e}')
            raise
                

 

 
    def fill_reference_tables(self):

        #check if tables exist 

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
            filered_df = self._prior_id_check(gdg_extract, 'gdg_id', 'gdgs', engine = self.engine, logger = self.logger) 
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
            evidence_review_extract_filtered = self._prior_id_check(evidence_review_extract, 'evidence_review_id', 'evidence_reviews', engine = self.engine, logger = self.logger)
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
            filtered_ground_truth_df = self._prior_id_check(ground_truth_df, 'ground_truth_article_id', 'ground_truth_articles', engine = self.engine, logger = self.logger) 
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
    

    @staticmethod
    def _prior_id_check(df, idcol, table_name, engine = None, logger = None):
        temp_table = f"temp_{table_name}_ids"
        try:
            temp_df = df.copy()
            if table_name == 'evidence_reviews': 
                temp_df[idcol] = temp_df[idcol].astype(str)
                sql_type = VARCHAR(50)
            else: 
                temp_df[idcol] = temp_df[idcol].astype(int)
                sql_type = Integer
    
            with engine.begin() as conn:
                temp_table = f"temp_{table_name}_ids"
                temp_df[[idcol]].to_sql(temp_table, conn, if_exists='replace', index=False, 
                                dtype={idcol: sql_type})  #
                query = f"""
                    SELECT temp.{idcol} 
                    FROM {temp_table} temp
                    WHERE NOT EXISTS (
                        SELECT 1 
                        FROM {table_name} existing_table 
                        WHERE existing_table.{idcol} = temp.{idcol}
                    )
                """
                result = conn.execute(text(query))
                new_ids = [row[0] for row in result.fetchall()]
                #drop temporary table 
                conn.execute(text(f"DROP TABLE IF EXISTS {temp_table} CASCADE"))
            

            filtered_df = temp_df[temp_df[idcol].isin(new_ids)].copy()
            logger.info(f"Found {len(filtered_df)} new ids for {table_name}")

            return filtered_df
        except Exception as e:

            logger.error(f"Error finding new ids for {table_name}: {e}")
            #still do clean up 
            with engine.connect() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {temp_table} CASCADE"))
            raise 
    
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
        search_results_to_insert = []
        topic_specific_search_results_to_insert = []


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
                    self.search_type = 'topic-specific'
                elif search_type_name == 'overarching': 
                    self.search_type = 'overarching'
                else: 
                    self.logger.error(f"Error: search type {search_type_name} not found")
                
                search_strategy_type_name = '_'.join([file_name[3],file_name[4]])
                if search_strategy_type_name == 'boolkw_search' or 'consolidated_boolkw': 
                    search_strategy_type = 'boolean-kw'
                elif search_strategy_type_name == 'topic_search': 
                    search_strategy_type = 'openalex-topic-search'

                if self.search_type == 'overarching': 
                    if file.name == 'oa_overarching_consolidated_topic_search_results.parquet': 
                        self.logger.warning(f"Skipping {file.name} as its too large")
                    else:
                        searchstrat_year_start_insert.append(1990)
                        searchstrat_year_end_insert.append(2020)
                        evidence_review_id_insert.append('overall')
                        search_strategy_id_insert.append(search_strategy_id_count)
                        database_insert.append(self.database_name)
                        search_types_insert.append(self.search_type)
                        search_strategy_types_insert.append(search_strategy_type)

                        search_result_df = pd.read_parquet(file)
                        search_results_to_insert.append({
                            'database_name' : self.database_name, 
                            'search_strategy_id' : search_strategy_id_count, 
                            'search_result_df' : search_result_df, 
                            'search_type' : self.search_type
                        })
                        search_strategy_id_count += 1
                        

                elif self.search_type == 'topic-specific': 
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
                    search_strat_id_list = list(range(search_strategy_id_count, search_strategy_id_count + n_entries,1))
                    search_strategy_id_insert.extend(search_strat_id_list)
                    database_insert.extend([self.database_name] * n_entries)
                    search_types_insert.extend([self.search_type] * n_entries)
                    search_strategy_types_insert.extend([search_strategy_type] * n_entries)
                    searchstrat_year_start_insert.extend(_er_extract['searchstrat_year_start'].tolist())
                    searchstrat_year_end_insert.extend(_er_extract['searchstrat_year_end'].tolist())
                    evidence_review_id_list = _er_extract['evidence_review_id'].tolist()
                    evidence_review_id_insert.extend(evidence_review_id_list)
                    #update search strategy id count 
                    search_strategy_id_count += n_entries

                    topic_specific_search_results_to_insert.append({
                        'database_name' : self.database_name, 
                        'search_type' : self.search_type, 
                        'search_strategy_id_list' : search_strat_id_list, 
                        'evidence_review_id_list' :evidence_review_id_list, 
                        'file_path' : file
                    })

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
        search_strategies_df['vector_search'] = False

        #check columns 
        if self._df_sqltable_column_check(search_strategies_df, table_name): 
            filtered_search_strategies_df = self._prior_id_check(search_strategies_df, 'search_strategy_id', table_name, engine = self.engine, logger = self.logger)
            if not filtered_search_strategies_df.empty: 
                with self.engine.begin() as conn:
                    self.logger.info(f"Inserting new search strategies into database, number of entries detected: {len(filtered_search_strategies_df)}")
                    filtered_search_strategies_df.to_sql(table_name, conn, if_exists='append', index=False, method = 'multi')
            else: 
                self.logger.info(f"No new search strategies to insert")

        self.logger.info(f"Search strategies inserted into database")  
        
        return search_results_to_insert, topic_specific_search_results_to_insert

    def _handle_topic_specific_search(self, search_strategy_id_list, evidence_review_id_list, file_path, database_name, search_type): 
        self.logger.info(f"Handling topic specific search for {file_path}")
        self.database_name = database_name
        self.search_type = search_type
        df = pd.read_parquet(file_path)
        df = self._clean_publication_year(df)
        self.topicsearch_pubyear_clean_flag = True
        for search_strategy_id, evidence_review_id in zip(search_strategy_id_list, evidence_review_id_list):
            if evidence_review_id == 'overall': 
                group_df = df.copy()
            else:
                group_df = df.query(f"question_id == '{evidence_review_id}'").copy()
            self.logger.info(f'Inserting topic specific results for evidence review id : {evidence_review_id}. Corresponding search strategy id : {search_strategy_id}')
            self.migrate_search_result_articles(search_strategy_id, group_df, database_name, search_type)

    def _grab_latest_search_result_article_id(self): 
                # Get  current max ID from database if exists
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT COALESCE(MAX(search_result_article_id), 0) FROM search_result_articles"))
            self.current_article_id = result.scalar() + 1
    
    def migrate_search_result_articles(self, search_strategy_id_count, search_result_article_df, database_name, search_type): 
        
        table_name = 'search_result_articles'

        db_id_mappings = {
            db: col for db, col in zip(
                self.database_id_mapping['database'],
                self.database_id_mapping['idcol']
            )
        }
        self.database_name = database_name
        self.search_type = search_type
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
            if not self.topicsearch_pubyear_clean_flag: 
                _extract = self._clean_publication_year(_extract)
        except Exception as e:
            self.logger.error(f"Failed to clean publication years: {e}")
            raise

        if self._df_sqltable_column_check(_extract, table_name): 
            filtered_df = self._prior_id_check(_extract, 'search_result_article_id', table_name, engine = self.engine, logger = self.logger)
            if not filtered_df.empty: 
                try: 
                    with self.engine.begin() as conn:
                        self.logger.info(f"Inserting new search result articles into database, for {self.database_name}, search strategy id : {search_strategy_id_count}, search_type : {self.search_type}")
                        filtered_df.to_sql(table_name, conn, if_exists='append', index=False, method='multi')
                except Exception as e: 
                    self.logger.error(f"Error inserting search result articles into database")
                    raise e 
            else: 
                self.logger.info(f"No new search result articles to insert")


    def _clean_publication_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize publication year column, handling various edge cases"""
        self.logger.info(f"Cleaning publication year for {self.database_name}")
        df_copy = df.copy()
        
        try:
            # Step 1: Convert to string and clean
            df_copy['publication_year'] = (df_copy['publication_year']
                .astype(str)
                .str.replace('//', '', regex=False)  # Remove //
                .str.strip('/')     # Remove trailing/leading slashes
                .str.strip()        # Remove whitespace
                .str.split(',')     # Split on comma (if multiple years)
                .str[0]             # Take first value
                .str.strip()        # Clean up any remaining whitespace
                .replace(['', 'nan', 'None', 'NaT'], pd.NA)
            )
            # Step 2: Convert to numeric, coercing errors to NaN
            df_copy['publication_year'] = pd.to_numeric(df_copy['publication_year'], errors='coerce')
            # Step 3: Convert to nullable integer
            df_copy['publication_year'] = df_copy['publication_year'].astype('Int64')
            
            # Log results
            total = len(df_copy)
            valid = df_copy['publication_year'].notna().sum()
            self.logger.info(f"Publication year cleaning results - Total: {total}, Valid: {valid}, Missing: {total-valid}")
            
            return df_copy
            
        except Exception as e:
            self.logger.error(f"Error cleaning publication years: {e}")
            # Show sample of problematic values
            problem_values = df[df['publication_year'].notna()]['publication_year'].head()
            self.logger.error(f"Sample of problematic values: {problem_values}")
            raise

    def drop_all_tables_data(self): 
        self.logger.info("Dropping all tables data before migrating all data")
        try:
            with self.engine.begin() as conn:  # Use begin() for transaction
                # Fix: Remove extra quotes and execute statements separately
                queries = [
                    "SET session_replication_role = 'replica';",
                    """
                    DROP TABLE IF EXISTS 
                        search_result_articles,
                        evaluation_results,
                        search_strategies,
                        evidence_reviews,
                        databases,
                        search_types,
                        search_strategy_types,
                        gdgs,
                        ground_truth_articles
                    CASCADE;
                    """,
                    "SET session_replication_role = 'origin';"
                ]
                
                for query in queries:
                    conn.execute(text(query))
                    
            self.logger.info("Successfully dropped all tables")
            
        except Exception as e:
            self.logger.error(f"Error dropping tables: {e}")
            raise

    def check_data_migration(self):
        
        if self._check_tables() and self._check_table_length(): 
            self.logger.info(f"All tables exist and have the correct length")
            return True
        else: 
            self.logger.error(f"Error: Tables do not exist or have incorrect length")
            return False

    def _check_tables(self):
        try:
            with self.engine.begin() as conn:
                # Get list of temporary tables (only do this once)
                result = conn.execute(text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_name LIKE 'temp_%'"
                ))
                temp_tables = [row[0] for row in result]
                
                if temp_tables:
                    self.logger.info(f"Found temporary tables: {temp_tables}")
                    # Drop all temporary tables in a single transaction
                    for table_name in temp_tables:
                        conn.execute(text(f"DROP TABLE IF EXISTS {table_name} CASCADE"))
                    self.logger.info(f"Successfully dropped {len(temp_tables)} temporary tables")
                else:
                    self.logger.info("No temporary tables found")


                #check tables again 
                result = conn.execute(text(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
                )) 

                tables = [row[0] for row in result]
                if not set(self.table_list).issubset(set(tables)): 
                    self.logger.warning(f"Tables do not match expected table list, expected: {self.table_list}, found: {tables}")
                    return False
                else: 
                    self.logger.info(f"All appropriate tables exist")
                    return True

                    
        except Exception as e:
            self.logger.error(f"Error cleaning up temporary tables: {e}")
            raise
    
    def _check_table_length(self): 
        self.logger.info(f"Checking table length")


        for table_name in self.table_list: 
            with self.engine.connect() as conn: 
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                table_length = result.scalar()
                self.logger.info(f"Table {table_name} length: {table_length}")
                
                try: 
                    assert table_length >= self.expected_table_length[table_name], f"Table {table_name} length does not match at least the expected length, expected: {self.expected_table_length[table_name]}, found: {table_length}"
                except AssertionError as e: 
                    self.logger.error(f"Error: Table {table_name} length does not match expected length, expected: {self.expected_table_length[table_name]}, found: {table_length}")
                    return False
                
        return True


    
    

if __name__ == '__main__':
    os.environ['PGHOST'] = '/var/run/postgresql'
    db_name = os.getenv('DB_NAME')
    db_user = os.getenv('DB_USER')
    db_pwd = os.getenv('DB_PWD')
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT')
    logger = LoggerConfig.setup_logger(logger_name = 'sql_data_migration')
    sql_instance = sql_data_migration(db_name, db_user, db_pwd, db_host, db_port, logger)
    sql_instance.fill_reference_tables()
    sql_instance.migrate_gdg_data()
    sql_instance.migrate_ground_truth_data()
    overarching_search_results_to_insert, topic_specific_search_results_to_insert = sql_instance.migrate_search_strategies()
    for search_results in overarching_search_results_to_insert: 
        sql_instance.migrate_search_result_articles(search_results['search_strategy_id'], search_results['search_result_df'], search_results['database_name'], search_results['search_type'])
    for search_results in topic_specific_search_results_to_insert: 
        sql_instance._handle_topic_specific_search(search_results['search_strategy_id_list'], search_results['evidence_review_id_list'], search_results['file_path'], search_results['database_name'], search_results['search_type'])
    # sql_instance.migrate_search_result_data()

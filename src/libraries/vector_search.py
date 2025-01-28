from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import numpy as np 
import pandas as pd 
import torch 
from pathlib import Path
from tqdm.auto import tqdm

from sqlalchemy import create_engine
from sqlalchemy import text
from dotenv import load_dotenv
import os 
#add libraries to path 
from libraries.logging_config import LoggerConfig
from libraries.sql_procedures import sql_procedures
from libraries.eval import search_evaluation
load_dotenv()

class vector_search_implementation(): 

    def __init__(self, model_name = 'allenai/specter2_base', logger = None, df = None): 

        self.model = AutoAdapterModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.logger = logger
        #check if we're in WSL environment 
        self._database_check()
        #check for cuda 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
    
        self.model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
        self.model.to(self.device)

        self.sql_procedures = sql_procedures(logger = self.logger, engine = self.engine)

    def _database_check(self):
        self.wsl_flag = os.environ.get('WSL_DISTRO_NAME') is not None
        if self.wsl_flag: 
            os.environ['PGHOST'] = '/var/run/postgresql' 
        db_name = os.getenv('DB_NAME')
        db_user = os.getenv('DB_USER')
        db_pwd = os.getenv('DB_PWD')
        db_host = os.getenv('DB_HOST') 
        self.logger.info(f'Connecting to database {db_name}')
        try: 
            self.engine = create_engine(f'postgresql://{db_user}:{db_pwd}@{db_host}:5432/{db_name}')
            self.logger.info(f'Connected to database {db_name}')
        except Exception as e: 
            self.logger.error(f'Error connecting to database {db_name}: {e}')
            raise e 
        #check existence of ground truth table 
        self.logger.info(f'Checking existence of ground truth table')
        try: 
            with self.engine.connect() as conn: 
                
                result = conn.execute(text("SELECT COUNT(*) FROM ground_truth_articles")).scalar()
                if result == 0: 
                    self.logger.error(f'No ground truth articles found')
                    raise Exception(f'Ground truth table does not exist')
                else: 
                    self.logger.info(f'Ground truth table exists, with {result} rows')
        except Exception as e: 
            self.logger.error(f'Error checking existence of ground truth table: {e}')
            raise e 
        
        self.logger.info(f'Checking existence of search result article table')
        try: 
            with self.engine.connect() as conn: 
                result = conn.execute(text("SELECT COUNT(*) FROM search_result_articles")).scalar()
                if result == 0: 
                    self.logger.error(f'Search result article table does not exist')
                    raise Exception(f'Search result article table does not exist')
                else: 
                    self.logger.info(f'Search result article table exists, with {result} rows')
        except Exception as e: 
            self.logger.error(f'Error checking existence of search result article tables: {e}')
            raise e 


    
    def _generate_embeddings(self, text): 
        # Debug device locations

        encoded_input = self.tokenizer(text, 
                                    return_tensors='pt', 
                                    padding=True, 
                                    truncation=True, 
                                    max_length = 512)
        
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        with torch.no_grad():
            output = self.model(**encoded_input)
            embeddings = output.last_hidden_state[:,0,:].detach().cpu().numpy()
        return embeddings 
        
    def generate_goldset_embeddings(self): 
        

        goldset_df = self.sql_procedures.retrieve_query_goldset(evidence_review_id = 'overall')
        
       
        goldset_df['text'] = goldset_df['title']  + self.tokenizer.sep_token + goldset_df['abstract']
        
        tqdm.pandas(desc=f"Generating embeddings for query goldset", 
                   unit="article",
                   ncols=125)
        
        goldset_df['embeddings'] = goldset_df['text'].progress_apply(
            self._generate_embeddings
        )
        #add to sql database 
        self.sql_procedures.add_embeddings_to_sql(input_df = goldset_df, embedding_table_name = 'query_goldset_embeddings', linking_id = 'ground_truth_article_id')


    def generate_searchspace_embeddings(self, database: int): 
        '''
        Given database id, generate embeddings for search result articles associated with that database 

        Arg: 
            database: int, database id to generate embeddings for 
        '''

        #retrieve all search result articles associated with a database 
        with self.engine.connect() as conn: 
                database_name = conn.execute(text(f"SELECT database_name FROM databases WHERE database_id = {database}")).scalar()

        self.logger.info(f'Retrieving search result articles for database {database_name}')
        search_result_articles_df = self.sql_procedures.retrieve_search_result_articles_databaseview(database_id = database)

        self.logger.info(f'Retrieved {len(search_result_articles_df)} search result articles for database {database_name}')
        try:
            if search_result_articles_df['title'].isnull().sum() > 0 or search_result_articles_df['abstract'].isnull().sum() > 0:
                self.logger.warning(f'There are null values for title or abstract in the search result articles for database {database_name}, filling empty values with empty strings')
                search_result_articles_df['title'] = search_result_articles_df['title'].fillna('')
                search_result_articles_df['abstract'] = search_result_articles_df['abstract'].fillna('')
            
            self.logger.info('Tokenizing search result articles')
            search_result_articles_df['text'] = search_result_articles_df['title'] + self.tokenizer.sep_token + search_result_articles_df['abstract'] 
            
            self.logger.info(f'Generating embeddings for search result articles for database {database_name}')
            tqdm.pandas(desc=f"Generating embeddings for search result articles for database {database_name}", 
                        unit="article",
                        ncols = 125) 
            search_result_articles_df['embeddings'] = search_result_articles_df['text'].progress_apply(
                self._generate_embeddings
            )

            #retrieve database name 
            self.logger.info(f'Adding embeddings for search result articles associated with database: {database_name}')
            self.sql_procedures.add_embeddings_to_sql(input_df = search_result_articles_df, embedding_table_name = f'searchspace_database_{database_name}_embeddings', linking_id = 'search_result_article_id')
        except ValueError as e: 
            self.logger.error(f'Error generating embeddings for search result articles associated with database: {database_name}: {e}')
            raise e 

    def overarching_vector_search_with_rrf(self): 

        '''
        Run vector serach with cosine similiarity. First - genreate goldset embeddings (if not already done)
        Then - generate searchspace embeddings for each database (if not already done)
        Then loop through serach strategy articles with search_type = 1, join with searchspace embeddings
        '''
        self.logger.info(f'Retrieving overarching search strats')
        overarching_searchstrat_df = self.sql_procedures.retrieve_searchstrat(search_type = 'overarching')

        #create new search strats - which is a copy of the overarching search start, but with a vector search flag
        temp_df = overarching_searchstrat_df.copy()
        temp_df['vector_search'] = True
        temp_df['search_strategy_type_id'] = 3
        temp_df['original_search_strategy_id'] = temp_df['search_strategy_id']
        self.logger.info(f'Creating new search strats with vector search flag for overarching search strats')
        input_searchstrat_df = self.sql_procedures.create_new_searchstrat_vectorsearch(temp_df)
        eval_metrics_df_list = []
        result_cutoff_df_list = []

        for original_searchstrat_id, searchstrat_id, evidence_review_id in zip(input_searchstrat_df['original_search_strategy_id'], input_searchstrat_df['search_strategy_id'], input_searchstrat_df['evidence_review_id'],): 
            self.sql_procedures.run_vector_search(original_searchstrat_id = original_searchstrat_id, searchstrat_id = searchstrat_id, evidence_review_id = evidence_review_id)
            #ranked results output 
            rrf_sim_result_df = self.sql_procedures.rrf_combine_results(searchstrat_id = searchstrat_id, evidence_review_id = evidence_review_id)
            #conduct evaluation 
            eval_metrics_df, result_cutoff_df = self._evaluate_vector_search(rrf_sim_result_df, evidence_review_id = evidence_review_id, search_strat_id = searchstrat_id,  search_type = 'overarching')
            eval_metrics_df_list.append(eval_metrics_df)
        
        eval_metrics_df_all = pd.concat(eval_metrics_df_list)

        return eval_metrics_df_all

    def topic_specific_vector_search_with_rrf(self): 

        """
        Retrieves overarching search strats and converts this to topic specific search strats using vector search 
        """
        #retrieve unique evidence review ids and construct search strat df to input into serach strat table 
        evidence_review_id_list = self.sql_procedures.retrieve_evidence_review_ids(search_type = 'topic-specific')
        overarching_searchstrat_df = self.sql_procedures.retrieve_searchstrat(search_type = 'ovearching')

        base_df = pd.DataFrame({
            'evidence_review_id': evidence_review_id_list,
            'search_type_id': 2,
            'search_strategy_type_id': 3,
            'vector_search': True,
            'searchstrat_year_start': 1990,
            'searchstrat_year_end': 2022,
            'searchdetail_file_path': 'placeholder'
        })
        
        database_df = pd.DataFrame({'database_id': [1, 2, 3, 4]})
        
        #create dict mapping database id to overarching search strat id 
        database_to_overarching_strat_map = {row['database_id']: row['search_strategy_id'] for _, row in overarching_searchstrat_df.iterrows()}
        
        input_searchstrat_df = (
            base_df
            .assign(key=1)
            .merge(database_df.assign(key=1), on='key')
            .drop('key', axis=1)
            .reset_index()
            .rename(columns={'index': 'search_strategy_id'})
            .assign(search_strategy_id=lambda x: x['search_strategy_id'] + 1)
            .assign(original_search_strategy_id=lambda x: x['database_id'].apply(lambda y: database_to_overarching_strat_map[y]))
        )
        assert len(input_searchstrat_df) == len(evidence_review_id_list) * len(database_df), "The number of rows in input_df should be equal to the number of evidence review IDs times 4."


        topic_specific_vs_searchstrat_df = self.sql_procedures.create_new_searchstrat_vectorsearch(input_searchstrat_df).drop(columns = ['original_search_strategy_id']).copy()

        eval_metrics_df_list = []

        result_cutoff_df_dct = {
            'database_id': [],
            'search_strategy_id': [],
            'evidence_review_id': [],
            'result_cutoff_df': []
        }

        for database_id, original_searchstrat_id, searchstrat_id, evidence_review_id in zip(topic_specific_vs_searchstrat_df['database_id'], topic_specific_vs_searchstrat_df['original_search_strategy_id'] ,topic_specific_vs_searchstrat_df['search_strategy_id'], topic_specific_vs_searchstrat_df['evidence_review_id']): 
            self.sql_procedures.run_vector_search(original_searchstrat_id = original_searchstrat_id, searchstrat_id = searchstrat_id, evidence_review_id = evidence_review_id)
            #ranked results output 
            rrf_sim_result_df = self.sql_procedures.rrf_combine_results(searchstrat_id = searchstrat_id, evidence_review_id = evidence_review_id)
            #conduct evaluation 
            eval_metrics_df, result_cutoff_df = self._evaluate_vector_search(rrf_sim_result_df, evidence_review_id = evidence_review_id, search_strat_id = searchstrat_id,  search_type = 'topic-specific')
            eval_metrics_df_list.append(eval_metrics_df)

            ### commented out - code to evaluate overall workflow 

            # result_cutoff_df_dct['database_id'].append(database_id)
            # result_cutoff_df_dct['search_strategy_id'].append(searchstrat_id)
            # result_cutoff_df_dct['evidence_review_id'].append(evidence_review_id)
            # result_cutoff_df_dct['result_cutoff_df'].append(result_cutoff_df)

        eval_metrics_topicspecific_all = pd.concat(eval_metrics_df_list)

        # overall_workflow_result_df = pd.DataFrame(result_cutoff_df_dct).groupby(['database_id']).apply(
        #     lambda x: pd.concat(x['result_cutoff_df'])).to_frame('combined_result_df').assign(
        #         evidence_review_id='overall',
        #         search_type='topic-specific',
        #     ).reset_index()
        # #evaluate entire workflow here 
        return eval_metrics_topicspecific_all


    def _evaluate_vector_search(self, rrf_sim_result_df, evidence_review_id, search_strat_id, search_type): 
        '''
        Evaluates RRF combined results on evaluation set 
        '''

        self.logger.info(f'Retrieving evaluation set for evidence review id {evidence_review_id}')
        evaluation_set_df = self.sql_procedures.retrieve_evaluation_set(evidence_review_id = evidence_review_id)
        #retrieve database name from database reference table givne search start id and database id 
        with self.engine.begin() as conn: 
            database_name = conn.execute(text(
                """
                SELECT d.database_name 
                FROM search_strategies ss
                JOIN databases d ON ss.database_id = d.database_id
                WHERE ss.search_strategy_id = :search_strat_id
                """
            ), {'search_strat_id': search_strat_id}).scalar()

        self.logger.info(f'Retrieving query goldset for evidence review id: {evidence_review_id}')
        query_goldset_df = self.sql_procedures.retrieve_query_goldset(evidence_review_id = evidence_review_id)
        eval_cls = search_evaluation(database = database_name, search_type = search_type, vector_search = True, logger = self.logger)

        self.logger.info(f'Running vector search evaluation pipeline')
        eval_metrics_df = eval_cls.run_vectorsearch_eval_pipeline(result_set = rrf_sim_result_df, evaluation_set = evaluation_set_df, query_goldset = query_goldset_df, database_name = database_name)
        self.logger.info(f'Vector search evaluation pipeline complete')
        return eval_metrics_df
    
    def _check_embeddings_exist(self): 
        """Check embedding tables and determine which need regeneration"""
        
        tables_to_regenerate = []
        embedding_stats = {}
        
        table_checks = {
            'query_goldset_embeddings': {
                'df': self.sql_procedures.retrieve_query_goldset(evidence_review_id='overall'),
                'desc': 'goldset'
            },
            'searchspace_database_pubmed_embeddings': {
                'df': self.sql_procedures.retrieve_search_result_articles_databaseview(database_id='1'),
                'desc': 'PubMed'
            },
            'searchspace_database_medline_embeddings': {
                'df': self.sql_procedures.retrieve_search_result_articles_databaseview(database_id='2'),
                'desc': 'Medline'
            },
            'searchspace_database_embase_embeddings': {
                'df': self.sql_procedures.retrieve_search_result_articles_databaseview(database_id='3'),
                'desc': 'Embase'
            },
            'searchspace_database_openalex_embeddings': {
                'df': self.sql_procedures.retrieve_search_result_articles_databaseview(database_id='4'),
                'desc': 'OpenAlex'
            },
        }
        
        for table_name, info in table_checks.items():
            
            expected_count = len(info['df'])
            
            check_query = f"""
                SELECT 
                    COUNT(*) as total_count,
                    COUNT(CASE WHEN embeddings IS NOT NULL THEN 1 END) as valid_count
                FROM {table_name}
            """
            
            try:
                with self.engine.connect() as conn:
                    self.logger.info(f"Performing check for {table_name}")
                    result = conn.execute(text(check_query)).fetchone()
                    
                    stats = {
                        'total': result[0],
                        'valid': result[1],
                        'expected': expected_count
                    }
                    embedding_stats[table_name] = stats
    
                    
                    needs_regeneration = (
                        result[0] != expected_count or  
                        result[1] != result[0]
                    )

                    if needs_regeneration:
                        tables_to_regenerate.append(table_name)
                        self.logger.warning(f"{info['desc']} embeddings need regeneration")
                        self.logger.warning(f"Total rows: {stats['total']}, Valid rows (non-null embeddings): {stats['valid']}, Expected rows: {stats['expected']}")
                    else:
                        self.logger.info(f"{info['desc']} embeddings are valid, length of embeddings table is equal to expected number of rows, and there are no null values in the embeddings table")
                        self.logger.info(f"Total rows: {stats['total']}, Valid rows (non-null embeddings): {stats['valid']}, Expected rows: {stats['expected']}")
            except Exception as e:
                self.logger.error(f"Error checking {table_name}: {str(e)}")
                tables_to_regenerate.append(table_name)
        
        return tables_to_regenerate, embedding_stats

    def generate_embeddings_if_needed(self):
        """Generate embeddings only for tables that need it"""
        
        tables_to_regenerate, stats = self._check_embeddings_exist()
        
        if not tables_to_regenerate:
            self.logger.info("All embedding tables are valid - no regeneration needed")
            return
        
        self.logger.info(f"Need to regenerate embeddings for: {tables_to_regenerate}")
        
        # Generate specific embeddings as needed
        if 'query_goldset_embeddings' in tables_to_regenerate:
            self.generate_goldset_embeddings()
            
        if any('searchspace_database' in table for table in tables_to_regenerate):
            for table in tables_to_regenerate: 
                database_name = table.split('_')[2]
                database_id = pd.read_sql(f"SELECT database_id FROM databases WHERE database_name = '{database_name}'", self.engine).iloc[0,0]
                self.generate_searchspace_embeddings(database_id)


    def run_vector_search(self, search_type):
        #run check if embeddings already exist
        self.sql_procedures.create_querygoldset_view()
        self.sql_procedures.setup_embeddings_table(materialized_view_name = 'query_goldset_view', embedding_table_name = 'query_goldset_embeddings', linking_id = 'ground_truth_article_id', id_dtype = 'INT')
        
        for database in [1, 2, 3, 4]: 
            self.sql_procedures.setup_searchspace_database_embedding_table(database_id = database)
        
        self.logger.info('Generating embeddings if needed')
        self.generate_embeddings_if_needed()

        if search_type == 'overarching': 
            eval_metrics_df = self.overarching_vector_search_with_rrf()
        elif search_type == 'topic-specific': 
            eval_metrics_df = self.topic_specific_vector_search_with_rrf()

        return eval_metrics_df
    
if __name__ == '__main__': 
    logger = LoggerConfig.setup_logger(logger_name = 'vector_search')
    vector_search_cls = vector_search_implementation(logger = logger)
    eval_metrics_df = vector_search_cls.run_vector_search(search_type = 'overarching')

    


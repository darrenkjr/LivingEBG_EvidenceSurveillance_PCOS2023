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


class vector_search_implementation(): 

    def __init__(self, model_name = 'allenai/specter2_base', logger = None, engine = None): 
        self.logger = logger
        self.logger.info(f'Initilialzing embedding model: {model_name}')
        self.model_name = model_name
        self.logger.info('Connecting to database')



        if engine is None: 
            try: 
                self.wsl_flag = os.environ.get('WSL_DISTRO_NAME') is not None
                if self.wsl_flag: 
                    os.environ['PGHOST'] = '/var/run/postgresql' 
                else: 
                    os.environ['PGHOST'] = 'localhost' 
                load_dotenv()
                db_name = os.getenv('DB_NAME')
                db_user = os.getenv('DB_USER')
                db_pwd = os.getenv('DB_PWD')
                db_host = os.getenv('DB_HOST') 
                self.logger.info(f'Connecting to database {db_name}')
                self.engine = create_engine(f'postgresql://{db_user}:{db_pwd}@{db_host}:5432/{db_name}')
                self.logger.info(f'Connected to database {db_name}')
            except Exception as e: 
                self.logger.error(f'Error connecting to database {db_name}: {e}')
                raise e 
        else: 
            self.engine = engine

        self.sql_procedures = sql_procedures(logger = self.logger, engine = self.engine)

    def _database_check(self):

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

    def generate_evidencereview_topic_embeddings(self): 
        '''
        Generate embeddings for evidence review topics 
        '''
        er_df = self.sql_procedures.retrieve_query_evidencereview_topics(evidence_review_id='overall')
        tqdm.pandas(desc=f"Generating embeddings for evidence review topics", 
                   unit="article",
                   ncols=125)
        
        er_df['embeddings'] = er_df['question'].progress_apply(
            self._generate_embeddings
        )
        self.sql_procedures.add_embeddings_to_sql(input_df = er_df, embedding_table_name = 'query_evidencereview_topic_embeddings', linking_id = 'evidence_review_id')


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

    def overarching_vector_search_with_rrf(self, vector_search_type: str): 

        '''
        Run vector serach with cosine similiarity. First - genreate goldset embeddings (if not already done)
        Then - generate searchspace embeddings for each database (if not already done)
        Then loop through serach strategy articles with search_type = 1, join with searchspace embeddings
        '''
        self.logger.info(f'Retrieving overarching search strats')
        overarching_searchstrat_df = self.sql_procedures.retrieve_searchstrat(search_type = 'overarching')

        #create new search strats - which is a copy of the overarching search start, but with a vector search flag
        temp_df = overarching_searchstrat_df.copy()

        input_searchstrat_df = pd.DataFrame()

        temp_df['vector_search'] = vector_search_type
        temp_df['search_strategy_type_id'] = 3
        temp_df['original_search_strategy_id'] = temp_df['search_strategy_id']
        search_type_id = temp_df['search_type_id'].unique()[0]
        self.logger.info(f'Creating new search strats with vector search flag for overarching search strats')
        _input_searchstrat_df = self.sql_procedures.create_new_searchstrat_vectorsearch(temp_df, vector_search_type = vector_search_type, search_type_id = search_type_id)
        input_searchstrat_df = pd.concat([input_searchstrat_df, _input_searchstrat_df])

        #retrieve original search strat id, evidence review id and new search strat id 
        eval_metrics_df_list = []
        result_cutoff_df_list = []
        for original_searchstrat_id, searchstrat_id, evidence_review_id, vs_type in zip(
            input_searchstrat_df['original_search_strategy_id'], 
            input_searchstrat_df['search_strategy_id'], 
            input_searchstrat_df['evidence_review_id'], 
            input_searchstrat_df['vector_search']): 

            self.sql_procedures.run_vector_search(original_searchstrat_id = original_searchstrat_id, searchstrat_id = searchstrat_id, evidence_review_id = evidence_review_id, vector_search_type = vs_type)
            #ranked results output 
            rrf_sim_result_df = self.sql_procedures.rrf_combine_results(searchstrat_id = searchstrat_id, evidence_review_id = evidence_review_id, original_searchstrat_id = original_searchstrat_id)
            #conduct evaluation 
            eval_metrics_df, result_cutoff_df = self._evaluate_vector_search(rrf_sim_result_df, evidence_review_id = evidence_review_id, search_strat_id = searchstrat_id,  search_type = 'overarching', vectorsearch_type = vs_type)
            eval_metrics_df_list.append(eval_metrics_df)
        
        eval_metrics_df_all = pd.concat(eval_metrics_df_list)

        return eval_metrics_df_all

    def topic_specific_vector_search_with_rrf(self, vector_search_type:str): 

        """
        Retrieves historic non vector search overarching search strats and converts this to topic specific search strats using vector search 
        """
        #retrieve unique evidence review ids and construct search strat df to input into serach strat table 
        evidence_review_id_list = self.sql_procedures.retrieve_evidence_review_ids(search_type = 'topic-specific')
        evidence_review_id_df = pd.DataFrame({'evidence_review_id': evidence_review_id_list})
        overarching_searchstrat_df = self.sql_procedures.retrieve_searchstrat(search_type = 'overarching')
    
        _df = overarching_searchstrat_df.copy()
        _df['vector_search'] = vector_search_type
        _df['search_strategy_type_id'] = 3

        #we need original search strat id in order to join with the search result article table 
        _df['original_search_strategy_id'] = _df['search_strategy_id']


        #overarching to topic specific search start conversion 
        search_type_id = 2
        _df['search_type_id'] = search_type_id


        #create cartesian product of evidence review id and search strat df 
        _df.drop(columns = ['evidence_review_id'], inplace = True)
        temp_df = pd.merge(_df, evidence_review_id_df, how = 'cross')
        temp_df['search_strategy_id'] = temp_df.index + 1
        assert len(temp_df) == len(_df) * len(evidence_review_id_df)

        topic_specific_vs_searchstrat_df = self.sql_procedures.create_new_searchstrat_vectorsearch(temp_df, vector_search_type = vector_search_type, search_type_id = search_type_id)
        eval_metrics_df_list = []
        for original_searchstrat_id, searchstrat_id, evidence_review_id, vs_type in zip(
            topic_specific_vs_searchstrat_df['original_search_strategy_id'] ,
            topic_specific_vs_searchstrat_df['search_strategy_id'], 
            topic_specific_vs_searchstrat_df['evidence_review_id'], 
            topic_specific_vs_searchstrat_df['vector_search']
            ): 
            
            self.sql_procedures.run_vector_search(
                original_searchstrat_id = original_searchstrat_id, 
                searchstrat_id = searchstrat_id, 
                evidence_review_id = evidence_review_id, 
                topic_specific_overall_flag = True, 
                vector_search_type = vs_type)
            
            #ranked results output 
            rrf_sim_result_df = self.sql_procedures.rrf_combine_results(
                searchstrat_id = searchstrat_id, 
                evidence_review_id = evidence_review_id,
                original_searchstrat_id = original_searchstrat_id, 
                )
            #conduct evaluation 
            eval_metrics_df, result_cutoff_df = self._evaluate_vector_search(
                rrf_sim_result_df, 
                evidence_review_id = evidence_review_id, 
                search_strat_id = searchstrat_id,  
                search_type = 'topic-specific', 
                vectorsearch_type = vs_type)
            eval_metrics_df_list.append(eval_metrics_df)

        eval_metrics_topicspecific_all = pd.concat(eval_metrics_df_list)
        return eval_metrics_topicspecific_all


    def _evaluate_vector_search(self, rrf_sim_result_df : pd.DataFrame, evidence_review_id : str, search_strat_id : int, search_type : str, vectorsearch_type : str): 
        '''
        Evaluates RRF combined results on evaluation set on a given serach strategy id 
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
        eval_cls = search_evaluation(database = database_name, search_type = search_type, vector_search = True, logger = self.logger)

        self.logger.info(f'Running vector search evaluation pipeline')
        search_strat_df = pd.read_sql(f"SELECT * FROM search_strategies WHERE search_strategy_id = {search_strat_id}", self.engine)
        

        #retrieve query vector ids  
        if vectorsearch_type == 'one-shot' or vectorsearch_type == 'few-shot': 
            idcol = 'ground_truth_article_id'
            query_vector_df = self.sql_procedures.retrieve_query_vector_ids(search_strat_id = search_strat_id, evidence_review_id = evidence_review_id, idcol = idcol)
        else: 
            query_vector_df = pd.DataFrame({'ground_truth_article_id': []})

        eval_metrics_df, cutoff_df = eval_cls.run_vectorsearch_eval_pipeline(result_set = rrf_sim_result_df, evaluation_set = evaluation_set_df, query_vector_df = query_vector_df, database_name = database_name, search_strat_df = search_strat_df)
        self.logger.info(f'Vector search evaluation pipeline complete')
        return eval_metrics_df, cutoff_df
    
    def _check_embeddings_exist(self): 
        """Check embedding tables and determine which need regeneration"""
        
        tables_to_regenerate = []
        embedding_stats = {}
        
        table_checks = {
            'query_goldset_embeddings': {
                'df': self.sql_procedures.retrieve_query_goldset(evidence_review_id='overall'),
                'desc': 'goldset'
            },
            'query_evidencereview_topic_embeddings': {
                'df': self.sql_procedures.retrieve_query_evidencereview_topics(evidence_review_id='overall'),
                'desc': 'evidence review topic'
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
    
        if tables_to_regenerate: 
            self.logger.info(f'Initializing embedding model: {self.model_name}')
            self.model = AutoAdapterModel.from_pretrained(self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)
            self.model.to(self.device)

            

            # Generate specific embeddings as needed
            if 'query_goldset_embeddings' in tables_to_regenerate:
                self.sql_procedures.setup_embeddings_table(materialized_view_name = 'query_goldset_view', embedding_table_name = 'query_goldset_embeddings', linking_table_name = 'ground_truth_articles', linking_id = 'ground_truth_article_id', id_dtype = 'INT')
                self.generate_goldset_embeddings()
                
            if any('searchspace_database' in table for table in tables_to_regenerate):
                for table in tables_to_regenerate: 
                    for database in [1, 2, 3, 4]: 
                        self.sql_procedures.setup_searchspace_database_embedding_table(database_id = database)
                    database_name = table.split('_')[2]
                    database_id = pd.read_sql(f"SELECT database_id FROM databases WHERE database_name = '{database_name}'", self.engine).iloc[0,0]
                    self.generate_searchspace_embeddings(database_id)

            if 'query_evidencereview_topic_embeddings' in tables_to_regenerate: 
                #set up embeddings tables for evidence review topics (to enable zero shot)
                self.sql_procedures.setup_embeddings_table(materialized_view_name = 'query_evidencereview_topic_view', embedding_table_name = 'query_evidencereview_topic_embeddings', linking_table_name = 'evidence_reviews', linking_id = 'evidence_review_id', id_dtype = 'VARCHAR(50)')
                self.generate_evidencereview_topic_embeddings()


    def run_vector_search(self, search_type, vector_search_type):

        if search_type == 'overarching': 
            eval_metrics_df = self.overarching_vector_search_with_rrf(vector_search_type)
            return eval_metrics_df
        elif search_type == 'topic-specific': 
            eval_metrics_df = self.topic_specific_vector_search_with_rrf(vector_search_type)
            return eval_metrics_df



    


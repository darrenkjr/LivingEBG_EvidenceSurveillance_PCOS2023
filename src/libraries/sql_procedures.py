from sqlalchemy import create_engine
from sqlalchemy import text
from dotenv import load_dotenv
import pandas as pd 
from pathlib import Path
from logging_config import LoggerConfig
import os 
class sql_procedures: 

    def __init__(self, logger = None, engine = None): 
        self.logger = logger
        self.engine = engine

    def retrieving_overarching_searchstrat(self): 

        '''
        Provides a datafram of search strategies that are overarching. First join search strat table to the reference tables (database, and search strategy, and search type)
        Then filter for overarching search strategies 
        '''

        query = """
            WITH overarching_search_strat AS (
                SELECT DISTINCT ON (ss.evidence_review_id)
                    ss.search_strategy_id, 
                    ss.evidence_review_id, 
                    ss.database_id, 
                    ss.searchstrat_year_start, 
                    ss.search_type_id, 
                    d.name as database_name, 
                    st.name as search_type_name, 
                    ss_type.name as search_strategy_type_name
                FROM search_strategies ss 
                JOIN databases d ON ss.database_id = d.database_id 
                JOIN search_types st ON ss.search_type_id = st.search_type_id 
                JOIN search_strategy_types ss_type ON ss.search_strategy_type_id = ss_type.search_strategy_type_id 
                WHERE st.name = 'overarching'     
                    AND ss.evidence_review_id = 'overall' 
                ORDER BY                          
                    ss.evidence_review_id,
                    ss.search_strategy_id,
                    ss.search_type_id,
                    ss.database_id
            )"""

        with self.engine.connect() as conn: 
            overarching_searchstrat_df = pd.read_sql(query, conn)
            return overarching_searchstrat_df
        
    def retrieve_search_result_articles(self, search_strat_id): 
        '''
        Retrieves search result articles for a given search strat id. Must join search result table to search result article table first, then query based on this 
        '''
        query = """
            SELECT 
                sr.search


        """

    def create_goldset_view(self): 

        

        '''
        Creates a view of the goldset, represnting articles that were included in the previous versions of the guideline, and articles at random from the new SRs (3)

        
        '''

        self.logger.info(f'Creating goldset view')

        original_goldset = pd.read_parquet(Path(__file__).parent.parent / 'dataset' / 'combined_goldset.parquet')
        gold_set_ids = "'" + "','".join(map(str, original_goldset['included_article_id'])) + "'"

        #recreate view in SQL 
        queries = [f"""
            CREATE MATERIALIZED VIEW IF NOT EXISTS goldset_view AS 
                SELECT * FROM ground_truth_articles 
                WHERE ground_truth_article_id IN ({gold_set_ids})
                ORDER BY ground_truth_article_id;
                """,
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_goldset_view_id 
                ON goldset_view (ground_truth_article_id);
                """
                ]
    
        try: 
            with self.engine.begin() as conn:
                for query in queries: 
                    conn.execute(text(query))
                self.logger.info(f'Goldset view created')
        except Exception as e: 
            self.logger.error(f'Error creating goldset view: {e}')
            raise e 
        
        #condcut check is is 1 : 1 with original goldset 
        with self.engine.connect() as conn: 
            sql_goldset = pd.read_sql("SELECT * FROM goldset_view", conn)
            assert set(original_goldset['included_reference']) == set(sql_goldset['included_reference']) and len(original_goldset) == len(sql_goldset)
            self.logger.info(f'Goldset view is 1 : 1 with original goldset')
            return sql_goldset


        
    def create_evaluation_set_view_2017included(self): 
        '''
        Creates a view of the evaluation set composed of articles newly included in the PCOS Guidelinee 2017. This is 
        used for the evaluation of search results, and is constructed from the ground truth table. 

        The view involves joining ground truth articles with the evidence reivews, and then using that to link to the search strategies, and then finally grabbing the searchstrat_year_start column. 
        It then filters for articles where the searchstrat_year_start is less than or equal to the year_pub_extract column, or where 
        the searchstrat_year_start is null. 
    
        '''

        queries = [
            """ 
             CREATE MATERIALIZED VIEW IF NOT EXISTS groundtruth_eval_set AS 
                WITH ground_truth_search_strat_link AS (
                    SELECT 
                        gt.*, 
                        ss.searchstrat_year_start
                    FROM 
                        ground_truth_articles gt 
                        LEFT JOIN (
                            SELECT DISTINCT ON (evidence_review_id)
                                *
                            FROM search_strategies
                            ORDER BY 
                                evidence_review_id,
                                search_strategy_id  
                        ) ss ON gt.evidence_review_id = ss.evidence_review_id
                )
                SELECT DISTINCT 
                    ground_truth_article_id, 
                    evidence_review_id, 
                    title, 
                    abstract, 
                    extracted_publication_year, 
                    included_reference,
                    author_year_format,
                    assessed_rob, 
                    retrieved_oa_id, 
                    retrieved_embase_id, 
                    retrieved_pubmed_id
                FROM 
                    ground_truth_search_strat_link gtss 
                WHERE 
                    (gtss.searchstrat_year_start <= gtss.extracted_publication_year
                    OR gtss.searchstrat_year_start IS NULL) 
                    AND evidence_review_id NOT IN ('5.7.5', '5.7.1') -- Drop known evidence review with known no new articles
                ORDER BY 
                    ground_truth_article_id;
                WITH DATA;
                """,
                """ 
                CREATE UNIQUE INDEX IF NOT EXISTS idx_groundtruth_eval_2017_id 
                ON groundtruth_eval_set_2017new (ground_truth_article_id);
                """
        ]
        
        try:
            with self.engine.begin() as conn:
                self.logger.info(f'Creating ground truth evaluation set materalized view')
                # Create view
                for query in queries: 
                    conn.execute(text(query))
                result = conn.execute(text("SELECT COUNT(*) FROM groundtruth_eval_set_2017new"))
                actual_count = result.scalar()
                
                # Compare with expected
                expected_count = pd.read_parquet(Path(__file__).parent.parent / 'dataset' / 'groundtruth_eval.parquet').shape[0]
                
                # Check counts but only warn if mismatch
                if actual_count != expected_count:
                    self.logger.warning(f"Row count mismatch: expected {expected_count} rows, but got {actual_count}. Check SQL query.")
                else:
                    self.logger.info(f"Ground truth evaluation set view created successfully with {actual_count} rows")
                
        except Exception as e:
            self.logger.error(f"Error creating ground truth evaluation set view: {e}")
            raise
            
        eval_df = pd.read_sql("SELECT * FROM groundtruth_eval_set_2017new", self.engine)
        return eval_df
    # Compare the data directly

    def setup_embeddings_table(self, materialized_view_name, embedding_table_name, linking_id, id_dtype): 
        '''
        Creates a table in the sql database for storing embeddings, linked to a materialized view.
        Args: 
            materialized_view_name: name of the materialized view (e.g., 'groundtruth_eval_set_2017new')
            embedding_table_name: name of the embeddings table (e.g., 'article_embeddings')
            linking_id: the id column for linking (e.g., 'ground_truth_article_id')
        '''


        try:
            with self.engine.begin() as conn:
                check_view_query = f"""
                    SELECT EXISTS (
                        SELECT FROM pg_matviews 
                        WHERE matviewname = '{materialized_view_name}'
                    );
                    """
                result = conn.execute(text(check_view_query))
                if not result.scalar():
                    raise ValueError(f"Materialized view {materialized_view_name} does not exist")
        

                setup_query = f"""
                    CREATE EXTENSION IF NOT EXISTS vector;, 

                    CREATE TABLE IF NOT EXISTS {embedding_table_name} (
                        {linking_id} {id_dtype} PRIMARY KEY,
                        embedding vector(768),
                        FOREIGN KEY ({linking_id}) 
                            REFERENCES {materialized_view_name}({linking_id})
                            ON DELETE CASCADE
                    );
                """
                conn.execute(text(setup_query))
                
                self.logger.info(f'Embeddings table {embedding_table_name} created')
        except Exception as e: 
            self.logger.error(f'Error creating embeddings table {embedding_table_name}: {e}')
            raise e 

    def add_embeddings_to_sql(self, embedding_table_name, input_df, linking_id): 
        '''
        Adds embeddings to a table in the sql database. 

        ARg: 
            embedding_table_name: the name of the table to store the embeddings in. For example 'goldset_embeddings'
            input_df: Dataframe containing embedding and linking id columns 
            linking_id: the name of the column containing ids that links to the embeddings to the original input df
        '''

        embedding_data = [
            {
                linking_id : article_id, 
                'embedding' : embedding.to_list()
            }
            for article_id, embedding in zip(input_df[linking_id], input_df['embeddings'])
        ]

        upsert_query = f"""
            INSERT INTO {embedding_table_name} ({linking_id}, embedding) 
            VALUES (:article_id, :embedding::vector) 
            ON CONFLICT ({linking_id}) DO UPDATE SET embedding = EXCLUDED.embedding;
        """
        try: 
            with self.engine.begin() as conn: 
                conn.execute(text(upsert_query), embedding_data)
                self.logger.info(f'Embeddings added to {embedding_table_name}')
        except Exception as e: 
            self.logger.error(f'Error adding embeddings to {embedding_table_name}: {e}')
            raise e 


if __name__ == '__main__': 
    logger = LoggerConfig.setup_logger(logger_name = 'sql_view_creator')
    load_dotenv()
    db_name = os.getenv('DB_NAME')
    db_user = os.getenv('DB_USER')
    db_pwd = os.getenv('DB_PWD')
    db_host = os.getenv('DB_HOST')
    sql_engine = create_engine(f'postgresql://{db_user}:{db_pwd}@localhost:5432/{db_name}')
    sql_instance = sql_procedures(logger, sql_engine)
    sql_instance.create_evaluation_set_view_2017included()
    sql_instance.create_goldset_view()
                              
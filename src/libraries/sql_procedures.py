from sqlalchemy import create_engine
from sqlalchemy import text
from dotenv import load_dotenv
import pandas as pd 
from pathlib import Path
from libraries.logging_config import LoggerConfig
import os 
from pgvector.sqlalchemy import Vector
from libraries.sql_data_migration import sql_data_migration

class sql_procedures: 

    def __init__(self, logger = None, engine = None): 
        self.logger = logger
        self.engine = engine
        #register pgvector with sqlalchemy 
        #create extension vector if it doesnt exist 
        with self.engine.connect() as conn: 
            conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))


    def retrieve_searchstrat(self,search_type): 

        '''
        Provides a datafram of search strategies that are overarching. First join search strat table to the reference tables (database, and search strategy, and search type)
        Then filter for overarching search strategies 
        '''

        query = f"""
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
                WHERE st.name = '{search_type}' 
                    AND ss.vector_search = FALSE
                ORDER BY                          
                    ss.evidence_review_id,
                    ss.search_strategy_id,
                    ss.search_type_id,
                    ss.database_id
            )"""

        with self.engine.connect() as conn: 
            overarching_searchstrat_df = pd.read_sql(query, conn)
            return overarching_searchstrat_df
    
 
        
    def _retrieve_search_result_articles_by_searchstratid(self, searchstrat_id): 
            '''
            Returns search result articles for a given search strategy id 
            '''
            query = f"""
                SELECT * FROM search_result_articles WHERE search_strategy_id = {searchstrat_id}
            """
            with self.engine.connect() as conn: 
                search_result_articles_df = pd.read_sql(query, conn)
                return search_result_articles_df    

        
    def retrieve_search_result_articles_databaseview(self, database_id): 
        '''
        Returns unique search result articles for a given database id. First join search result articles to search strategies, then filter for database id 
        '''
        query = f"""
            SELECT
                sr.search_result_article_id, 
                sr.title, 
                sr.abstract, 
                sr.original_id
            FROM search_result_articles sr 
            JOIN search_strategies ss ON sr.search_strategy_id = ss.search_strategy_id 
            WHERE ss.database_id = {database_id}
        """
        with self.engine.begin() as conn: 
            search_result_articles_df = pd.read_sql(query, conn)
            return search_result_articles_df

    def create_querygoldset_view(self): 

        '''
        Creates a materalized view from the ground truth table, containing 3 random articles that were included in the previous versions of the guideline, 
        and 3 random articles from the new SRs 
        
        '''

        self.logger.info(f'Creating goldset view')
        #recreate view in SQL - actually change this to random (3) articles per evidence review 
        ground_truth_df = pd.read_parquet(Path(__file__).parent.parent / 'dataset' / 'fullgroundtruth_valid_apimerge_df.parquet')
        ground_truth_df['year_pub_extract'] = pd.to_numeric(
                ground_truth_df['year_pub_extract'], 
                errors='coerce'
            ).astype('Int64')  
        ground_truth_df.reset_index(inplace = True)

        _srupdate_previous = ground_truth_df.query('sr_update == "Y" & title.notna() & abstract.notna() & year_pub_extract <= searchstrat_year_start').copy()
        _srupdate_goldset = _srupdate_previous.groupby('question_id').apply(
            lambda x: x.sample(
                n=min(3, len(x)),  # x is already filtered for non-empty title/abstract
                replace=False,
                random_state=42
            )
        ).reset_index(drop=True)

        _newsr_articles = ground_truth_df.query('sr_update != "Y" & title.notna() & abstract.notna()').copy()
        _newsr_goldset = _newsr_articles.groupby('question_id').apply(
            lambda x: x.sample(
                n=min(3, len(x)),  # x is already filtered for non-empty title/abstract
                replace=False,
                random_state=42
            )
        ).reset_index(drop=True)

        goldset_df = pd.concat([_srupdate_goldset, _newsr_goldset], ignore_index = True)
        goldset_ids = "'" + "','".join(map(str, goldset_df['included_article_id'])) + "'"


        queries = [f"""
            CREATE MATERIALIZED VIEW IF NOT EXISTS query_goldset_view AS 
                SELECT * FROM ground_truth_articles 
                WHERE ground_truth_article_id IN ({goldset_ids})
                ORDER BY ground_truth_article_id;
                """,
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_query_goldset_view_id 
                ON query_goldset_view (ground_truth_article_id);
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
        

        
    def create_evaluation_set_view_2017included(self): 
        '''
        Creates a view of the evaluation set composed of articles newly included in the PCOS Guidelinee 2017. This is 
        used for the evaluation of search results, and is constructed from the ground truth table. 

        The view involves joining ground truth articles with the evidence reivews, and then using that to link to the search strategies, and then finally grabbing the searchstrat_year_start column. 
        It then filters for articles where the searchstrat_year_start is less than or equal to the year_pub_extract column, or where 
        the searchstrat_year_start is null. 
    
        '''

        view_name = "groundtruth_eval_set_2017new"  # Define name once
        
        queries = [
            f""" 
            CREATE MATERIALIZED VIEW IF NOT EXISTS {view_name} AS 
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
                    AND evidence_review_id NOT IN ('5.7.5', '5.7.1')
                ORDER BY 
                    ground_truth_article_id
            """,
            f""" 
            CREATE UNIQUE INDEX IF NOT EXISTS idx_groundtruth_eval_2017_id 
            ON {view_name} (ground_truth_article_id)
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
        

                setup_query = [f"""
                    CREATE EXTENSION IF NOT EXISTS vector;""", 

                    f"""CREATE TABLE IF NOT EXISTS {embedding_table_name} (
                        {linking_id} {id_dtype} PRIMARY KEY,
                        embeddings vector(768),
                        FOREIGN KEY ({linking_id}) 
                            REFERENCES ground_truth_articles({linking_id})
                            ON DELETE CASCADE
                    );
                """]
                for query in setup_query: 
                    conn.execute(text(query))
                
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
            linking_id: the name of the column containing ids that links the embeddings to the original input df, for example 
                        ('ground_truth_article_id', or 'search_result_article_id')
        '''

        try:
            # Create list of dictionaries with direct values
            data = [
                {
                    linking_id: int(id_val),
                    'embeddings': emb.flatten().tolist()  # Just pass as list, no Vector wrapper needed
                }
                for id_val, emb in zip(input_df[linking_id], input_df['embeddings'])
            ]

            upsert_query = f"""
                INSERT INTO {embedding_table_name} ({linking_id}, embeddings) 
                VALUES (:{linking_id}, :embeddings)
                ON CONFLICT ({linking_id}) 
                DO UPDATE SET embeddings = EXCLUDED.embeddings;
            """
            
            # Use simple insert
            with self.engine.begin() as conn:
                conn.execute(
                    text(upsert_query),
                    data
                )
                self.logger.info(f'Embeddings added to {embedding_table_name}')

        except Exception as e:
            self.logger.error(f'Error adding embeddings to {embedding_table_name}: {e}')
            raise e
        
    def setup_searchspace_database_embedding_table(self, database_id): 
        '''
        Creates a table in the sql database for storing embeddings, for search result articles associated with a given database id 
        '''
        #get database name 
        with self.engine.begin() as conn: 
            result = conn.execute(text(f"SELECT database_name FROM databases WHERE database_id = {database_id}"))
            database_name = result.scalar()

        query = f"""
            CREATE TABLE IF NOT EXISTS searchspace_database_{database_name}_embeddings (
                search_result_article_id INT PRIMARY KEY,
                original_id VARCHAR(50),
                embeddings vector(768),
                FOREIGN KEY (search_result_article_id) 
                    REFERENCES search_result_articles(search_result_article_id)
                    ON DELETE CASCADE
            );
        """

        try: 
            with self.engine.begin() as conn: 
                conn.execute(text(query))
            self.logger.info(f'Searchspace database embedding table created for database: {database_name}')
        except Exception as e: 
            self.logger.error(f'Error creating searchspace database embedding table for database: {database_name}: {e}')
            raise e 
        
    def create_new_searchstrat_vectorsearch(self, input_df): 
        '''
        Creates a new search strategy, with a vector search flag.

        Args: 
            input_df: dataframe containing search strategy information 
        Returns: 
            None, but creates a new search strategy entry in the databse 
        '''
        #add a vector search flag to the search strategy table 
        add_vs_flag_col = f"""
            DO $$ 
            BEGIN 
                IF NOT EXISTS (
                    SELECT 1 
                    FROM information_schema.columns 
                    WHERE table_name='search_strategies' 
                    AND column_name='vector_search'
                ) THEN 
                    ALTER TABLE search_strategies 
                    ADD COLUMN vector_search BOOLEAN DEFAULT FALSE;
                END IF;
            END $$;
        """

        add_new_searchtype_query = """
            DO $$ 
            BEGIN 
                IF NOT EXISTS (
                    SELECT 1 FROM search_types 
                    WHERE name = 'boolean-kw_vectorsearch'
                ) THEN 
                    INSERT INTO search_types (search_type_id, name) 
                    VALUES (3, 'boolean-kw_vectorsearch');
                END IF;
            END $$;
        """

        #insert new search stragies into the table 
        with self.engine.begin() as conn: 
            latest_search_strat_id = conn.execute(text("SELECT MAX(search_strategy_id) FROM search_strategies")).scalar()
        input_df['search_strategy_id'] = input_df.index + latest_search_strat_id + 1


        with self.engine.begin() as conn: 
            conn.execute(text(add_vs_flag_col))
            conn.execute(text(add_new_searchtype_query))
            #check for existing ids 
            filtered_input_df = sql_data_migration._prior_id_check(input_df, 'search_strategy_id', 'search_strategies', engine = self.engine, logger = self.logger)
            filtered_input_df.to_sql('search_strategies', conn, if_exists = 'append', index = False)

        return filtered_input_df

    def run_vector_search(self, original_searchstrat_id, searchstrat_id, evidence_review_id, topic_specific_overall_flag = False): 
        '''
        Runs vector search for a given search strategy id, for a given evidence_review_id
        First - retrieve query vectors, then retrieve searchspace embeddings with: 
            given search strat id, corresponding database id, retrieve search result articles and correspdoing embeddings 

        Arg: 
            original_searchstrat_id: the original search strategy that is used to retrieve the correct searchresult articles 
            searchstrat_id: the new search strategy id to run vector search for 
            evidence_review_id: the evidence review id to run vector search for.
                eg: overall = all goldset articles are query vectors, else, goldset articles associated with evidence review id are query vectors 

        Returns: 
            view of search results, with cosine similiarity socres and corresponding rank
        '''

        self.logger.info(f'Cleaning up previous vector search results if applicable')

        
        if topic_specific_overall_flag: 
            goldset_query_filter = f"WHERE evidence_review_id = {evidence_review_id}"
        else: 
            goldset_query_filter = f""
            evidence_review_id = 'overall'


        clean_up_query = f"""
            DROP MATERIALIZED VIEW IF EXISTS vector_search_results_{searchstrat_id}_{evidence_review_id} CASCADE;
        """



        vector_search_query = f""" 

        CREATE MATERIALIZED VIEW IF NOT EXISTS "vector_search_results_{searchstrat_id}_{evidence_review_id}" AS 
        WITH query_vectors AS (
            SELECT ground_truth_article_id, embedding 
            FROM goldset_embeddings {goldset_query_filter}
        ),

        search_result_articles_emb AS (
            SELECT
                sra.search_result_article_id, 
                sra.original_id, 
                ss.database_id, 
                ss.search_strategy_id, 
                CASE    
                    WHEN ss.database_id = 1 THEN (
                        SELECT embedding 
                        FROM searchspace_database_pubmed_embeddings 
                        WHERE search_result_article_id = sra.search_result_article_id
                    )
                    WHEN ss.database_id = 2 THEN (
                        SELECT embedding 
                        FROM searchspace_database_medline_embeddings 
                        WHERE search_result_article_id = sra.search_result_article_id
                    )
                    WHEN ss.database_id = 3 THEN (
                        SELECT embedding 
                        FROM searchspace_database_embase_embeddings 
                        WHERE search_result_article_id = sra.search_result_article_id
                    )
                    WHEN ss.database_id = 4 THEN (
                        SELECT embedding 
                        FROM searchspace_database_openalex_embeddings 
                        WHERE search_result_article_id = sra.search_result_article_id
                    )
                END as embedding 
            FROM search_result_articles sra 
            JOIN search_strategies ss ON sra.search_strategy_id = ss.search_strategy_id 
            WHERE ss.search_strategy_id = {original_searchstrat_id}
        ),

        raw_cosine_sim_ranked AS (
            SELECT 
                query_vectors.ground_truth_article_id, 
                search_result_articles_emb.search_result_article_id, 
                search_result_articles_emb.search_strategy_id, 
                1 - (query_vectors.embedding <=> search_result_articles_emb.embedding) AS cosine_similarity,
                ROW_NUMBER() OVER (
                PARTITION BY ground_truth_article_id 
                ORDER BY cosine_similarity DESC) as rank
            FROM query_vectors 
            CROSS JOIN search_result_articles_emb  
            WHERE search_result_articles_emb.embedding IS NOT NULL
            )

        SELECT * FROM raw_cosine_sim_ranked;

        """
        try:
            with self.engine.begin() as conn: 

                self.logger.info(f'Cleaning up previous vector search results for search strategy id: {searchstrat_id} and corresponding evidence review id: {evidence_review_id}')
                conn.execute(text(clean_up_query))

                self.logger.info(f'Running vector search for search strategy id: {searchstrat_id} and corresponding evidence review id: {evidence_review_id}')
                conn.execute(text(vector_search_query))

                self.logger.info(f'Vector search completed for search strategy id: {searchstrat_id} and corresponding evidence review id: {evidence_review_id}')

        except Exception as e: 
            self.logger.error(f'Error running vector search for search strategy id: {searchstrat_id} and corresponding evidence review id: {evidence_review_id}: {e}')
            raise e 

    def rrf_combine_results(self, searchstrat_id, evidence_review_id): 
        '''
        Combines the results of the vector search into a single dataframe, with RRF scores 
        '''

        query = f"""
            SELECT * FROM vector_search_results_{searchstrat_id}_{evidence_review_id}
        """

        with self.engine.begin() as conn: 
            sim_df = pd.read_sql(query, conn)
        
        sim_df['rrf_score'] = 1 / (60 + sim_df['rank'])
        rrf_sim_df = sim_df.groupby('search_result_article_id').agg({
            'rrf_score': 'sum',
            'raw_cosine_sim': 'mean',
            'search_strategy_id': 'first'
            }).reset_index().copy()

        rrf_sim_df['combined_rank'] = rrf_sim_df['rrf_score'].rank(method = 'first', ascending = False)
        rrf_sim_df = rrf_sim_df.sort_values(by = ['rrf_score'], ascending = False)
        
        return rrf_sim_df
    
    def retrieve_evaluation_set(self, evidence_review_id): 
        '''
        Retrieves the evaluation set for a given evidence review id 
        '''
        if evidence_review_id == 'overall': 
            query = f"""
                SELECT * FROM groundtruth_eval_set_2017new
            """
        else: 
            query = f"""
                SELECT * FROM groundtruth_eval_set_2017new WHERE evidence_review_id = {evidence_review_id}
            """

        with self.engine.begin() as conn: 
            eval_set_df = pd.read_sql(query, conn)
        return eval_set_df

    def retrieve_query_goldset(self, evidence_review_id): 
        '''
        Retrieves the goldset for a given evidence review id 
        
        '''
        if evidence_review_id == 'overall': 
            query = f"""
                SELECT * FROM query_goldset_view
            """
        else: 
            query = f"""
            SELECT * FROM query_goldset_view WHERE evidence_review_id = {evidence_review_id}
            """
        with self.engine.begin() as conn: 
            query_goldset_df = pd.read_sql(query, conn)
        return query_goldset_df
    
    
    def retrieve_evidence_review_ids(self, search_type: str) -> list: 
        '''
        Retrieves unique evidence review ids for topic specific searches 

        Arg: 
            search_type: the search type to retrieve evidence review ids for. 
                eg: 'overarching' or 'topic-specific'
        Returns: 
            list of evidence review ids 
        '''
        query = f"""
            SELECT DISTINCT evidence_review_id 
                FROM search_strategies
                JOIN search_types ON search_strategies.search_type_id = search_types.search_type_id
                WHERE search_types.name = {search_type}
        """
        with self.engine.begin() as conn: 
            evidence_review_ids = pd.read_sql(query, conn)['evidence_review_id'].tolist()
        return evidence_review_ids

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
    sql_instance.create_query_goldset_view()
                              
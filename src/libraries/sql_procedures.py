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
        if engine is None: 
            wsl_flag = os.environ.get('WSL_DISTRO_NAME') is not None
    
            if wsl_flag: 
                os.environ['PGHOST'] = '/var/run/postgresql' 
            else: 
                os.environ['PGHOST'] = 'localhost' 
            load_dotenv()
            db_name = os.getenv('DB_NAME')
            db_user = os.getenv('DB_USER')
            db_pwd = os.getenv('DB_PWD')
            db_host = os.getenv('DB_HOST')
            db_port = os.getenv('DB_PORT')
            self.engine = create_engine(f'postgresql://{db_user}:{db_pwd}@{db_host}:{db_port}/{db_name}')
        else: 
            self.engine = engine
        #register pgvector with sqlalchemy 
        #create extension vector if it doesnt exist 
        with self.engine.connect() as conn: 
            conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))


    def retrieve_searchstrat(self, search_type): 

        '''
        Provides a datafram of search strategies that are overarching. First join search strat table to the reference tables (database, and search strategy, and search type)
        Then filter for overarching search strategies 
        '''
        if search_type == 'overarching': 
            distinct_on_clause = ''
        elif search_type == 'topic-specific': 
            distinct_on_clause = 'DISTINCT ON (ss.evidence_review_id)'
        else: 
            raise ValueError(f'Invalid search type: {search_type}')

        query = f"""
            WITH search_strat AS (
                SELECT {distinct_on_clause}
                    ss.search_strategy_id, 
                    ss.evidence_review_id, 
                    ss.database_id, 
                    ss.searchstrat_year_start, 
                    ss.searchstrat_year_end,
                    ss.search_type_id, 
                    ss.searchdetail_file_path,
                    d.database_name as database_name, 
                    st.name as search_type_name, 
                    ss_type.name as search_strategy_type_name
                FROM search_strategies ss 
                JOIN databases d ON ss.database_id = d.database_id 
                JOIN search_types st ON ss.search_type_id = st.search_type_id 
                JOIN search_strategy_types ss_type ON ss.search_strategy_type_id = ss_type.search_strategy_type_id 
                WHERE st.name = '{search_type}' 
                    AND ss.vector_search = 'no'
                ORDER BY                          
                    ss.evidence_review_id,
                    ss.search_strategy_id,
                    ss.search_type_id,
                    ss.database_id
                    )

            SELECT * FROM search_strat
            """

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
        groundtruth_evidence_review_ids = ground_truth_df['question_id'].unique()
        _goldset_evidence_review_ids = pd.concat([_srupdate_goldset, _newsr_goldset])['question_id'].unique()
        missing_ids = list(set(groundtruth_evidence_review_ids) - set(_goldset_evidence_review_ids))
        self.logger.info(f'Missing evidence review ids from goldset detected: {missing_ids}, attempt to fix, ie: treating as new sr')
        _missing_ids_df = ground_truth_df.query('question_id in @missing_ids').copy()
        _missing_ids_df = _missing_ids_df.groupby('question_id').apply(
            lambda x: x.sample(
                n=min(3, len(x)),  # x is already filtered for non-empty title/abstract
                replace=False,
                random_state=42
            )
        ).reset_index(drop=True)
        
        goldset_df = pd.concat([_srupdate_goldset, _newsr_goldset, _missing_ids_df], ignore_index = True)
        goldset_evidence_review_ids = goldset_df['question_id'].unique()
        assert set(goldset_evidence_review_ids) == set(groundtruth_evidence_review_ids), f'Goldset evidence review ids are not consistent with ground truth evidence review ids. Missing ids : {set(groundtruth_evidence_review_ids) - set(_goldset_evidence_review_ids)}'
        
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
        
    def create_query_evidencereview_topic_view(self): 
        '''
        Creates a view of the evidence reviews, with the topic of the evidence review as a column 
        '''

        #get relevant evidence review ids 
        evidence_review_ids = pd.read_sql(
            f"""SELECT DISTINCT er.evidence_review_id 
            FROM evidence_reviews er
            JOIN search_strategies ss 
            ON er.evidence_review_id = ss.evidence_review_id
            WHERE er.evidence_review_id != 'overall'""", self.engine)
         
        evidence_review_ids = "'" + "','".join(map(str, evidence_review_ids['evidence_review_id'])) + "'"


        queries = [f"""
            CREATE MATERIALIZED VIEW IF NOT EXISTS query_evidencereview_topic_view AS 
                SELECT * FROM evidence_reviews
                   WHERE evidence_review_id IN ({evidence_review_ids})
                """,
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_query_evidencereview_topic_view_id 
                ON query_evidencereview_topic_view (evidence_review_id);
                """
                ]
    
        try: 
            with self.engine.begin() as conn:
                for query in queries: 
                    conn.execute(text(query))
                self.logger.info(f'Query evidence review topic view created')
        except Exception as e: 
            self.logger.error(f'Error creating evidence review topic view: {e}')
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

    def setup_embeddings_table(self, materialized_view_name, embedding_table_name, linking_table_name, linking_id, id_dtype): 
        '''
        Creates a table in the sql database for storing embeddings, linked to a materialized view.
        Args: 
            materialized_view_name: name of the materialized view
            embedding_table_name: name of the embeddings table (e.g., 'article_embeddings')
            linking_table_name: name of the table that the embeddings are linked to (e.g., 'ground_truth_articles')
            linking_id: the id column for linking (e.g., 'ground_truth_article_id')
            id_dtype: the data type of the id column (e.g., 'INT')
        '''


        try:
            self.logger.info(f'Setting up embeddings table for {materialized_view_name}')
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
                            REFERENCES {linking_table_name}({linking_id})
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
                    linking_id: int(id_val) if linking_id == 'ground_truth_article_id' else id_val,
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
        
    def create_new_searchstrat_vectorsearch(self, input_df, vector_search_type, search_type_id): 
        '''
        Creates a new search strategy, with a vector search flag.

        Args: 
            input_df: dataframe containing search strategy information 
        Returns: 
            None, but creates a new search strategy entry in the databse 
        '''

        with self.engine.begin() as conn: 
            # First check which original search strategies already have vector search versions
            #grab column names 
            col_query = """
                SELECT column_name 
                FROM information_schema.columns
                WHERE table_name = 'search_strategies' 
                AND table_schema = 'public';
            """
            relevant_cols = pd.read_sql(col_query, conn)['column_name'].tolist()

            #grab existing vector search strategies 
            existing_check = f"""
                SELECT *
                FROM search_strategies 
                WHERE vector_search = '{vector_search_type}' AND search_type_id = {search_type_id};
            """
            existing_df = pd.read_sql(existing_check, conn)

            if existing_df.empty: 
                self.logger.info(f'No existing vector search strategies found. Adding all input search strategies')
                new_df = input_df.copy()
            else:
                self.logger.info(f'Existing vector search strategies found for vector search type: {vector_search_type}. Number of existing vector search strategies: {existing_df.shape[0]}')
                self.logger.info(f'Performing checks')
                #check input data with existing data with merge 
                compare_cols = relevant_cols.copy()
                compare_cols.remove('search_strategy_id')
                comparison = input_df[compare_cols].merge(
                    existing_df[compare_cols], 
                    on = compare_cols, 
                    how = 'left', 
                    indicator = True)
                
                new_df = input_df.loc[comparison['_merge'] == 'left_only'].copy()
            
            if len(new_df) > 0: 
                self.logger.info(f'Found {len(new_df)} new search strategies to add')
                
                # Get latest ID and generate new sequential IDs
                latest_search_strat_id = conn.execute(text(
                    "SELECT COALESCE(MAX(search_strategy_id), 0) FROM search_strategies;"
                )).scalar()

                new_df['search_strategy_id'] = range(
                    latest_search_strat_id + 1, 
                    latest_search_strat_id + 1 + len(new_df)
                )

                # No need to drop _merge as it's not in new_df
                new_df[relevant_cols].to_sql('search_strategies', conn, if_exists='append', index=False)
                self.logger.info(f"Created {len(new_df)} new vector search strategies")
                return new_df
            
            else: 
                self.logger.info('No new search strategies to add')
                try: 

                    assert len(existing_df) == len(input_df), 'The number of existing search strategies does not match the number of input search strategies'
                    existing_df['original_search_strategy_id'] = input_df['original_search_strategy_id']
                    return existing_df
                except AssertionError as e: 
                    self.logger.error(f'Error matching existing search strategies to input original search strategies: {e}')
                    raise e 
                


    def run_vector_search(self, original_searchstrat_id, searchstrat_id, evidence_review_id, vector_search_type, topic_specific_overall_flag = False): 
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

 

        query_table_name, query_vector_id, goldset_query_filter, evidence_review_id = self._prepare_query_vector_queries(topic_specific_overall_flag, vector_search_type, evidence_review_id)

        vector_search_query = f""" 

        CREATE MATERIALIZED VIEW IF NOT EXISTS "vector_search_results_{searchstrat_id}_{evidence_review_id}" AS 
        WITH query_vectors AS (
            SELECT {query_vector_id}, embeddings 
            FROM {query_table_name} {goldset_query_filter}
        ),

        -- Retrieve search result articles and embeddings, depending on the database id,  taken from the orignal search strategy id
        
        search_result_articles_emb AS (
            SELECT
                sra.search_result_article_id, 
                sra.original_id, 
                ss.database_id, 
                ss.search_strategy_id, 
                CASE    
                    WHEN ss.database_id = 1 THEN (
                        SELECT embeddings 
                        FROM searchspace_database_pubmed_embeddings 
                        WHERE search_result_article_id = sra.search_result_article_id
                    )
                    WHEN ss.database_id = 2 THEN (
                        SELECT embeddings 
                        FROM searchspace_database_medline_embeddings 
                        WHERE search_result_article_id = sra.search_result_article_id
                    )
                    WHEN ss.database_id = 3 THEN (
                        SELECT embeddings 
                        FROM searchspace_database_embase_embeddings 
                        WHERE search_result_article_id = sra.search_result_article_id
                    )
                    WHEN ss.database_id = 4 THEN (
                        SELECT embeddings 
                        FROM searchspace_database_openalex_embeddings 
                        WHERE search_result_article_id = sra.search_result_article_id
                    )
                END as embeddings 
            FROM search_result_articles sra 
            JOIN search_strategies ss ON sra.search_strategy_id = ss.search_strategy_id 
            WHERE ss.search_strategy_id = {original_searchstrat_id}
        ),

        -- Calculate cosine similarity between query vectors and search result articles  
        raw_cosine_sim AS (
            SELECT 
                query_vectors.{query_vector_id}, 
                search_result_articles_emb.search_result_article_id, 
                search_result_articles_emb.search_strategy_id, 
                1 - (query_vectors.embeddings <=> search_result_articles_emb.embeddings) AS cosine_similarity
            FROM query_vectors 
            CROSS JOIN search_result_articles_emb  
            WHERE search_result_articles_emb.embeddings IS NOT NULL
            ), 

        -- Rank the cosine similarity scores 
        raw_cosine_sim_ranked AS (
            SELECT
                {query_vector_id}, 
                search_result_article_id, 
                search_strategy_id, 
                cosine_similarity, 
                ROW_NUMBER() OVER (
                    PARTITION BY {query_vector_id} 
                    ORDER BY cosine_similarity DESC
                ) as rank
            FROM raw_cosine_sim
            )

        SELECT * FROM raw_cosine_sim_ranked
        ORDER BY {query_vector_id}, rank;
        """

        try:
            with self.engine.begin() as conn: 

                # self.logger.info(f'Cleaning up previous vector search results for search strategy id: {searchstrat_id} and corresponding evidence review id: {evidence_review_id}')
                # conn.execute(text(clean_up_query))

                self.logger.info(f'Running vector search for search strategy id: {searchstrat_id} and corresponding evidence review id: {evidence_review_id} for vector search type: {vector_search_type}')
                conn.execute(text(vector_search_query))
                self.logger.info(f'Vector search completed for search strategy id: {searchstrat_id} and corresponding evidence review id: {evidence_review_id} for vector search type: {vector_search_type}')

        except Exception as e: 
            self.logger.error(f'Error running vector search for search strategy id: {searchstrat_id} and corresponding evidence review id: {evidence_review_id}: {e}')
            raise e 
    
    def retrieve_query_vector_ids(self, search_strat_id, evidence_review_id, idcol) -> pd.DataFrame: 
        '''
        Retrieves the query vector ids for a given search strategy id and evidence review id 
        '''
        query = f"""
            SELECT {idcol} FROM "vector_search_results_{search_strat_id}_{evidence_review_id}"
        """
        with self.engine.begin() as conn:
            return pd.read_sql(query, conn)
        
    def _prepare_query_vector_queries(self, topic_specific_overall_flag, vector_search_type, evidence_review_id): 
        '''
        Prepares the query vector queries for the vector search 
        '''
        if topic_specific_overall_flag:  # Topic-specific case
            if vector_search_type == 'zero-shot': 
                query_table_name = 'query_evidencereview_topic_embeddings'
                id = 'evidence_review_id'
                goldset_query_filter = f"WHERE evidence_review_id = '{evidence_review_id}'"  

            elif vector_search_type == 'one-shot': 
                query_table_name = 'query_goldset_embeddings'
                id = 'ground_truth_article_id'
                goldset_query_filter = f"""
                    JOIN (
                        SELECT DISTINCT ON (qgv.evidence_review_id) 
                            qgv.{id} as selected_article_id,
                            qgv.evidence_review_id
                        FROM query_goldset_view qgv
                        ORDER BY 
                            qgv.evidence_review_id,
                            MD5(qgv.{id}::text)
                        ) selected_goldset_articles 
                    ON {query_table_name}.{id} = selected_goldset_articles.selected_article_id
                    WHERE selected_goldset_articles.evidence_review_id = '{evidence_review_id}'
                """
                
            elif vector_search_type == 'few-shot': 
                query_table_name = 'query_goldset_embeddings'
                id = 'ground_truth_article_id'
                goldset_query_filter = f"""
                    JOIN (
                        SELECT  
                            qgv.{id} as selected_article_id,
                            qgv.evidence_review_id
                        FROM query_goldset_view qgv
                        ORDER BY 
                            qgv.evidence_review_id
                        ) selected_goldset_articles 
                    ON {query_table_name}.{id} = selected_goldset_articles.selected_article_id
                    WHERE selected_goldset_articles.evidence_review_id = '{evidence_review_id}'
                """

        else:  #overarching case 
            evidence_review_id = 'overall'
            if vector_search_type == 'zero-shot': 
                query_table_name = 'query_evidencereview_topic_embeddings'
                id = 'evidence_review_id'
                 #get all evidence review topic embeddings 
                goldset_query_filter = f""

            elif vector_search_type == 'one-shot': 
            
                query_table_name = 'query_goldset_embeddings'
                id = 'ground_truth_article_id'
                #get one article embedding per evidence review id (total query vector should equal all evidence review topic embeddings)
                goldset_query_filter = f"""
                    JOIN (
                        SELECT DISTINCT ON (evidence_review_id) 
                            qgv.{id} as selected_article_id
                        FROM query_goldset_view qgv
                        ORDER BY 
                            qgv.evidence_review_id,
                            MD5(qgv.{id}::text)
                        ) selected_goldset_articles 
                    ON {query_table_name}.{id} = selected_goldset_articles.selected_article_id
                """
            elif vector_search_type == 'few-shot': 
                query_table_name = 'query_goldset_embeddings'
                id = 'ground_truth_article_id'
                #get all prepared goldset embeddings (which amounts to 3 articles per evidence review id)
                goldset_query_filter = f""

        return query_table_name, id, goldset_query_filter, evidence_review_id


    def rrf_combine_results(self, original_searchstrat_id,searchstrat_id, evidence_review_id): 
        '''
        Combines the results of the vector search into a single dataframe, with RRF scores 
        '''

        query = f"""
            SELECT * FROM "vector_search_results_{searchstrat_id}_{evidence_review_id}"
        """
        self.logger.info(f'Retrieving vector search results for search strategy id: {searchstrat_id} and corresponding evidence review id: {evidence_review_id}')
        with self.engine.begin() as conn: 
            sim_df = pd.read_sql(query, conn)
            
        
        #calculate rrf score 
        if not sim_df.empty: 
            self.logger.info(f'Calculating RRF scores')
            sim_df['rrf_score'] = 1 / (60 + sim_df['rank'])
            rrf_sim_df = sim_df.groupby('search_result_article_id').agg({
                'rrf_score': 'sum', #sum of rrf scores across all input query vectors
                'cosine_similarity': 'mean', #averge cosine similiarity across all input query vectors
                'search_strategy_id': 'first' #search strategy id of the input query vector
                }).reset_index().copy()

            rrf_sim_df['combined_rank_rrf'] = rrf_sim_df['rrf_score'].rank(method = 'first', ascending = False) 
            #note - if there is a tie, this will get sequential ranks 
            rrf_sim_df = rrf_sim_df.sort_values(by = ['rrf_score'], ascending = False)

            #retrieve corresponding serach result articles based on search_strat_id (original)
            sra_df = pd.read_sql(f"SELECT * FROM search_result_articles WHERE search_strategy_id = {original_searchstrat_id}", self.engine)
            try: 
                with self.engine.begin() as conn: 
                    assert len(rrf_sim_df) == len(sra_df)
                    self.logger.info(f'rrf_sim_df and sra_df have the same number of rows, moving on to merging')
                    rrf_sim_df = rrf_sim_df.merge(sra_df, on='search_result_article_id', how='left', suffixes = ('_rrf', '_unranked'))
            
            except AssertionError as e: 
                self.logger.debug(f'Current search strat id: {searchstrat_id}, evidence review id: {evidence_review_id}, original search strat id: {original_searchstrat_id}')
                self.logger.error(f'The number of search result articles does not match the number of search result articles in the vector search results: {e}')
                
                raise e 
            except Exception as e: 
                self.logger.error(f'Error retrieving search result articles for search strategy id: {searchstrat_id} and corresponding evidence review id: {evidence_review_id}: {e}')
                raise e 
        
            return rrf_sim_df
        
        else: 
            self.logger.warning(f'No vector search results found for search strategy id: {searchstrat_id} and corresponding evidence review id: {evidence_review_id}.')
            #check evaluation set fo wheter current evidence review id is in there 
            eval_set_df = self.retrieve_evaluation_set(evidence_review_id)
            if evidence_review_id not in eval_set_df['evidence_review_id'].unique(): 
                self.logger.warning(f'Current evidence review id: {evidence_review_id} is not in the evaluation set. This is likely due to no new articles included for current evidence review id. Verify this is intended.')
                return pd.DataFrame()
            else: 
                self.logger.error(f'Evaluation set found for search strategy id: {original_searchstrat_id} and corresponding evidence review id: {evidence_review_id}.')
                raise Exception('Unexpected error occured. ')

    
    def retrieve_evaluation_set(self, evidence_review_id : str): 
        '''
        Retrieves the evaluation set for a given evidence review id 
        '''
        if evidence_review_id == 'overall': 
            query = f"""
                SELECT * FROM groundtruth_eval_set_2017new
            """
        else: 
            query = f"""
                SELECT * FROM groundtruth_eval_set_2017new WHERE evidence_review_id = '{evidence_review_id}'
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
    
    def retrieve_query_evidencereview_topics(self, evidence_review_id): 
        if evidence_review_id == 'overall': 
            query = f"""
                SELECT * FROM query_evidencereview_topic_view
            """
        else: 
            query = f"""
            SELECT * FROM query_evidencereview_topic_view WHERE evidence_review_id = {evidence_review_id}
            """
        with self.engine.begin() as conn: 
            query_evidencereview_topic_df = pd.read_sql(query, conn)
        return query_evidencereview_topic_df

    
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
                WHERE search_types.name = '{search_type}'
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
                              
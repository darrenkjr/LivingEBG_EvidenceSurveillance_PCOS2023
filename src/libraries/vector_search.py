from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import numpy as np 
import pandas as pd 
import torch 
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy import text
from dotenv import load_dotenv
import os 
from logging_config import LoggerConfig
from sql_procedures import sql_procedures
load_dotenv()

class vector_search_implementation(): 

    def __init__(self, model_name = 'allenai/specter2_base', logger = None, query_goldset_flag = False, df = None): 

        self.model = AutoAdapterModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.logger = logger
        self.query_goldset_flag = query_goldset_flag
        self.input_df = df
        goldset_path = Path(__file__).parent.parent / 'dataset' / 'combined_goldset.parquet'
        #check if we're in WSL environment 
        self._database_check()
        #check for cuda 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)

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
                conn.execute(text("SELECT COUNT(*) FROM ground_truth"))
                result = conn.fetchone()
                if result[0] == 0: 
                    self.logger.error(f'Ground truth table does not exist')
                    raise Exception(f'Ground truth table does not exist')
                else: 
                    self.logger.info(f'Ground truth table exists, with {result[0]} rows')
        except Exception as e: 
            self.logger.error(f'Error checking existence of ground truth table: {e}')
            raise e 
        
        self.logger.info(f'Checking existence of search result article table')
        try: 
            with self.engine.connect() as conn: 
                conn.execute(text("SELECT COUNT(*) FROM search_result_articles"))
                result = conn.fetchone()
                if result[0] == 0: 
                    self.logger.error(f'Search result table does not exist')
                    raise Exception(f'Search result table does not exist')
                else: 
                    self.logger.info(f'Search result table exists, with {result[0]} rows')
        except Exception as e: 
            self.logger.error(f'Error checking existence of search result article tables: {e}')
            raise e 



    def generate_vector_db(self): 
        '''
        Generates a dataframe with vector embeddings for a given title and abstract pair.
        '''
        #generate embeddings 
        embeddings = self.generate_embeddings_batch()

        #save embeddings 
        self.save_embeddings(embeddings)

    
    def generate_embeddings_batch(self): 
        '''
        Generates vector embeddings for a given title and abstract pair.
        
        '''
        # combine title and abstract, and keep the id 
        if self.query_goldset_flag: 
            #query articles 
            _df = self.input_df[['included_article_id', 'title', 'abstract']].copy()
        else: 
            #candidate articles 
            _df = self.input_df[[f'{self.eval_id_col}', 'title', 'abstract']].copy()
        
        #combine title and abstract 
        _df['text'] = _df['title'] + self.tokenizer.sep_token + _df['abstract']
        
        #tokenize 
        encoded_input = self.tokenizer(_df['text'], 
                                       return_tensors='pt', 
                                       padding=True, 
                                       truncation=True, 
                                       max_length = 512)
        
        output = self.model(**encoded_input)
        embeddings = output.last_hidden_state[:,0,:].detach().numpy()

        #save embeddings 
        _df['embeddings'] = embeddings
        _df.to_parquet(save_path)
        filename = 'query_goldset' if self.query_goldset_flag else 'query'
        save_path = Path(__file__).parent.parent / 'dataset' / 'embeddings' / f'{filename}_embeddings.parquet'

        return embeddings 
    
    def generate_goldset_embeddings(self): 
        sql_procedures_cls = sql_procedures(logger = self.logger, engine = self.engine)
        goldset_df = sql_procedures_cls.create_goldset_view()
        sql_procedures_cls.setup_embeddings_table(materialized_view_name = 'goldset_view', embedding_table_name = 'goldset_embeddings', linking_id = 'ground_truth_article_id', id_dtype = 'INT')
        
        
        goldset_df['text'] = goldset_df['title'] + self.tokenizer.sep_token + goldset_df['abstract']
        encoded_input = self.tokenizer(goldset_df['text'], 
                                       return_tensors='pt', 
                                       padding=True, 
                                       truncation=True, 
                                       max_length = 512)
        output = self.model(**encoded_input)
        embeddings = output.last_hidden_state[:,0,:].detach().numpy()
        goldset_df['embeddings'] = embeddings
        #add to sql database 
        sql_procedures_cls.add_embeddings_to_table(input_df = goldset_df, embedding_table_name = 'goldset_embeddings', linking_id = 'ground_truth_article_id')

    def generate_searchspace_embeddings(self): 
        sql_procedures_cls = sql_procedures(logger = self.logger, engine = self.engine)

        #get search strategies - do ovearrching first
        overarchingsearchstrat_df = sql_procedures_cls.create_overarching_searchstrat_view()
        #search strategy id is linked to search result article id 
        search_strat_id = overarchingsearchstrat_df['search_strategy_id'].tolist()
        #loop through search strats, gte search result articles, genreate embeddingsa nd uspert 
        for search_strat_id in search_strat_id: 





#genreate embeddings for title and abstracts 
if __name__ == '__main__': 
    #testing small scale 
    logger = LoggerConfig.setup_logger(logger_name = 'vector_search')
    vector_search_cls = vector_search_implementation(logger = logger)
    #generate gold set embeddings 
    #check length of goldset df 
    print(len(vector_search_cls.goldset_df))
    #check that goldset df title and abstracta rte not empty 
    print(vector_search_cls.goldset_df[['title', 'abstract']].isnull().sum())
    # vector_search_cls.generate_goldset_embeddings()


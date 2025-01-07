from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import numpy as np 
import pandas as pd 
import torch 
from pathlib import Path



class vector_search_implementation(): 

    def __init__(self, model_name = 'allenai/specter2', logger = None, query_goldset_flag = False, df = None): 

        self.model = AutoAdapterModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.logger = logger
        self.query_goldset_flag = query_goldset_flag
        self.input_df = df

        #check for cuda 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)

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
    




#genreate embeddings for title and abstracts 

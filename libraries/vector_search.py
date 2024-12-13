
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
import numpy as np 
import pandas as pd 



class vector_search_implementation(): 

    def __init__(self, model_name = 'allenai/specter2'): 

        self.model = AutoAdapterModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model.load_adapter("allenai/specter2", source="hf", load_as="proximity", set_active=True)

    def generate_embeddings(self, title, abstract): 
        '''
        Generates vector embeddings for a given title and abstract pair. Should work for both query and candidate articles 

        Args: 
            title (str): Title of the article
            abstract (str): Abstract of the article

        Returns: 
            embeddings (list): List of floats representing the vector embeddings of the title and abstract
        '''

        #concatenate title and abstract
        text = title + self.tokenizer.sep_token + abstract
        #tokenize
        encoded_input = self.tokenizer(text, 
                                       return_tensors='pt', 
                                       padding=True, 
                                       truncation=True, 
                                       return_token_type_ids=False,
                                         max_length = 512)
        output = self.model(**encoded_input)
        embeddings = output.last_hidden_state[:,0,:].detach().numpy()

        return embeddings 
    
    


#genreate embeddings for title and abstracts 

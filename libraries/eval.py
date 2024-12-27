from pathlib import Path
import pandas as pd 
import rispy
import pyarrow
from dataclasses import dataclass, asdict

#read in ground 
@dataclass 
class eval_metrics: 
    nnr: int
    recall: float
    precision: float
    f1_score: float
    f2_score: float
    f3_score: float

class search_evaluation: 

    
    database_eval_id_mappings = {
        'oa': 'retrieved_oa_id',
        'emb': 'retrieved_embase_id',
        'pubmed': 'retrieved_pubmed_id'
    }

    search_results_id_mappings = {
        'oa': 'id',
        'emb': 'accession_number',
        'pubmed': 'pmid'
    } 


    def __init__(self, database: str, search_type: str, vector_search: bool = False, logger = None, strategy_type = None): 
        self.database = database
        self.search_type = search_type
        self.vector_search_flag = vector_search
        self.eval_id_col = self.database_eval_id_mappings[self.database]
        self.index_id_col = 'included_article_id'
        self.groundtruth_df = self._load_groundtruth()
        self.goldset_df = self._load_gold_set()
        self.logger = logger
        self.strategy_type = strategy_type



    def _load_groundtruth(self): 
        self.groundtruth_path = Path(__file__).parent.parent / 'dataset' / 'fullgroundtruth_valid_apimerge_df.parquet'
        return pd.read_parquet(self.groundtruth_path)
    
    def _load_gold_set(self): 
        self.gold_set_path = Path(__file__).parent.parent / 'dataset' / 'combined_goldset.parquet'
        return pd.read_parquet(self.gold_set_path)
    
    def _load_search_results(self): 

        self.logger.info('Loading search results...')

        #boo
        database_search_results_path_mappings = {
            'oa': (Path(__file__).parent.parent / 'search_results' / 'openalex' / self.search_type / self.strategy_type 
                if self.strategy_type else 
                Path(__file__).parent.parent / 'search_results' / 'openalex' / self.search_type),
            'emb': Path(__file__).parent.parent / 'search_results' / 'embase' / self.search_type,
            'pubmed': Path(__file__).parent.parent / 'search_results' / 'pubmed' / self.search_type,
            'ovid_medline': Path(__file__).parent.parent / 'search_results' / 'ovid_medline' / self.search_type,
        }

        self.search_results_path = database_search_results_path_mappings[self.database]
        self.results_df = pd.DataFrame()

        #check if more than 1 file 
        for file in self.search_results_path.iterdir():
            if file.suffix == '.parquet': 
                df = pd.read_parquet(file)
                self.results_df = pd.concat([self.results_df, df], ignore_index=True)
            if file.suffix == '.ris': 
                try: 
                    with open(file, 'r', encoding='utf-8') as f: 
                        df = rispy.load(f, skip_unknown_tags = True)
                except ValueError as e: 
                    self.logger.info(f'Error loading {file}: {e}')
                    continue
                self.results_df = pd.concat([self.results_df, df], ignore_index=True)

        self.result_id_col = self.search_results_id_mappings[self.database]

        #save cosolidated rsults 
        self.logger.info('Saving consolidated results...')
        if sum(1 for _ in self.search_results_path.iterdir()) > 1 : 
            self.results_df.to_parquet(self.search_results_path / f'{self.database}_{self.search_type}_consolidated_results.parquet')

        
    def process_search_results(self): 
        self._load_search_results()

        if self.database == 'oa': 
            self.results_df[self.result_id_col] = self.results_df[self.result_id_col].str.replace('https://openalex.org/', '')
            #convert to lowercase 
        self.results_df[self.result_id_col] = self.results_df[self.result_id_col].str.lower()
        #drop duplicates 
        self.logger.info(f'Dropping duplicates from {self.database} search results...')
        self.logger.info(f'Number of duplicates: {self.results_df[self.result_id_col].duplicated().sum()}')
        self.results_df.drop_duplicates(subset = self.result_id_col, inplace=True)

        self.results_df.rename(columns = {self.result_id_col: f'retrieved_{self.database}_id'}, inplace=True)

        #check against gold set 
        matched_goldset_ids = set(self.results_df[self.eval_id_col]).intersection(set(self.goldset_df[self.eval_id_col]))
        self.match_goldset_results_df = self.goldset_df[self.goldset_df[self.eval_id_col].isin(matched_goldset_ids)]
        missed_goldset_ids = set(self.goldset_df[self.eval_id_col]).difference(set(self.results_df[self.eval_id_col]))
        self.missed_goldset_results_df = self.goldset_df[self.goldset_df[self.eval_id_col].isin(missed_goldset_ids)]

        # check against ground truth
        
        matched_ids = set(self.results_df[self.eval_id_col]).intersection(set(self.groundtruth_df[self.eval_id_col]))
        self.match_results_df = self.groundtruth_df[self.groundtruth_df[self.eval_id_col].isin(matched_ids)]
        missed_ids = set(self.groundtruth_df[self.eval_id_col]).difference(set(self.results_df[self.eval_id_col]))
        self.missed_results_df = self.groundtruth_df[self.groundtruth_df[self.eval_id_col].isin(missed_ids)]
        self.calc_eval_metrics()
        


    def calc_eval_metrics(self) -> eval_metrics:  
        #number needed to read - need to check if this is a parquet file or a RIS file 
        self.logger.info('Calculating evaluation metrics...')
        nnr = len(self.results_df) 
        recall = len(self.match_results_df) / len(self.groundtruth_df)
        precision = len(self.match_results_df) / len(self.results_df)
        f1 = self._calc_fscore(precision, recall, 1)
        f2 = self._calc_fscore(precision, recall, 2)
        f3 = self._calc_fscore(precision, recall, 3)

        self.logger.info(f'Results for {self.database} search type {self.search_type} with vector search {self.vector_search_flag}:')
        self.logger.info(f'Number needed to read: {nnr}')
        self.logger.info(f'Recall: {recall}')
        self.logger.info(f'Precision: {precision}')
        self.logger.info(f'F1 score: {f1}')
        self.logger.info(f'F2 score: {f2}')
        self.logger.info(f'F3 score: {f3}')

        return eval_metrics(nnr, recall, precision, f1, f2, f3)
    
    @staticmethod 
    def _calc_fscore(precision: float, recall: float, beta: float = 1) -> float: 
        return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

    
    def save_eval_results(self, metrics: eval_metrics): 

        self.logger.info('Saving evaluation results...')
        save_eval_path = Path(__file__).parent.parent / 'evaluation_results' / self.database / self.search_type
        save_data_path = Path(__file__).parent.parent / 'evaluation_results' / self.database / self.search_type / 'matched_missed_results'
        if not save_eval_path.exists(): 
            save_eval_path.mkdir(parents=True, exist_ok=True)
        if not save_data_path.exists(): 
            save_data_path.mkdir(parents=True, exist_ok=True)
        
        results_dict = {
            'database': self.database,
            'search_type': self.search_type,
            'vector_search': self.vector_search_flag,
            **asdict(metrics)
        }

        filename = f'{self.search_type}_{self.database}_{self.vector_search_flag}_eval.parquet'
        evalmetrics_df = pd.DataFrame.from_dict([results_dict])
        evalmetrics_df.to_parquet(save_eval_path / filename) 

        self.match_results_df.to_parquet(save_data_path / f'matched_results_{self.database}_{self.search_type}_{self.vector_search_flag}.parquet')
        self.missed_results_df.to_parquet(save_data_path / f'missed_results_{self.database}_{self.search_type}_{self.vector_search_flag}.parquet')
        self.match_goldset_results_df.to_parquet(save_data_path / f'matched_goldset_results_{self.database}_{self.search_type}_{self.vector_search_flag}.parquet')
        self.missed_goldset_results_df.to_parquet(save_data_path / f'missed_goldset_results_{self.database}_{self.search_type}_{self.vector_search_flag}.parquet')

    def run_eval_pipeline(self): 
        self.process_search_results()
        metrics = self.calc_eval_metrics()
        self.save_eval_results(metrics)


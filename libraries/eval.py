from pathlib import Path
import pandas as pd 
import rispy
import pyarrow
from dataclasses import dataclass, asdict
from datetime import datetime

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


    def __init__(self, database: str, search_type: str, vector_search: bool = False, logger = None, strategy_type = 'boolkw_search'): 
        self.database = database
        self.search_type = search_type
        self.vector_search_flag = vector_search
        self.eval_id_col = self.database_eval_id_mappings[self.database]
        self.index_id_col = 'included_article_id'
        self.groundtruth_df = self._load_groundtruth()
        self.goldset_df = self._load_gold_set()
        self.logger = logger
        #default strategy type is boolkw_search 
        self.strategy_type = strategy_type

        #set up search results path 
        database_search_results_path_mappings = {
            'oa': (Path(__file__).parent.parent / 'search_results' / 'openalex' / self.search_type / self.strategy_type 
                if self.strategy_type else 
                Path(__file__).parent.parent / 'search_results' / 'openalex' / self.search_type),
            'emb': Path(__file__).parent.parent / 'search_results' / 'embase' / self.search_type,
            'pubmed': Path(__file__).parent.parent / 'search_results' / 'pubmed' / self.search_type,
            'ovid_medline': Path(__file__).parent.parent / 'search_results' / 'ovid_medline' / self.search_type,
        }

        self.search_results_path = database_search_results_path_mappings[self.database]
        self.consolidated_results_path = self.search_results_path / f'{self.database}_{self.search_type}_consolidated_{self.strategy_type}_results.parquet'
        self.save_eval_path = Path(__file__).parent.parent / 'evaluation_results' / self.database / self.search_type / self.strategy_type if self.strategy_type \
        else Path(__file__).parent.parent / 'evaluation_results' / self.database / self.search_type

        if not self.save_eval_path.exists(): 
            self.save_eval_path.mkdir(parents=True, exist_ok=True)

    def _load_groundtruth(self): 
        self.groundtruth_path = Path(__file__).parent.parent / 'dataset' / 'fullgroundtruth_valid_apimerge_df.parquet'
        return pd.read_parquet(self.groundtruth_path)
    
    def _load_gold_set(self): 
        self.gold_set_path = Path(__file__).parent.parent / 'dataset' / 'combined_goldset.parquet'
        return pd.read_parquet(self.gold_set_path)
    
    def _load_search_results(self): 

        self.logger.info('Loading search results...')

        #check if folder has files in it 
        if not self.search_results_path.iterdir(): 
            self.logger.error('No search results found in folder, check paths') 
            raise Exception('No search results found in folder, check paths')

        self.results_df = pd.DataFrame()

        if self.database == 'oa':  
            if self.search_type == 'overarching' and self.strategy_type == 'oatopic_search':
                self.logger.info(f'Consolidating search results for Database: {self.database}, Search Type: {self.search_type}, Strategy Type: {self.strategy_type}...')
                for file in self.search_results_path.iterdir(): 
                #consolidate results if we are doing an ovarcing openalex topic search 
                    if file.suffix == '.parquet': 
                        df = pd.read_parquet(file)
                        self.results_df = pd.concat([self.results_df, df], ignore_index=True)
                    else: 
                        raise ValueError(f'File {file} is not a parquet file')
                #save consolidated topic search results 
                self.results_df.to_parquet(self.consolidated_results_path)
                self.logger.info('Consolidated search results saved to: ', self.consolidated_results_path)


            if self.search_type == 'overarching' and self.strategy_type == 'boolkw_search': 
                #extract query from file name 
                self.logger.info(f'Consolidating search results for Database: {self.database}, Search Type: {self.search_type}, Strategy Type: {self.strategy_type}...')
                for file in self.search_results_path.iterdir(): 
                    if file.suffix == '.parquet': 
                        query = file.name.split('_')[2].replace('.parquet', '')
                        df = pd.read_parquet(file)
                        df['query'] = query
                        self.results_df = pd.concat([self.results_df, df], ignore_index=True)
                
                self.results_df.to_parquet(self.consolidated_results_path)
                self.logger.info('Consolidated search results saved to: ', self.consolidated_results_path)
                        
        if self.database == 'emb' or self.database == 'ovid_medline': 
            #ris files are downloaded as batches from ovid medline so need to consolidate results 
            if self.search_type == 'overarching': 
                self.logger.info(f'Consolidating search results for Database: {self.database}, Search Type: {self.search_type}, Strategy Type: {self.strategy_type}...')
                for file in self.search_results_path.iterdir(): 
                    if file.suffix == '.ris': 
                        try: 
                            with open(file, 'r', encoding='utf-8') as f: 
                                df = rispy.load(f, skip_unknown_tags = True)
                                self.results_df = pd.concat([self.results_df, df], ignore_index=True)
                        except Exception as e: 
                            self.logger.error(f'Error loading {file}: {e}', exc_info=True)
                            continue
                self.results_df.to_parquet(self.consolidated_results_path)
                self.logger.info('Consolidated search results saved to: ', self.consolidated_results_path)

        self.result_id_col = self.search_results_id_mappings[self.database]
        #save cosolidated rsults 
        
        
    def process_search_results(self): 
        self._load_search_results()
        self.results_df[self.result_id_col] = self.results_df[self.result_id_col].str.lower()
        # Evaluate matches for each query group
        evalmetrics_df = pd.DataFrame()
        if self.database == 'oa': 
            self.results_df[self.result_id_col] = self.results_df[self.result_id_col].str.replace('https://openalex.org/', '')
            #if we are doing a boolkw search and overarching search, we need to evaluate matches for each query (when testing on goldset)
            if self.strategy_type == 'boolkw_search' and self.search_type == 'overarching': 
                query_list = self.results_df['query'].unique()
                if len(query_list) > 1: 
                    # Process all queries at once using groupby
                    grouped_results = self.results_df.groupby('query')
                    
                    for query, result_df_kwquery_oa in grouped_results:
                        self.logger.info(f'Evaluating matches for Database: {self.database}, Search type: {self.search_type}, Strategy type: {self.strategy_type}, query: {query}')
                        match_results_df, missed_results_df, match_goldset_results_df, missed_goldset_results_df = self._evaluate_matches(result_df_kwquery_oa)
                        
                        # Calculate metrics for goldset and groundtruth
                        metrics_goldset = self.calc_eval_metrics(match_goldset_results_df, self.goldset_df, result_df_kwquery_oa)
                        metrics_groundtruth = self.calc_eval_metrics(match_results_df, self.groundtruth_df, result_df_kwquery_oa)
                        
                        # Create dataframes for both metric types
                        metrics_goldset_df = pd.DataFrame.from_records([asdict(metrics_goldset)])
                        metrics_goldset_df['query'] = query
                        metrics_goldset_df['performance_on'] = 'goldset'
                        metrics_groundtruth_df = pd.DataFrame.from_records([asdict(metrics_groundtruth)])
                        metrics_groundtruth_df['query'] = query
                        metrics_groundtruth_df['performance_on'] = 'groundtruth'
                        evalmetrics_df = pd.concat([evalmetrics_df, metrics_goldset_df, metrics_groundtruth_df], ignore_index=True)

                        #save matched and missed results 
                        match_results_df.to_parquet(self.save_eval_path / f'matched_results_{self.database}_{self.search_type}_{self.strategy_type}_{query}.parquet')
                        missed_results_df.to_parquet(self.save_eval_path / f'missed_results_{self.database}_{self.search_type}_{self.strategy_type}_{query}.parquet')
                        match_goldset_results_df.to_parquet(self.save_eval_path / f'matched_goldset_results_{self.database}_{self.search_type}_{self.strategy_type}_{query}.parquet')
                        missed_goldset_results_df.to_parquet(self.save_eval_path / f'missed_goldset_results_{self.database}_{self.search_type}_{self.strategy_type}_{query}.parquet')

                        #save evaluation metrics 
                    self.save_eval_results(evalmetrics_df)

            else: 

                match_results_df, missed_results_df, match_goldset_results_df, missed_goldset_results_df = self._evaluate_matches(self.results_df)
                metrics_goldset = self.calc_eval_metrics(match_goldset_results_df, self.goldset_df, self.results_df)
                metrics_goldset_df = pd.DataFrame.from_records([asdict(metrics_goldset)])
                metrics_goldset_df['performance_on'] = 'goldset'
                metrics_groundtruth = self.calc_eval_metrics(match_results_df, self.groundtruth_df, self.results_df)
                metrics_groundtruth_df = pd.DataFrame.from_records([asdict(metrics_groundtruth)])
                metrics_groundtruth_df['performance_on'] = 'groundtruth'
                evalmetrics_df = pd.concat([evalmetrics_df, metrics_goldset_df, metrics_groundtruth_df], ignore_index=True)

                #save matched and missed results 
                match_results_df.to_parquet(self.save_eval_path / f'matched_results_{self.database}_{self.search_type}_{self.strategy_type}.parquet')
                missed_results_df.to_parquet(self.save_eval_path / f'missed_results_{self.database}_{self.search_type}_{self.strategy_type}.parquet')
                match_goldset_results_df.to_parquet(self.save_eval_path / f'matched_goldset_results_{self.database}_{self.search_type}_{self.strategy_type}.parquet')
                missed_goldset_results_df.to_parquet(self.save_eval_path / f'missed_goldset_results_{self.database}_{self.search_type}_{self.strategy_type}.parquet')

                self.save_eval_results(evalmetrics_df)

        #drop duplicates 
    def _evaluate_matches(self, results_df: pd.DataFrame): 
        self.logger.info(f'Dropping duplicates from {self.database} search results...')
        self.logger.info(f'Number of duplicates: {results_df[self.result_id_col].duplicated().sum()}')
        results_df.drop_duplicates(subset = self.result_id_col, inplace=True)
        results_df.rename(columns = {self.result_id_col: f'retrieved_{self.database}_id'}, inplace=True)

        #check against gold set 
        matched_goldset_ids = set(results_df[self.eval_id_col]).intersection(set(self.goldset_df[self.eval_id_col]))
        match_goldset_results_df = self.goldset_df[self.goldset_df[self.eval_id_col].isin(matched_goldset_ids)]
        missed_goldset_ids = set(self.goldset_df[self.eval_id_col]).difference(set(results_df[self.eval_id_col]))
        missed_goldset_results_df = self.goldset_df[self.goldset_df[self.eval_id_col].isin(missed_goldset_ids)]

        # check against ground truth
        matched_ids = set(results_df[self.eval_id_col]).intersection(set(self.groundtruth_df[self.eval_id_col]))
        match_results_df = self.groundtruth_df[self.groundtruth_df[self.eval_id_col].isin(matched_ids)]
        missed_ids = set(self.groundtruth_df[self.eval_id_col]).difference(set(results_df[self.eval_id_col]))
        missed_results_df = self.groundtruth_df[self.groundtruth_df[self.eval_id_col].isin(missed_ids)]
        return match_results_df, missed_results_df, match_goldset_results_df, missed_goldset_results_df
            
    def calc_eval_metrics(self, match_df: pd.DataFrame, comparison_df: pd.DataFrame, raw_results_df : pd.DataFrame) -> eval_metrics:   
        self.logger.info('Calculating evaluation metrics...')
        nnr = len(raw_results_df) 
        recall = len(match_df) / len(comparison_df)
        precision = len(match_df) / len(raw_results_df)
        f1 = self._calc_fscore(precision, recall, 1)
        f2 = self._calc_fscore(precision, recall, 2)
        f3 = self._calc_fscore(precision, recall, 3)

        self.logger.info(f'Results for {self.database}, search type {self.search_type}, strategy type {self.strategy_type}, vector search: {self.vector_search_flag}:')
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

    def save_eval_results(self, metrics_df: pd.DataFrame): 

        self.logger.info('Saving evaluation results...')

        try: 
            metadata_dict = {
                'database': self.database,
                'search_type': self.search_type,
                'search_strategy': self.strategy_type,
                'vector_search': self.vector_search_flag,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
            }

            metadata_df = pd.DataFrame.from_dict([metadata_dict])
            evalmetrics_df = pd.concat([metadata_df, metrics_df], axis=1)

            filename = f'{self.search_type}_{self.database}_{self.strategy_type}_eval.parquet'
            evalmetrics_df.to_parquet(self.save_eval_path / filename) 
            self.logger.info(f'Evaluation results saved to: {self.save_eval_path / filename}')

        except Exception as e: 
            self.logger.error(f'Error saving evaluation results: {e}')
            raise


    def run_eval_pipeline(self): 
        self.process_search_results()

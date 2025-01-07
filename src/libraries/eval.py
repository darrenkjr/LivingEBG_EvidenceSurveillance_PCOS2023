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
        'embase': 'retrieved_embase_id',
        'pubmed': 'retrieved_pubmed_id', 
        'medline' : 'retrieved_pubmed_id',
    }

    search_results_id_mappings = {
        'oa': 'id',
        'embase': 'accession_number',
        'pubmed': 'pmid',
        'medline' : 'id'
    } 

    question_id_mappings = {
        '1.5.1, 1.5.2' : '1.5',
        '1.4.1, 1.4.2' : '1.4',
        '1.10.' : '1.10', 
        '2.1.1 / 2.1.2' : '2.1',
        '4.2 / 4.3' : '4.2.4.3.combined',
        '4.10.' : '4.10', 
    }

    def __init__(self, database: str, search_type: str, vector_search: bool = False, logger = None, strategy_type = 'boolkw_search'): 
        self.logger = logger
        self.database = database
        self.search_type = search_type
        self.vector_search_flag = vector_search

        #columns corresponding with id columns from database 
        self.eval_id_col = self.database_eval_id_mappings[self.database]
        self.result_id_col = self.search_results_id_mappings[self.database]
        self.index_id_col = 'included_article_id'
        self.groundtruth_df = self._load_groundtruth()
        self.goldset_df = self._load_gold_set()
        
        #default strategy type is boolkw_search 
        self.strategy_type = strategy_type

        #set up search results path 
        database_search_results_path_mappings = {
            'oa': (Path(__file__).parent.parent / 'search_results' / 'openalex' / self.search_type / self.strategy_type 
                if self.strategy_type else 
                Path(__file__).parent.parent / 'search_results' / 'openalex' / self.search_type),
            'embase': Path(__file__).parent.parent / 'search_results' / 'embase' / self.search_type,
            'pubmed': Path(__file__).parent.parent / 'search_results' / 'pubmed' / self.search_type,
            'medline': Path(__file__).parent.parent / 'search_results' / 'medline' / self.search_type,
        }

        self.search_results_path = database_search_results_path_mappings[self.database]
        self.consolidated_results_path = self.search_results_path / f'{self.database}_{self.search_type}_consolidated_{self.strategy_type}_results.parquet'
        self.save_eval_path = Path(__file__).parent.parent / 'evaluation_results' / self.database / self.search_type / self.strategy_type if self.strategy_type \
        else Path(__file__).parent.parent / 'evaluation_results' / self.database / self.search_type

        self.save_eval_missed_results_path = Path(__file__).parent.parent / 'evaluation_results' / self.database / self.search_type / self.strategy_type / 'matched_missed_results' if self.strategy_type \
        else Path(__file__).parent.parent / 'evaluation_results' / self.database / self.search_type / 'matched_missed_results'

        if not self.save_eval_path.exists(): 
            self.save_eval_path.mkdir(parents=True, exist_ok=True)

        if not self.save_eval_missed_results_path.exists(): 
            self.save_eval_missed_results_path.mkdir(parents=True, exist_ok=True)

    def _load_groundtruth(self): 
        self.groundtruth_path = Path(__file__).parent.parent / 'dataset' / 'fullgroundtruth_valid_apimerge_df.parquet'
        _df = pd.read_parquet(self.groundtruth_path)
        #lower case id cols 
        mask = _df[self.eval_id_col].notna() & (_df[self.eval_id_col] != '')
        _df.loc[mask, self.eval_id_col] = (_df.loc[mask, self.eval_id_col]
                                        .str.lower()
                                        .str.strip())
        
        self.logger.info(f"Processed {mask.sum()} non-empty IDs out of {len(_df)} total entries")
        return _df
    
    def _load_gold_set(self): 
        self.gold_set_path = Path(__file__).parent.parent / 'dataset' / 'combined_goldset.parquet'
        _df = pd.read_parquet(self.gold_set_path)
        mask = _df[self.eval_id_col].notna() & (_df[self.eval_id_col] != '')
        _df.loc[mask, self.eval_id_col] = (_df.loc[mask, self.eval_id_col]
                                        .str.lower()
                                        .str.strip())
        
        self.logger.info(f"Processed {mask.sum()} non-empty IDs out of {len(_df)} total entries")
        return _df
    
    def _load_overarching_search_results(self): 

        self.logger.info('Loading search results...')

        #check if folder has files in it 
        if not self.search_results_path.iterdir(): 
            self.logger.error('No search results found in folder, check paths') 
            raise Exception('No search results found in folder, check paths')
        
        #check if consolidated results path exists no need to do load again 
        if self.consolidated_results_path.exists(): 
            self.logger.info(f'Consolidated results already exist, loading from {str(self.consolidated_results_path)}')
            self.results_df = pd.read_parquet(self.consolidated_results_path)

        else: 
            self.results_df = pd.DataFrame()

            if self.database == 'oa' or self.database == 'pubmed':  

                if self.search_type == 'overarching' and self.strategy_type == 'boolkw_search': 
                    #extract query from file name 
                    self.logger.info(f'Consolidating search results for Database: {self.database}, Search Type: {self.search_type}, Strategy Type: {self.strategy_type}...')
                    for file in self.search_results_path.iterdir(): 
                        if file.suffix == '.parquet': 
                            query = file.name.split('_')[2].replace('.parquet', '')
                            df = pd.read_parquet(file)
                            df['query'] = query
                            self.results_df = pd.concat([self.results_df, df], ignore_index=True)
                            
                elif self.search_type == 'overarching' and self.strategy_type == 'oatopic_search':
                    self.logger.info(f'Consolidating search results for Database: {self.database}, Search Type: {self.search_type}, Strategy Type: {self.strategy_type}...')
                    for file in self.search_results_path.iterdir(): 
                    #consolidate results if we are doing an ovarcing openalex topic search 
                        if file.suffix == '.parquet': 
                            df = pd.read_parquet(file)
                            self.results_df = pd.concat([self.results_df, df], ignore_index=True)
                        else: 
                            raise ValueError(f'File {file} is not a parquet file')


            if self.database == 'embase' or self.database == 'medline': 
                #ris files are downloaded as batches from ovid medline so need to consolidate results 
                if self.search_type == 'overarching': 
                    self.logger.info(f'Consolidating search results for Database: {self.database}, Search Type: {self.search_type}, Strategy Type: {self.strategy_type}...')
                    for file in self.search_results_path.iterdir(): 
                        if file.suffix == '.ris': 
                            try: 
                                with open(file, 'r', encoding='utf-8') as f: 
                                    df = pd.DataFrame(rispy.load(f, skip_unknown_tags = True))
                                    self.results_df = pd.concat([self.results_df, df], ignore_index=True)
                            except Exception as e: 
                                self.logger.error(f'Error loading {file}: {e}', exc_info=True)
                                raise
                        

            self.results_df.to_parquet(self.consolidated_results_path)
            self.logger.info(f'Consolidated search results saved to: {str(self.consolidated_results_path)}')
        
        
            #save cosolidated rsults  
    
    def _load_topic_specific_search_results(self): 

        self.logger.info('Loading search results...')

        #check if cosonolidated results path arlready exists 
        
        if self.consolidated_results_path.exists(): 
            self.results_df = pd.read_parquet(self.consolidated_results_path)
            self.logger.info(f'Consolidated search results already exists, loading from {str(self.consolidated_results_path)}')
            return


        else: 
            self.groundtruth_df['question_id'] = self.groundtruth_df['question_id'].map(lambda x: self.question_id_mappings.get(x, x))
            valid_question_id = list(self.groundtruth_df['question_id'].unique())
            self.results_df = pd.DataFrame()

            for file in self.search_results_path.iterdir(): 
                if file.suffix == '.ris': 
                    # Do all string operations in one line
                    current_question_id = file.stem.rsplit('_', 1)[0].replace('gdg', '').replace('_', '.')
                    #read current file 
                    with open(file, 'r', encoding='utf-8') as f: 
                            df = pd.DataFrame(rispy.load(f, skip_unknown_tags = True))
                            df['question_id'] = current_question_id
                            self.results_df = pd.concat([self.results_df, df], ignore_index=True)
            
            self.results_df.to_parquet(self.consolidated_results_path)
            self.logger.info(f'Consolidated search results saved to: {str(self.consolidated_results_path)}')
            
            
            try:    
                assert set(valid_question_id).issubset(set(self.results_df['question_id'].unique())), \
                    f"Found invalid question IDs: {set(valid_question_id) - set(self.results_df['question_id'].unique())}"
                    
            except AssertionError: 
                self.logger.error('Found invalid question ids between results and input ground truth dataframes. Might want to recheck results from search retrieval.')
                raise


    def process_search_results(self): 
        '''
        WORK IN PROGRESS
        
        
        '''
        self.results_df[self.result_id_col] = self.results_df[self.result_id_col].str.lower()
        if self.database == 'oa': 
            #remove leading 'https://openalex.org/' from result_id_col
            self.results_df[self.result_id_col] = self.results_df[self.result_id_col].str.replace('https://openalex.org/', '')

        evalmetrics_df = pd.DataFrame()
        match_results_df, missed_results_df = self._evaluate_matches(self.groundtruth_df, self.results_df)

        #overall metrics 
        self.question_id = 'overall'
        metrics_groundtruth = self.calc_eval_metrics(match_results_df, self.groundtruth_df, self.results_df)
        metrics_groundtruth_df = pd.DataFrame.from_records([asdict(metrics_groundtruth)])
        metrics_groundtruth_df['performance_on'] = 'groundtruth'
        metrics_groundtruth_df['question_id'] = self.question_id
        evalmetrics_df = pd.concat([metrics_groundtruth_df], ignore_index=True)
        #save matched and missed results 
        self._save_eval_results(evalmetrics_df)
        self._save_match_missed_results(match_results_df, missed_results_df)

        if self.search_type == 'topic_specific': 
            grouped_evalmetrics_df = pd.DataFrame()
            for question_id, grouped_df in self.results_df.groupby('question_id'): 
                self.question_id = question_id
                grouped_groundtruth_df = self.groundtruth_df[self.groundtruth_df['question_id'] == self.question_id]
                match_results_df, missed_results_df = self._evaluate_matches(grouped_groundtruth_df, grouped_df)
                metrics_groundtruth = self.calc_eval_metrics(match_results_df, grouped_groundtruth_df, grouped_df)
                metrics_groundtruth_df = pd.DataFrame.from_records([asdict(metrics_groundtruth)])
                metrics_groundtruth_df['performance_on'] = 'groundtruth'
                metrics_groundtruth_df['question_id'] = self.question_id
                grouped_evalmetrics_df = pd.concat([grouped_evalmetrics_df, metrics_groundtruth_df], ignore_index=True)
            
        
            self._save_eval_results(grouped_evalmetrics_df)


    def _evaluate_matches(self, comparison_df: pd.DataFrame, results_df: pd.DataFrame): 
        '''
        Evaluate matches between comparison_df and results_df

        Args: 
            comparison_df: dataframe with ground truth data
            results_df: dataframe with search results

        Returns: 
            matched_results_df: dataframe with matched results
            missed_results_df: dataframe with missed results
        '''
        # Clean IDs for matching
        results_df = results_df.copy()
        comparison_df = comparison_df.copy()
        
        results_df['clean_id'] = results_df[self.result_id_col].astype(str).str.lower().str.strip()
        comparison_df['clean_id'] = comparison_df[self.eval_id_col].astype(str).str.lower().str.strip()
        
        # Find matches using merge
        matched_results_df = pd.merge(
            comparison_df,
            results_df,
            left_on='clean_id',
            right_on='clean_id',
            how='inner'
        )
        
        # Find missed using merge
        missed_results_df = comparison_df[
            ~comparison_df['clean_id'].isin(matched_results_df['clean_id'])
        ]
        
        # Log results
        self.logger.info(f"Total matches found: {len(matched_results_df)}")
        self.logger.info(f"Unique matched IDs: {matched_results_df['clean_id'].nunique()}")
        self.logger.info(f"Missed entries: {len(missed_results_df)}")
        
        # Clean up temporary column
        matched_results_df = matched_results_df.drop('clean_id', axis=1)
        missed_results_df = missed_results_df.drop('clean_id', axis=1)
        
        return matched_results_df, missed_results_df
            

            
    def calc_eval_metrics(self, match_df: pd.DataFrame, comparison_df: pd.DataFrame, raw_results_df : pd.DataFrame) -> eval_metrics:   
        self.logger.info(f'Calculating evaluation metrics for current {self.question_id}...')
        nnr = len(raw_results_df)

        #check length of match df 
        if len(match_df) == 0: 
            self.logger.warning(f'No matches found for {self.question_id}')
            return eval_metrics(nnr, 0, 0, 0, 0, 0)
        
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
    
    
    def _calc_fscore(self, precision: float, recall: float, beta: float = 1) -> float: 
        try: 
            return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
        except ZeroDivisionError: 
            self.logger.warning(f'Recall is {recall}, Precision is {precision}, returning 0')
            return 0

    def _save_eval_results(self, evalmetrics_df: pd.DataFrame): 

        self.logger.info('Saving evaluation results...')

        try: 
            metadata_dict = {
                'database': self.database,
                'search_type': self.search_type,
                'search_strategy': self.strategy_type,
                'vector_search': self.vector_search_flag,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
            }

            evalmetrics_df = evalmetrics_df.assign(**metadata_dict)
            filename = f'{self.search_type}_{self.database}_{self.strategy_type}_eval.parquet'
            evalmetrics_df.to_parquet(self.save_eval_path / filename) 
            self.logger.info(f'Evaluation results saved to: {self.save_eval_path / filename}')

        except Exception as e: 
            self.logger.error(f'Error saving evaluation results: {e}')
            raise

    def _save_match_missed_results(self, match_results_df: pd.DataFrame, missed_results_df: pd.DataFrame): 
        match_results_df.to_parquet(self.save_eval_path / f'matched_results_{self.database}_{self.search_type}_{self.strategy_type}.parquet')
        missed_results_df.to_parquet(self.save_eval_missed_results_path / f'missed_results_{self.database}_{self.search_type}_{self.strategy_type}.parquet')

    def run_eval_pipeline(self): 
        if self.search_type == 'overarching': 
            self._load_overarching_search_results()
        elif self.search_type == 'topic_specific': 
            self._load_topic_specific_search_results()
        
        self.process_search_results()

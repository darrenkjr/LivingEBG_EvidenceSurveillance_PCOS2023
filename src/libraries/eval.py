from pathlib import Path
import pandas as pd 
import rispy
import pyarrow
from dataclasses import dataclass, asdict
from datetime import datetime
import re

#read in ground 
@dataclass 
class eval_metrics: 
    nnr: int
    n_retrieved: int 
    n_missed : int
    recall: float
    precision: float
    f1_score: float
    f2_score: float
    f3_score: float

@dataclass
class vectorsearch_eval_metrics: 
    nnr_raw: int # nnr before cutoff 
    rank_new_relevant : int #nnr before screening first correct result (not in goldset)
    rank_first_relevant : int #nnr before screening first correct result (all))
    nnr_threshold: int # total nnr after applying similiarity threshold
    n_retrieved: int 
    n_missed: int
    recall: float
    precision: float
    precision_cutoff: float
    f1_score: float
    f2_score: float
    f3_score: float
    recall_at_10 : float 
    recall_at_100: float 
    recall_at_1000: float
    similarity_threshold_cutoff: float
    mrr: float
    mrr_new_relevant: float




class search_evaluation: 

    database_eval_id_mappings = {
        'oa': 'retrieved_oa_id',
        'openalex' : 'retrieved_oa_id',
        'embase': 'retrieved_embase_id',
        'pubmed': 'retrieved_pubmed_id', 
        'medline' : 'retrieved_pubmed_id',
    }

    search_results_id_mappings = {
        'oa': 'id',
        'openalex' : 'id',
        'embase': 'accession_number',
        'pubmed': 'pmid',
        'medline' : 'id'
    } 

    question_id_mappings = {
        '4.2.4.3.combined' : '4.2/4.3', 
        '1.5' : '1.5.1/1.5.2', 
        '1.4' : '1.4.1/1.4.2', 
        '2.1' : '2.1.1/2.1.2', 
        '1.9.1.embase' : '1.9.1', 
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
        self.eval_groundtruth_df = self._load_eval_groundtruth()
        self.goldset_df = self._load_gold_set()

        #default year rage limit to parse 
        self.start_year_cutoff = 1990
        self.end_year_cutoff = 2022
        
        #default strategy type is boolkw_search 
        self.strategy_type = strategy_type

        #set up search results path 
        database_search_results_path_mappings = {
            'oa': (Path(__file__).parent.parent / 'search_results' / 'openalex' / self.search_type / self.strategy_type 
                if self.strategy_type else 
                Path(__file__).parent.parent / 'search_results' / 'openalex' / self.search_type),
            'openalex' : (Path(__file__).parent.parent / 'search_results' / 'openalex' / self.search_type / self.strategy_type 
                if self.strategy_type else 
                Path(__file__).parent.parent / 'search_results' / 'openalex' / self.search_type), 
            'embase': Path(__file__).parent.parent / 'search_results' / 'embase' / self.search_type,
            'pubmed': Path(__file__).parent.parent / 'search_results' / 'pubmed' / self.search_type,
            'medline': Path(__file__).parent.parent / 'search_results' / 'medline' / self.search_type,
        }

        self.search_results_path = database_search_results_path_mappings[self.database]

        self._consolidated_results_dir = Path(__file__).parent.parent / 'consolidated_results' 
        self._consolidated_results_dir.mkdir(parents=True, exist_ok=True)
        self.consolidated_results_path = self._consolidated_results_dir / f'{self.database}_{self.search_type}_consolidated_{self.strategy_type}_results.parquet'
        self.save_eval_path = Path(__file__).parent.parent / 'evaluation_results' / self.database / self.search_type / self.strategy_type if self.strategy_type \
        else Path(__file__).parent.parent / 'evaluation_results' / self.database / self.search_type

        self.save_eval_missed_results_path = Path(__file__).parent.parent / 'evaluation_results' / self.database / self.search_type / self.strategy_type  if self.strategy_type \
        else Path(__file__).parent.parent / 'evaluation_results' / self.database / self.search_type 

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
        
        self.logger.info(f"Processed {mask.sum()} non-empty IDs out of {len(_df)} total entries for ground truth articles")
        return _df
    
    def _load_eval_groundtruth(self): 
        self.eval_groundtruth_path = Path(__file__).parent.parent / 'dataset' / 'groundtruth_eval.parquet'
        _df = pd.read_parquet(self.eval_groundtruth_path)
                #lower case id cols 
        mask = _df[self.eval_id_col].notna() & (_df[self.eval_id_col] != '')
        _df.loc[mask, self.eval_id_col] = (_df.loc[mask, self.eval_id_col]
                                        .str.lower()
                                        .str.strip())
        
        self.logger.info(f"Processed {mask.sum()} non-empty database specific IDs out of {len(_df)} total entries for evaluation articles (articles newly added to guideline in 2017)")
        return _df
    
    def _load_gold_set(self): 
        self.gold_set_path = Path(__file__).parent.parent / 'dataset' / 'combined_goldset.parquet'
        _df = pd.read_parquet(self.gold_set_path)
        mask = _df[self.eval_id_col].notna() & (_df[self.eval_id_col] != '')
        _df.loc[mask, self.eval_id_col] = (_df.loc[mask, self.eval_id_col]
                                        .str.lower()
                                        .str.strip())
        
        self.logger.info(f"Processed {mask.sum()} non-empty IDs out of {len(_df)} total entries for gold set articles")
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
            results_df = pd.read_parquet(self.consolidated_results_path)
            return results_df


        else: 
            result_df = pd.DataFrame()

            if self.database == 'oa' or self.database == 'pubmed':  

                if self.search_type == 'overarching' and self.strategy_type == 'boolkw_search': 
                    #extract query from file name 
                    self.logger.info(f'Consolidating search results for Database: {self.database}, Search Type: {self.search_type}, Strategy Type: {self.strategy_type}...')
                    for file in self.search_results_path.iterdir(): 
                        if file.suffix == '.parquet': 
                            df = pd.read_parquet(file)
                            result_df = pd.concat([result_df, df], ignore_index=True)
                            
                elif self.search_type == 'overarching' and self.strategy_type == 'topic_search':
                    self.logger.info(f'Consolidating search results for Database: {self.database}, Search Type: {self.search_type}, Strategy Type: {self.strategy_type}...')
                    for file in self.search_results_path.iterdir(): 
                    #consolidate results if we are doing an ovarcing openalex topic search 
                        if file.suffix == '.parquet': 
                            df = pd.read_parquet(file)
                            result_df = pd.concat([result_df, df], ignore_index=True)
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
                                    result_df = pd.concat([result_df, df], ignore_index=True)
                            except Exception as e: 
                                self.logger.error(f'Error loading {file}: {e}', exc_info=True)
                                raise

                if self.database == 'medline': 
                    result_df.rename(columns = {'Y1' : 'publication_year'}, inplace=True)
                    #ensure it is int 
                result_df['publication_year'] = result_df['publication_year'].astype(str).str.replace('//', '')
                result_df['publication_year'] = result_df['publication_year'].astype(str).str.split(',').str[0]
                result_df['publication_year'] = result_df['publication_year'].astype(int)
                
            self.logger.info(f'Filtering results for year range {self.start_year_cutoff} to {self.end_year_cutoff}')
            _result_df = result_df.query(f'publication_year >= {self.start_year_cutoff} & publication_year <= {self.end_year_cutoff}')
            results_df = _result_df.copy()
            results_df_dedupe = results_df.drop_duplicates(subset = self.result_id_col)
            results_df_dedupe.to_parquet(self.consolidated_results_path)
            self.logger.info(f'Consolidated search results saved to: {str(self.consolidated_results_path)}')
            return results_df_dedupe

    
    def _load_topic_specific_search_results(self): 
        self.logger.info(f'Loading comparison set')
        comparison_set = self.eval_groundtruth_df.copy()

        self.logger.info('Loading search results...')

        #check if cosonolidated results path arlready exists 
        if self.consolidated_results_path.exists(): 
            results_df = pd.read_parquet(self.consolidated_results_path)
            results_df = self._standardize_results(results_df)
        else: 
            self.logger.info(f'Consolidated search results do not exist, loading from {str(self.search_results_path)}')
            results_df = pd.DataFrame()

            for file in self.search_results_path.iterdir(): 
                if file.suffix == '.ris': 

                    #read current file 
                    with open(file, 'r', encoding='utf-8') as f: 
                                                # Do all string operations in one line
                        self.logger.info(f'Processing current file: {file.name}')
                        current_question_id = file.stem.rsplit('_', 1)[0].replace('gdg', '').replace('_', '.')
                        if current_question_id == '.4.9' or current_question_id in self.question_id_mappings: 
                            self.logger.warning(f'Question id requiring mapping detected : {current_question_id}, mapping to {self.question_id_mappings[current_question_id]}')
                            current_question_id = self.question_id_mappings[current_question_id]
                        self.logger.info(f'Processing current detected question id:: {current_question_id}')
                        df = pd.DataFrame(rispy.load(f, skip_unknown_tags = True))
                        df['question_id'] = current_question_id
                        df['origin_file'] = file.name
                        results_df = pd.concat([results_df, df], ignore_index=True)
            
            results_df = self._standardize_results(results_df)
            _question_id_list = list(results_df['question_id'].unique()) 
            _comparison_question_id_list = list(comparison_set['question_id'].unique())
            try: 
                assert set(_comparison_question_id_list).issubset(set(_question_id_list)) , f'Question ids do not match between results and comparison set'
            except AssertionError as e: 
                missing_question_ids = set(_comparison_question_id_list) - set(_question_id_list)
                self.logger.error(f'Missing question_ids in results: {missing_question_ids}')
                raise
            results_df.to_parquet(self.consolidated_results_path)
            self.logger.info(f'Consolidated search results saved to: {str(self.consolidated_results_path)}')
            

        return results_df

    def _standardize_results(self, df):    # Clean IDs
        if self.result_id_col in df.columns:
            df[self.result_id_col] = df[self.result_id_col].astype(str).str.lower().str.strip()
        return df 
    


    def process_search_results(self, results_df): 
        '''
        Evaluates search results, saves matched and missed results, and returns evaluation metrics
        
        '''
        results_df[self.result_id_col] = results_df[self.result_id_col].str.lower()
        if self.database == 'oa': 
            #remove leading 'https://openalex.org/' from result_id_col
            results_df[self.result_id_col] = results_df[self.result_id_col].str.replace('https://openalex.org/', '').str.lower().str.strip()
            self.eval_groundtruth_df[self.eval_id_col] = self.eval_groundtruth_df[self.eval_id_col].str.replace('https://openalex.org/', '').str.lower().str.strip()

        evalmetrics_df = pd.DataFrame()
        self.logger.info(f'Evaluating matches for {self.database} {self.search_type} {self.strategy_type} {self.vector_search_flag}')

        match_results_df, missed_results_df = self._evaluate_matches(self.eval_groundtruth_df, results_df)

        #overall metrics 
        self.question_id = 'overall'
        metrics_groundtruth = self.calc_eval_metrics(match_results_df, self.eval_groundtruth_df, results_df)
        metrics_groundtruth_df = pd.DataFrame.from_records([asdict(metrics_groundtruth)])
        metrics_groundtruth_df['performance_on'] = 'newlyadded_2023edition'
        metrics_groundtruth_df['question_id'] = self.question_id
        evalmetrics_df = pd.concat([metrics_groundtruth_df], ignore_index=True)
        #save matched and missed results 
        evalmetrics_df['database'] = self.database
        evalmetrics_df['search_type'] = self.search_type
        evalmetrics_df['strategy_type'] = self.strategy_type
        evalmetrics_df['vector_search'] = self.vector_search_flag
        self._save_match_missed_results(match_results_df, missed_results_df)


        if self.search_type == 'topic_specific': 
            grouped_evalmetrics_df = pd.DataFrame()
            for question_id, grouped_groundtruth_df in self.eval_groundtruth_df.groupby('question_id'): 
                self.question_id = question_id
                grouped_df = results_df[results_df['question_id'] == question_id]
                match_results_df, missed_results_df = self._evaluate_matches(grouped_groundtruth_df, grouped_df)
                metrics_groundtruth = self.calc_eval_metrics(match_results_df, grouped_groundtruth_df, grouped_df)
                metrics_groundtruth_df = pd.DataFrame.from_records([asdict(metrics_groundtruth)])
                metrics_groundtruth_df['performance_on'] = 'newlyadded_2023edition'
                metrics_groundtruth_df['question_id'] = question_id
                metrics_groundtruth_df['database'] = self.database
                metrics_groundtruth_df['search_type'] = self.search_type
                metrics_groundtruth_df['strategy_type'] = self.strategy_type
                metrics_groundtruth_df['vector_search'] = self.vector_search_flag
                grouped_evalmetrics_df = pd.concat([grouped_evalmetrics_df, metrics_groundtruth_df], ignore_index=True)
            
                self._save_match_missed_results(match_results_df, missed_results_df)

            evalmetrics_df = pd.concat([evalmetrics_df, grouped_evalmetrics_df], ignore_index=True)


        first_few_cols = ['question_id', 'performance_on', 'database', 'search_type', 'strategy_type', 'vector_search']
        other_cols = [col for col in evalmetrics_df.columns if col not in first_few_cols]
        evalmetrics_df = evalmetrics_df.reindex(columns=first_few_cols + other_cols)

        return evalmetrics_df      

        
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
        
        results_df['clean_id'] = results_df[self.result_id_col].astype(str).str.lower().str.strip().str.replace(r's\+', '')
        comparison_df['clean_id'] = comparison_df[self.eval_id_col].astype(str).str.lower().str.strip().str.replace(r's\+', '')
        results_df_dedupe = results_df.drop_duplicates(subset = 'clean_id')
        # Log after cleaning
        self.logger.info(f"Comparison / evaluation set - Total: {len(comparison_df)}, Unique IDs: {comparison_df['clean_id'].nunique()}")
        self.logger.info(f'Evaluating based on {self.result_id_col} in results df, and {self.eval_id_col} in comparison df')
        # Find matches using merge
        matched_results_df = pd.merge(
            comparison_df,
            results_df_dedupe,
            left_on='clean_id',
            right_on='clean_id',
            how='inner',
            suffixes = ('', '_results')
        )
        
        # Find missed using merge
        missed_results_df = comparison_df[
            ~comparison_df['clean_id'].isin(matched_results_df['clean_id'])
        ]


            
    # Validate
        assert len(matched_results_df) + len(missed_results_df) == len(comparison_df), \
            f"Total rows don't match: {len(matched_results_df)} + {len(missed_results_df)} != {len(comparison_df)}"
        
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
        

        #check length of match df 
        if len(match_df) == 0: 
            self.logger.warning(f'No matches found for {self.question_id}')
            return eval_metrics(nnr = 0, n_retrieved = 0, n_missed = len(comparison_df), recall = 0, precision = 0, f1_score = 0, f2_score = 0, f3_score = 0)
        
        recall = len(match_df) / len(comparison_df)
        precision = len(match_df) / len(raw_results_df)
        nnr = 1/precision
        n_retrieved = len(match_df)
        n_missed = len(comparison_df) - len(match_df)
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

        return eval_metrics(nnr, n_retrieved, n_missed, recall, precision, f1, f2, f3)
    
    def _evaluate_vs_matches(self, evaluation_df : pd.DataFrame, rrf_sim_result_df : pd.DataFrame, database_name : str) -> pd.DataFrame: 


        eval_id_dct = {
            'pubmed' : 'retrieved_pubmed_id',
            'medline' : 'retrieved_pubmed_id', 
            'embase' : 'retrieved_embase_id', 
            'openalex' : 'retrieved_oa_id'
        }
        eval_id_col = eval_id_dct[database_name]
        
        if database_name == 'openalex': 
            #clean up leading https://openalex.org/
            rrf_sim_result_df['original_id'] = rrf_sim_result_df['original_id'].str.replace('https://openalex.org/', '').str.lower().str.strip()
            #make sure same treatment foe evaluation df 
            evaluation_df[eval_id_col] = evaluation_df[eval_id_col].str.replace('https://openalex.org/', '').str.lower().str.strip()
        else: 
            rrf_sim_result_df['original_id'] = rrf_sim_result_df['original_id'].str.lower().str.strip()
            evaluation_df[eval_id_col] = evaluation_df[eval_id_col].str.lower().str.strip()


        matches = evaluation_df.merge(
        rrf_sim_result_df,
        left_on=eval_id_col,
        right_on='original_id',
        how='inner')

        return matches

    
    def calc_vectorsearch_metrics(self, comparison_df: pd.DataFrame, raw_results_df: pd.DataFrame, query_vector_df: pd.DataFrame) -> vectorsearch_eval_metrics:
        '''
        Calculates vector search metrics for a given evaluation set (dependent on evidence_review_id further upstream) raw results set, 
        and query_goldset_df (also dependent on evidence_review_id further upstream)

        Args: 
            comparison_df: dataframe with evaluation set
            raw_results_df: dataframe with raw results set
            query_goldset_df: dataframe with query goldset

        Returns: 
            vectorsearch_eval_metrics: dataclass with vector search metrics
        '''

        if len(comparison_df) == 0: 
            self.logger.warning(f'No comparison / evalutation set found for current evidence_review_id. Verify this is intended. Returning empty results.')
            return vectorsearch_eval_metrics(
                nnr_raw= 0, rank_new_relevant= 0, nnr_threshold= 0, mrr = 0,
                n_retrieved= 0, n_missed= 0, 
                recall= 0, precision= 0, precision_cutoff= 0, 
                f1_score= 0, f2_score= 0, f3_score= 0, 
                recall_at_10= 0, recall_at_100= 0, recall_at_1000= 0, 
                similarity_threshold_cutoff= 0), pd.DataFrame()

        elif len(comparison_df) > 0: 
                        
            #dedupe results df, and make sure that it is sorted by ranking 
            raw_results_df_dedupe = raw_results_df.sort_values('combined_rank_rrf').copy()
            #evaluate matches 
            matches = self._evaluate_vs_matches(comparison_df, raw_results_df_dedupe, self.database)

            if len(matches) > 0:

                recall = len(matches) / len(comparison_df) #recall of the search (raw)
                #non ranked metrics 
                n_retrieved = len(matches) #total number of sucessful matches / results retrived 
                n_missed = len(comparison_df) - len(matches) #total number of missed results 
                precision = len(matches) / len(raw_results_df_dedupe) #precision of the search (raw)
                nnr_raw = 1/precision # total number of results to read before looking at ranking 
                precision_cutoff = precision
                
                f1 = self._calc_fscore(precision, recall, 1)
                f2 = self._calc_fscore(precision, recall, 2)
                f3 = self._calc_fscore(precision, recall, 3)
                #ranked metrics 
                matches_filter_goldset = matches[~matches['ground_truth_article_id'].isin(query_vector_df['ground_truth_article_id'])]
                #find first match that isn't in the goldset (ie: a theoretically new match)
                rank_new_relevant = matches_filter_goldset['combined_rank_rrf'].min()
                rank_first_relevant = matches['combined_rank_rrf'].min()
                mrr_new_relevant = 1/rank_new_relevant #mean reciprocal rank 
                mrr = 1/rank_first_relevant #mean reciprocal rank 
                
            
                recall_at_k_dct = {}
                for k in [10, 100, 1000]: 
                    if len(matches) > 0: 
                        top_k_ids = raw_results_df_dedupe.head(k)['original_id']
                        # Count matches in top k (including duplicates)
                        matches_at_k = matches[matches['original_id'].isin(top_k_ids)].shape[0]
                        recall_at_k = matches_at_k / len(comparison_df)
                    
                    recall_at_k_dct[f'recall_at_{k}'] = recall_at_k

                similarity_threshold_cutoff = matches['cosine_similarity'].min()
                
                result_cutoff_df = raw_results_df_dedupe.query(f'cosine_similarity >= {similarity_threshold_cutoff}')
                
                
                matched_cutoff = self._evaluate_vs_matches(comparison_df, result_cutoff_df, self.database)
                recall_cutoff = len(matched_cutoff) / len(comparison_df)
                precision_cutoff = len(matched_cutoff) / len(result_cutoff_df)
                nnr_threshold = 1/precision_cutoff
                
                assert recall_cutoff == recall 
                 #check recall for the trimmed results 
                recall_at_10, recall_at_100, recall_at_1000 = recall_at_k_dct['recall_at_10'], recall_at_k_dct['recall_at_100'], recall_at_k_dct['recall_at_1000']

            elif len(matches) == 0: 
                self.logger.warning(f'No matches found for current evidence_review_id. Returning 0 recall.')
                return vectorsearch_eval_metrics(
                    nnr_raw= 'N/A', rank_new_relevant= 'N/A', nnr_threshold= 'N/A',
                    n_retrieved= 0, n_missed= len(comparison_df), 
                    recall= 0, precision= 0, precision_cutoff= 0, 
                    f1_score= 0, f2_score= 0, f3_score= 0, 
                    recall_at_10= 0, recall_at_100= 0, recall_at_1000= 0, 
                    similarity_threshold_cutoff= 0, mrr_new_relevant= 0, mrr= 0, rank_first_relevant= 0), pd.DataFrame()
                

            assert (0 <= recall) and (recall <= 1), f'Recall is {recall}, which is not a valid recall value'

            self.logger.info(f'Total number of sucessful matches / results retrived: {n_retrieved}')
            self.logger.info(f'Total number of missed results: {n_missed}')
            self.logger.info(f'Recall of the search (raw): {recall}')
            self.logger.info(f'Precision of the search (raw): {precision}')
            self.logger.info(f'NNR of the search (raw): {nnr_raw}')
            self.logger.info(f'Similarity threshold cutoff: {similarity_threshold_cutoff}')
            self.logger.info(f'Recall of the search (cutoff): {recall_cutoff}')
            self.logger.info(f'Precision of the search (cutoff): {precision_cutoff}')
            self.logger.info(f'NNR of the search (cutoff): {nnr_threshold}')
            self.logger.info(f'Rank of first match not in goldset: {rank_new_relevant}')
            self.logger.info(f'Rank of first relevant match (including goldset): {rank_first_relevant}')
            self.logger.info(f'Mean reciprocal rank (traditional): {mrr}')
            self.logger.info(f'Mean reciprocal rank (including goldset): {mrr_new_relevant}')
            self.logger.info(f'Recall at first 10: {recall_at_10}')
            self.logger.info(f'Recall at first 100: {recall_at_100}')
            self.logger.info(f'Recall at first 1000: {recall_at_1000}')
            
            

            return vectorsearch_eval_metrics(nnr_raw= nnr_raw, rank_new_relevant= rank_new_relevant, nnr_threshold= nnr_threshold, n_retrieved= n_retrieved, n_missed= n_missed, recall= recall, precision= precision, precision_cutoff= precision_cutoff, f1_score= f1, f2_score= f2, f3_score= f3, recall_at_10= recall_at_10, recall_at_100= recall_at_100, recall_at_1000= recall_at_1000, similarity_threshold_cutoff= similarity_threshold_cutoff, mrr_new_relevant= mrr_new_relevant, mrr= mrr, rank_first_relevant= rank_first_relevant), result_cutoff_df
        
    
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

        if self.search_type == 'overarching': 
            match_results_df.to_parquet(self.save_eval_path / f'matched_results_{self.database}_{self.search_type}_{self.strategy_type}.parquet')
            missed_results_df.to_parquet(self.save_eval_missed_results_path / f'missed_results_{self.database}_{self.search_type}_{self.strategy_type}.parquet')

            #also save as csv for troubleshooting (missed reuslts only)
            missed_results_df.to_csv(self.save_eval_missed_results_path / f'missed_results_{self.database}_{self.search_type}_{self.strategy_type}.csv')
            #matched results to csv 
            match_results_df.to_csv(self.save_eval_path / f'matched_results_{self.database}_{self.search_type}_{self.strategy_type}.csv')
        
        elif self.search_type == 'topic_specific': 
            #grab question_id from match_results_df
            question_id = self.question_id
            # Clean question_id
            # Replace dots and slashes with underscores
            question_id = question_id.replace('.', '_')
            question_id = question_id.replace('/', '_')
            
            # Remove backslashes if any
            question_id = re.sub(r'\\+', '', question_id)
            # Remove double backslashes

            #create directory if it doesn't exist
            self.logger.info(f'Saving results for question_id: {question_id}')
            (self.save_eval_missed_results_path / f'question_{question_id}').mkdir(parents=True, exist_ok=True)
            match_results_df.to_parquet(self.save_eval_path / f'question_{question_id}' / f'matched_results_{self.database}_{question_id}.parquet')
            missed_results_df.to_parquet(self.save_eval_missed_results_path / f'question_{question_id}' / f'missed_results_{self.database}_{question_id}.parquet')
            match_results_df.to_csv(self.save_eval_path / f'question_{question_id}' / f'matched_results_{self.database}_{question_id}.csv')
            missed_results_df.to_csv(self.save_eval_missed_results_path / f'question_{question_id}' / f'missed_results_{self.database}_{question_id}.csv')


    def run_eval_pipeline(self): 

        if self.vector_search_flag == False: 

            if self.search_type == 'overarching': 
                result_df = self._load_overarching_search_results()
            elif self.search_type == 'topic_specific': 
                result_df = self._load_topic_specific_search_results()
            
            return self.process_search_results(result_df)

    def run_vectorsearch_eval_pipeline(self, result_set: pd.DataFrame, evaluation_set: pd.DataFrame, query_vector_df: pd.DataFrame, database_name: str, search_strat_df : pd.DataFrame): 
        if len(evaluation_set) == 0: 
            self.logger.warning(f'No evaluation set found for current evidence_review_id. REturning empty results. Potential cause is due to no new articles included for current evidence_review_id. Verify this is intended.')
            return pd.DataFrame(), pd.DataFrame()
        elif len(evaluation_set) > 0: 
            vector_search_metrics, result_cutoff_df = self.calc_vectorsearch_metrics(comparison_df = evaluation_set, raw_results_df = result_set, query_vector_df = query_vector_df)
            vector_search_metrics_df = pd.DataFrame.from_records([asdict(vector_search_metrics)])
            
            #add metadata 
            vector_search_metrics_df['database'] = database_name
            vector_search_metrics_df['search_type'] = search_strat_df['search_type_id']
            vector_search_metrics_df['search_strategy'] = search_strat_df['search_strategy_type_id']
            vector_search_metrics_df['vector_search_type'] = search_strat_df['vector_search']
            vector_search_metrics_df['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M')
            vector_search_metrics_df['search_strategy_id'] = search_strat_df['search_strategy_id']
            vector_search_metrics_df['evidence_review_id'] = search_strat_df['evidence_review_id']
            return vector_search_metrics_df, result_cutoff_df




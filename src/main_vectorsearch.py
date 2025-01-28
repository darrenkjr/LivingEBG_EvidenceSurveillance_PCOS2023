from pathlib import Path
import sys 

src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))
from libraries.vector_search import vector_search_implementation
from libraries.sql_data_migration import sql_data_migration
from libraries.logging_config import LoggerConfig
import os 
import dotenv
dotenv.load_dotenv()
import platform

def main(): 
    ''''
    Main function running vector search 

    1. Migrate data to postgresql 
    2. Generate Embeddings for goldset and searchspace 
    3. Run vector search + RRF 
    4. Evaluate vector search (basic metrics)
    
    '''
    logger = LoggerConfig.setup_logger(logger_name = 'main_vectorsearch')
    # Check both WSL and if PostgreSQL socket exists

    logger.info('Setting up environment variables for postgresql connection')
    wsl_flag = os.environ.get('WSL_DISTRO_NAME') is not None
    if wsl_flag: 
        os.environ['PGHOST'] = '/var/run/postgresql' 
    else: 
        os.environ['PGHOST'] = 'localhost' 
    db_name = os.getenv('DB_NAME')
    db_user = os.getenv('DB_USER')
    db_pwd = os.getenv('DB_PWD')
    db_host = os.getenv('DB_HOST')
    db_port = os.getenv('DB_PORT')

    logger.info(f'Migrating data to postgresql')
    sql_instance = sql_data_migration(db_name, db_user, db_pwd, db_host, db_port, logger)
    
    logger.info(f'Filling reference tables')
    sql_instance.fill_reference_tables()
    
    logger.info(f'Migrating goldset data')
    sql_instance.migrate_gdg_data()
    
    logger.info(f'Migrating ground truth data')
    sql_instance.migrate_ground_truth_data()
    
    logger.info(f'Migrating search strategies')
    
    overarching_search_results_to_insert, topic_specific_search_results_to_insert = sql_instance.migrate_search_strategies()
    
    for search_results in overarching_search_results_to_insert: 
        sql_instance.migrate_search_result_articles(search_results['search_strategy_id'], search_results['search_result_df'], search_results['database_name'], search_results['search_type'])
    for search_results in topic_specific_search_results_to_insert: 
        sql_instance._handle_topic_specific_search(search_results['search_strategy_id_list'], search_results['evidence_review_id_list'], search_results['file_path'], search_results['database_name'], search_results['search_type'])

    logger.info('Initializing vector search')
    vector_search_cls = vector_search_implementation(logger = logger)
    for search_type in ['overarching', 'topic-specific']: 
        logger.info(f'Running vector search implementation for search type: {search_type}')
        eval_metrics_df = vector_search_cls.run_vector_search(search_type)
        eval_results_path = Path(__file__).parent / 'evaluation_results' / 'overall' 
        eval_results_path.mkdir(parents=True, exist_ok=True) 

    include_header = True
    eval_metrics_df.to_csv(
        eval_results_path / 'overall_evalmetrics_df.csv',  # path is the first positional argument
        mode='a',  # append mode
        header=not (eval_results_path / 'overall_evalmetrics_df.csv').exists() if include_header else False,
        index=False
    )

if __name__ == '__main__': 
    main()


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
import pandas as pd 
dotenv.load_dotenv()
import platform
from pandas import ExcelWriter
from openpyxl import load_workbook
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

    #perform data migration checks 


    logger.info(f'Performing data migration checks')
    sql_instance = sql_data_migration(db_name, db_user, db_pwd, db_host, db_port, logger)
    if not sql_instance.check_data_migration(): 
        logger.info(f'Data migration checks failed, regenerating all tables')
        sql_instance.drop_all_tables_data()
        logger.info(f'Creating all tables')
        sql_instance._create_tables()
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
    
    elif sql_instance.check_data_migration(): 
        logger.info(f'Data migration checks passed, skipping data migration')

        
    logger.info('Initializing vector search')
    vector_search_cls = vector_search_implementation(logger = logger)
    vector_search_cls.sql_procedures.create_querygoldset_view()
    vector_search_cls.sql_procedures.create_evaluation_set_view_2017included()
    vector_search_cls.sql_procedures.create_query_evidencereview_topic_view()
       #run check if embeddings already exist
    vector_search_cls.generate_embeddings_if_needed()

    for search_type in ['topic-specific']: 
        logger.info(f'Running vector search implementation for search type: {search_type}')
        eval_metrics_df = pd.DataFrame()


        for vector_search_type in ['zero-shot', 'one-shot', 'few-shot']: 
            _metrics_df = vector_search_cls.run_vector_search(search_type, vector_search_type)
            eval_results_path = Path(__file__).parent / 'evaluation_results' / 'overall' 
            eval_results_path.mkdir(parents=True, exist_ok=True) 
            _metrics_df['vector_search_type'] = vector_search_type
            eval_metrics_df = pd.concat([eval_metrics_df, _metrics_df])
        
        sheet_name = f'{search_type}_vs'

        #temporary - eventually add a function to write to database
        excel_path = eval_results_path / 'overall_evalmetrics_df.xlsx'
        logger.info(f'Saving evaluation metrics for {search_type} vector search to {excel_path}')
        with ExcelWriter(excel_path, 
                        mode='a' if excel_path.exists() else 'w',
                        engine='openpyxl',
                        if_sheet_exists='replace') as writer:
            eval_metrics_df.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False
            )
        logger.info(f'Vector search complete for {search_type}')

    logger.info(f'Vector search complete')

if __name__ == '__main__': 
    main()


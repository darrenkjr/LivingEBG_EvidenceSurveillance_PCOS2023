from pathlib import Path
import sys 

src_path = Path(__file__).parent
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))
from libraries.vector_search import vector_search_implementation
from libraries.sql_data_migration import sql_data_migration
from libraries.logging_config import LoggerConfig
from libraries.sql_procedures import sql_procedures
import os 
import dotenv
import pandas as pd 
dotenv.load_dotenv()
from sqlalchemy import create_engine
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
    engine = create_engine(f'postgresql://{db_user}:{db_pwd}@{db_host}:{db_port}/{db_name}')

    #perform data migration checks 


    logger.info(f'Performing data migration checks')
    sql_instance = sql_data_migration(db_name, db_user, db_pwd, db_host, db_port, logger, engine)
    if not sql_instance.check_data_migration(): 
        logger.warning(f'Data migration checks failed, regenerating all tables')
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
            logger.info(f'Migrating overarching search results')
            sql_instance.migrate_search_result_articles(search_results['search_strategy_id'], search_results['search_result_df'], search_results['database_name'], search_results['search_type'])
        for search_results in topic_specific_search_results_to_insert: 
            logger.info(f'Migrating topic specific search results')
            sql_instance._handle_topic_specific_search(search_results['search_strategy_id_list'], search_results['evidence_review_id_list'], search_results['file_path'], search_results['database_name'], search_results['search_type'])
    
    elif sql_instance.check_data_migration(): 
        logger.info(f'Data migration checks passed, skipping data migration')

        
    logger.info('Initializing vector search')
    sql_procedures_cls = sql_procedures(logger = logger, engine = engine)
    sql_procedures_cls.create_querygoldset_view()
    sql_procedures_cls.create_evaluation_set_view_2017included()
    sql_procedures_cls.create_query_evidencereview_topic_view()
    vector_search_cls = vector_search_implementation(logger = logger, engine = engine)
    vector_search_cls.generate_embeddings_if_needed()

    for search_type in ['topic-specific','overarching']: 
        logger.info(f'Running vector search implementation for search type: {search_type}')
        eval_metrics_df = pd.DataFrame()


        for vector_search_type in ['one-shot', 'few-shot', 'zero-shot']: 
            _metrics_df = vector_search_cls.run_vector_search(search_type, vector_search_type)
            eval_results_path = Path(__file__).parent / 'evaluation_results' / 'overall' 
            eval_results_path.mkdir(parents=True, exist_ok=True) 
            _metrics_df['vector_search_type'] = vector_search_type
            eval_metrics_df = pd.concat([eval_metrics_df, _metrics_df])
        
        sheet_name = f'{search_type}_vs'

        #temporary - eventually add a function to write to database
        excel_path = eval_results_path / 'overall_evalmetrics_df.xlsx'
        if excel_path.exists():
            try:
                # Try to load existing data from sheet
                existing_df = pd.read_excel(excel_path, sheet_name=sheet_name)
                combined_df = pd.concat([existing_df, eval_metrics_df], ignore_index=True)
                
                with pd.ExcelWriter(excel_path, mode='a', if_sheet_exists='replace') as writer:
                    combined_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
            except ValueError as e:  # Sheet doesn't exist
                with pd.ExcelWriter(excel_path, mode='a') as writer:
                    eval_metrics_df.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            # New file
            eval_metrics_df.to_excel(
                excel_path,
                sheet_name=sheet_name,
                index=False
            )
            logger.info(f'Vector search complete for {search_type}')

    logger.info(f'Vector search complete')

if __name__ == '__main__': 
    main()


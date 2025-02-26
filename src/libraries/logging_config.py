import logging 
from pathlib import Path
from datetime import datetime 
import colorlog

class LoggerConfig: 

    @staticmethod
    def setup_logger(
        logger_name: str, 
        log_dir : Path = Path(__file__).parent.parent / 'logs', 
        console_log_level : int = logging.INFO,
        file_log_level : int = logging.WARNING,
        log_level : int = logging.DEBUG
    ) -> logging.Logger: 
        
        '''
        Set up logger for inspection later: 

        Args: 
            logger_name (str): Name of the logger (eg: oa_overarching_search)
            log_dir: where log files are to be saved 
            log_level: logging level (eg: logging.DEBUG, logging_info)

        Returns: 
            logger.logger : ie: a logger instance to track progress and errors  
        '''
        #create log directory if it doesn't exist 
        log_dir.mkdir(parents=True, exist_ok=True)

        #create logger
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)

        if not logger.handlers: 

            #create file handler to write to log file 
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
            file_handler = logging.FileHandler(log_dir / f'{logger_name}_{timestamp}.log')
            file_handler.setLevel(file_log_level)

            #create stream handler to print to console 
            console_handler = logging.StreamHandler()
            console_handler.setLevel(console_log_level)

            #create formatter 
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)

            #color coding for console 
            console_formatter = colorlog.ColoredFormatter(
                '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
            console_handler.setFormatter(console_formatter)

            #add handlers to logger 
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        return logger 


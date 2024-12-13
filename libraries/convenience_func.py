import polars as pl 
from datetime import datetime, timedelta

def date_range_generator(start_date = '1960-01-01', end_date = '2022-09-30'):
    '''
    Generate a list of non-overlapping date ranges, one year apart

    Args:
        start_date (str): Start date in the format 'YYYY-MM-DD'
        end_date (str): End date in the format 'YYYY-MM-DD'

    Returns:
        list: List of dictionaries with keys 'start_date' and 'end_date' for each range 
    '''
    date_ranges = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    
    while current_date < end:
        # Calculate the end of this range (either next year - 1 day, or the final end date)
        range_end = min(
            datetime(current_date.year + 1, current_date.month, current_date.day),
            end
        )
        
        date_ranges.append({
            'start': current_date.strftime('%Y-%m-%d'),
            'end': range_end.strftime('%Y-%m-%d')
        })
        
        # Move to start of next range
        current_date = range_end
    return date_ranges
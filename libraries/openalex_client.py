import pandas as pd 
from datetime import datetime, timedelta
import asyncio 
import aiohttp
from typing import List, Dict
import dotenv
dotenv.load_dotenv()
import os 
from libraries.convenience_func import ConvenienceFunc
from urllib.parse import urlparse, parse_qsl, urlencode
class OpenAlexClient:

    def __init__(self, max_concurrent_requests: int = 3, max_retries: int = 4, logger = None):
        """Initialize client with rate limiting queue"""
        self.logger = logger
        self.email = os.getenv('email')
        self.request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.result_queue = asyncio.Queue()
        self.max_retries = max_retries
        self.date_list = ConvenienceFunc(logger=logger).date_range_generator()
        self.session = None
        self.task_queue = asyncio.Queue()
        self.max_concurrent_requests = max_concurrent_requests
        self.logger.info(f'Initializing OpenAlexClient with following params: Email: {self.email}, Max concurrent requests: {self.max_concurrent_requests}, Max retries: {self.max_retries}')
        

    async def __aenter__(self):
        """Async context manager entry"""
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            self.session = None

    def _build_topicsearch_url(self, topic_id: str, start_date: str, end_date: str, cursor: str = '*'):
        """Build URL with email parameter for polite pool"""
        return (f"https://api.openalex.org/works?"
                f"select=id,ids,title,abstract_inverted_index,publication_date&"
                f"filter=primary_topic.id:{topic_id},"
                f"type:article|review,"
                f"from_publication_date:{start_date},"
                f"to_publication_date:{end_date}&"
                f"per-page=200&"
                f"cursor={cursor}&"
                f"mailto={self.email}")
    
    def _build_oa_topics_url(self, id_list:list): 
        '''
        Build OA api URL for each OA id chunk in list 
        '''
        id_chunk = '|'.join(id_list)
        return f'https://api.openalex.org/works?filter=ids.openalex:{id_chunk}&select=id,title,primary_topic,primary_location&per-page=200&mailto={self.email}'

        
    async def retrieve_oa_data(self, oa_ids): 
        '''
        Retrieve topics for each oa id in the oa gold set df 
        '''

        #take the od_ids series and split into chunks of 50
        self.logger.info(f'Retrieving OA data for {len(oa_ids)} ids')

        if isinstance(oa_ids, pd.Series): 
            oa_ids_list = oa_ids.tolist()
        else: 
            oa_ids_list = oa_ids
        oa_ids_chunks = [oa_ids_list[i:i+100] for i in range(0, len(oa_ids_list), 100)]
        #build url for each chunk

       #start session if not already started 
        if not self.session: 
            self.session = aiohttp.ClientSession()

        #retrieve data for each url, 1 task per url / chunk 
        for chunk in oa_ids_chunks: 
            url = self._build_oa_topics_url(chunk)
            await self.task_queue.put((chunk, url))

        completed_tasks = 0 
        task_count = self.task_queue.qsize()
        self.logger.info(f'Starting task queue for OA retrieval. {task_count} tasks to complete.')

        worker_tasks = [
            asyncio.create_task(self._worker_task()) for _ in range(self.max_concurrent_requests)
        ]

        all_results = list()
        while completed_tasks < task_count: 
            try: 
                results = await self.result_queue.get() 
                if results: 
                    #results is a tuple of (date_range (dict), results (list of dictionaries))
                    all_results.append(results) 
                    completed_tasks += 1
                    self.logger.info(f'Completed {completed_tasks} of {task_count} tasks')

                self.result_queue.task_done()
            except Exception as e: 
                self.logger.error(f'Error processing results: {e}')

        for task in worker_tasks: 
            task.cancel()
        
        await asyncio.gather(*worker_tasks)

        if all_results: 
            all_results_df = pd.DataFrame()
            self.logger.info('Processing OA results')
            for url_chunk, api_results in all_results: 
                all_results_df = pd.concat([all_results_df, pd.DataFrame(api_results)], ignore_index=True)
            return pd.DataFrame(all_results_df)
        else: 
            return pd.DataFrame()

    

    async def retrieve_data(self, url: str) -> List[Dict]:
        """Retrieve data with proper rate limiting"""
        cursor = '*'
        results = []
        retry_count = 0
    
        async with self.request_semaphore:  # Use semaphore for rate limiting
            while cursor:
                try:
                    # Update URL with cursor
                    parsed_url = urlparse(url)
                    query_params = dict(parse_qsl(parsed_url.query))
                    if cursor != '*':
                        query_params['cursor'] = cursor
                    new_query = urlencode(query_params)
                    current_url = parsed_url._replace(query=new_query).geturl()
                    async with self.session.get(current_url) as response:
                        if response.status == 200:
                            data = await response.json()
                            cursor = data['meta'].get('next_cursor')
                            #get total count from api 
                            results.extend(data['results'])
                            retry_count = 0
                            
                            if cursor is None:
                                self.logger.info('Reached end of pagination')
                                break
                                
                        elif response.status == 429:  # Rate limit hit
                            self.logger.warning(f'Rate limit exceeded for {current_url} - waiting...')
                            url_filter = query_params['filter']
                            self.logger.warning(f'Filter params for reference: {url_filter}')
                            retry_after = int(response.headers.get('Retry-After', '5'))
                            await asyncio.sleep(retry_after)
                            continue
                            
                        else:
                            response.raise_for_status()
                            
                    # Rate limiting delay
                    await asyncio.sleep(0.1)
                

                except aiohttp.ClientError as ce:
                    self.logger.error(f'Network/HTTP Error: {str(ce)}, current url: {current_url}')
                    url_filter = query_params['filter']
                    self.logger.error(f'Filter params for reference: {url_filter}')
                    retry_count += 1
                except Exception as e:
                    self.logger.error(f'Unexpected Error: {str(e)}, current url: {current_url}')
                    url_filter = query_params['filter']
                    self.logger.error(f'Filter params for reference: {url_filter}')
                    retry_count += 1

                if retry_count >= self.max_retries:
                    self.logger.warning(f'Max retries ({self.max_retries}) reached for {current_url}. Stopping.')
                    self.logger.error(f'Filter params for reference: {url_filter}')
                    break
                elif retry_count > 0:
                    wait_time = 2 ** retry_count
                    self.logger.warning(f'Retrying {current_url} in {wait_time} seconds...')
                    url_filter = query_params['filter']
                    self.warning(f'Filter params for reference: {url_filter}')
                    await asyncio.sleep(wait_time)
                    continue 
                
                    
        return results
    

    async def _worker_task(self): 
        '''
        Worker task to process a single date range and return results, depending on queue and concurrency limits 
        '''
        while True: 
            try: 
                #check if we should kill the worker 
                if self.task_queue.empty(): 
                    break 
                
                #grab available task 
                try: 
                    date_range, url = await asyncio.wait_for(self.task_queue.get(), timeout=60)

                except asyncio.TimeoutError: 
                    self.logger.warning('Timeout waiting for task - skipping')
                    break
                
                try: 
                    #process the task 
                    results = await self.retrieve_data(url)
                    if results: 
                        await self.result_queue.put((date_range, results))
                        remaining = self.task_queue.qsize()
                        self.logger.info('Task completed.')
                        self.logger.info(f'Status: {remaining} tasks remaining')

                except Exception as e: 
                    self.logger.error(f'Error processing task: {str(e)}')

                finally: 
                    self.task_queue.task_done()

            except asyncio.CancelledError: 
                break
            except Exception as e: 
                self.logger.error(f'Unexpected error in worker task: {str(e)}')
                break


    async def get_overarching_paginated_search_results(self, topic_id: str):
        """Process date ranges with controlled concurrency"""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        try:
            
            for date_range in self.date_list:
                self.logger.info(f'Building topic search URL for date range: {date_range}')
                url = self._build_topicsearch_url(topic_id, 
                                    start_date=date_range['start'],
                                    end_date=date_range['end'])
                await self.task_queue.put((date_range, url))
                
             

            completed_tasks = 0 
            task_count = self.task_queue.qsize()
            self.logger.info(f'Starting with {task_count} tasks')

            worker_tasks = [
                asyncio.create_task(self._worker_task()) for _ in range(self.max_concurrent_requests)
            ]

            all_results = list()
            while completed_tasks < task_count: 
                try: 
                    results = await self.result_queue.get() 
                    if results: 
                        #results is a tuple of (date_range (dict), results (list of dictionaries))
                        all_results.append(results) 
                        completed_tasks += 1
                        self.logger.info(f'Completed {completed_tasks} of {task_count} tasks')

                    self.result_queue.task_done()
                except Exception as e: 
                    self.logger.error(f'Error processing results: {e}')

            for task in worker_tasks: 
                task.cancel()
            
            await asyncio.gather(*worker_tasks)
            
            #all_results is a list of tuples (date_range (dict), results (list of dictionaries)), need to be converted to a dataframe. Note that publicaton date is ISO 8601 format 
            if all_results: 
                self.logger.info('Processing Results')
                all_results_df = pd.DataFrame()
                #unpack list of tuples
                for date_range, result_json in all_results: 
                    start_date = pd.to_datetime(date_range['start'], format = 'ISO8601')
                    end_date = pd.to_datetime(date_range['end'], format = 'ISO8601')
                    
                    # Convert list of JSON objects to DataFrame using json_normalize
                    results_df = pd.DataFrame(result_json)

                    # Convert dates
                    if 'publication_date' in results_df.columns:
                        results_df['publication_date'] = pd.to_datetime(results_df['publication_date'], format='ISO8601')
                    
                    results_df['retrieved_startdate'] = start_date
                    results_df['retrieved_enddate'] = end_date
                    results_df['retrieved_daterange'] = f'{start_date} - {end_date}'
                    
                    all_results_df = pd.concat([all_results_df, results_df], ignore_index=True)

                #unpack nested json columns (ids, and abstract inverted indices)
                id_df = pd.json_normalize(all_results_df['ids'])
                all_results_df = pd.concat([all_results_df, id_df], axis=1)
                all_results_df.drop(columns = 'ids', inplace=True)

                # Vectorized abstract decoding using numpy arrays for better performance
                mask = ~all_results_df['abstract_inverted_index'].isna()  # Create boolean mask for non-null abstracts
                all_results_df['abstract'] = None  # Initialize abstract column with None values
                if mask.any():  # Only process if there are any non-null abstracts
                    abstracts = all_results_df.loc[mask, 'abstract_inverted_index'].map(
                        lambda x: ' '.join(k for k, _ in sorted(
                            [(word, pos[0]) for word, pos in x.items() for _ in pos],
                            key=lambda x: x[1]
                        ))
                    )
                    all_results_df.loc[mask, 'abstract'] = abstracts

                all_results_df.drop(columns = 'abstract_inverted_index', inplace=True)
                return all_results_df   

            else: 
                return pd.DataFrame()
        
        finally: 
            if self.session: 
                await self.session.close()
                self.session = None

    def validate_results(self, results_df: pd.DataFrame, topic_id: str): 
        #run quick request - grap header results, validate against results df 
        assert len(results_df) == len(results_df['id'].unique()), self.logger.error('Number of results does not match number of unique ids')

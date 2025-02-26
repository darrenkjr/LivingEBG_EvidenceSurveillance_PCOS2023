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
import hashlib


class OpenAlexClient:

    def __init__(self, max_concurrent_requests: int = 3, max_retries: int = 4, logger = None):
        """Initialize client with rate limiting queue"""
        self.logger = logger
        self.email = os.getenv('email')
        self.api_key = hashlib.md5(self.email.encode()).hexdigest()
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
                f"select=id,ids,title,abstract_inverted_index,publication_year&"
                f"filter=primary_topic.id:{topic_id},"
                f"type:article|review,"
                f"from_publication_date:{start_date},"
                f"to_publication_date:{end_date}&"
                f"per-page=200&"
                f"cursor={cursor}&"
                f"apikey={self.api_key}")
    
    def _build_oa_topics_url(self, id_list:list): 
        '''
        Build OA api URL for each OA id chunk in list 
        '''
        id_chunk = '|'.join(id_list)
        return f'https://api.openalex.org/works?filter=ids.openalex:{id_chunk}&select=id,title,primary_topic,primary_location&per-page=200&api_key={self.api_key}'

    def _build_oa_booleankw_search_url(self, query: list, start_date: str, end_date: str, cursor: str = '*', search_filter: str = 'title_and_abstract.search'): 
        '''
        Build OA api URL for each keyword query in list. 
        search_filter: title_and_abstract.search, default.search (title, abs and full text), title.search (title only), abstract.search (abstract only)
        '''
        return f'https://api.openalex.org/works?filter={search_filter}:{query},from_publication_date:{start_date},to_publication_date:{end_date}&select=id,ids,title,abstract_inverted_index,publication_year&per-page=200&cursor={cursor}&api_key={self.api_key}'
    

    async def retrieve_oa_kwsearch_data(self, query : str) -> pd.DataFrame: 

        '''
        loops through keyword queries in given query_list and retrieve OA search results 
        '''
        session_created = False

        #take a look at oa_paginaed_results schema for similar pattern for use here - use   _build_oa_booleankw_search_url to build uRL to make async requests
        
        try:
            if not self.session and not hasattr(self, '_context_session'):
                self.session = aiohttp.ClientSession()
                session_created = True

            for date_range in self.date_list: 
                url = self._build_oa_booleankw_search_url(query, date_range['start'], date_range['end'])
                self.logger.info(f'Adding API request to queue: {url}')
                #put stuff in task queue 
                await self.task_queue.put((date_range, url))
            
            all_results = await self._create_execute_tasks()
            
            if not all_results: 
                self.logger.warning('No results found for query')
                return pd.DataFrame()
            
            return self._process_results(all_results)

        except Exception as e: 
            self.logger.error(f'Error creating and executing tasks: {e}')
            raise

        #clean up 
        finally:
            if session_created and self.session:
                await self.session.close()
                self.session = None

    async def _create_execute_tasks(self): 
        '''
        Manages task queue and worker tasks
        '''
        workers = []
        completed_tasks = 0 
        task_count = self.task_queue.qsize()
        results = []
        self.logger.info(f'Starting task queue for OA boolean keyword search retrieval. {task_count} tasks to complete.')

        try:
            # Create fixed number of workers
            for _ in range(min(self.max_concurrent_requests, task_count)):
                worker = asyncio.create_task(self._worker_task())
                workers.append(worker)

            # Wait for all tasks to be processed
            try: 
                await asyncio.wait_for(self.task_queue.join(), timeout = 1800)

            except asyncio.TimeoutError as e: 
                self.logger.error(f'Timeout error: {e}')
                raise
            except Exception as e: 
                self.logger.error(f'Error joining task queue: {e}')
                raise
            
            # Cancel any remaining workers
            for w in workers:
                w.cancel()
            
            # Wait for workers to finish
            await asyncio.gather(*workers, return_exceptions=True)
            
            # Collect results
            while not self.result_queue.empty():
                result = await self.result_queue.get()
                if result: 
                    results.append(result)
                    self.result_queue.task_done()
            return results

        except Exception as e:
            self.logger.error(f"Error in task execution: {e}")
            # Cancel all workers on error
            for w in workers:
                if not w.done():
                    w.cancel()
            raise
    
    def _process_results(self, all_results: list): 
        '''
        Process results from the task queue
        '''
        if all_results and isinstance(all_results, list): 
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
        
        else: 
            raise Exception('No results to process')

        #unpack nested json columns (ids, and abstract inverted indices)
        id_df = pd.json_normalize(all_results_df['ids'])
        all_results_df = pd.concat([all_results_df, id_df], axis=1)
        all_results_df.drop(columns = 'ids', inplace=True)

        # Vectorized abstract decoding
        if 'abstract_inverted_index' in all_results_df.columns: 
            all_results_df = self._unpack_abstracts(all_results_df)
        
        return all_results_df   
    
    @staticmethod 
    def _unpack_abstracts(all_results_df: pd.DataFrame): 
        '''
        Unpack abstracts from inverted index
        '''
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

    

    async def _retrieve_data(self, url: str) -> List[Dict]:
        """Retrieve data with proper rate limiting. Is executed in a worker task
        """
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

                    async with asyncio.timeout(180):    
                        async with self.session.get(current_url) as response:
                            if response.status == 200:
                                data = await response.json()
                                cursor = data['meta'].get('next_cursor')
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
                
                except asyncio.TimeoutError as te:
                    retry_count += 1 
                    wait_time = min(2 ** retry_count, 180)
                    self.logger.warning(f'Timeout error: {str(te)}, retrying in {wait_time} seconds...')
                    await asyncio.sleep(wait_time)
                

                except aiohttp.ClientError as ce:

                    retry_count += 1
                    wait_time = min(2 ** retry_count, 180)
                    self.logger.error(f'Network/HTTP Error: {str(ce)}, current url: {current_url}')
                    url_filter = query_params['filter']
                    self.logger.error(f'Filter params for reference: {url_filter}')
                    await asyncio.sleep(wait_time)

                except Exception as e:
                    self.logger.error(f'Unexpected Error: {str(e)}, current url: {current_url}')
                    url_filter = query_params['filter']
                    self.logger.error(f'Filter params for reference: {url_filter}')
                    retry_count += 1

                if retry_count >= self.max_retries:
                    self.logger.error(f'Max retries ({self.max_retries}) reached for {current_url}. Stopping.')
                    self.logger.error(f'Filter params for reference: {url_filter}')
                    break

                #small sleep time to prevent bursty requests 
                await asyncio.sleep(0.1)

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
                date_range, url = await asyncio.wait_for(self.task_queue.get(), timeout=60)
                try: 
                    
                    results = await asyncio.wait_for(self._retrieve_data(url), timeout=300)

                    if results: 
                        await self.result_queue.put((date_range, results))
                        remaining = self.task_queue.qsize()
                        self.logger.info('Task completed.')
                        self.logger.info(f'Status: {remaining} tasks remaining')

                except asyncio.TimeoutError as te:
                    self.logger.error(f'Timeout error: {str(te)}')
                    # Put the task back in queue for retry
                    await self.task_queue.put((date_range, url))
                    continue
            
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




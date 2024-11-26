import polars as pl 
from datetime import datetime, timedelta
import asyncio 
import aiohttp
from typing import List, Dict
import dotenv
import os 
from libraries.convenience_func import date_range_generator
from urllib.parse import urlparse, parse_qsl, urlencode
import random 

class OpenAlexClient:
    def __init__(self, max_concurrent_requests: int = 10, max_retries: int = 5):
        """Initialize client with rate limiting queue"""
        dotenv.load_dotenv()
        self.email = os.getenv('email')
        self.request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.result_queue = asyncio.Queue()
        self.max_retries = max_retries
        self.date_list = date_range_generator()
        self.session = None
        self.task_queue = asyncio.Queue()
        self.max_concurrent_requests = max_concurrent_requests

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

    def _build_url(self, topic_id: str, start_date: str, end_date: str, cursor: str = '*'):
        """Build URL with email parameter for polite pool"""
        return (f"https://api.openalex.org/works?"
                f"select=id,ids,title,abstract_inverted_index&"
                f"filter=primary_topic.id:{topic_id},"
                f"type:article|review,"
                f"from_publication_date:{start_date},"
                f"to_publication_date:{end_date}&"
                f"per-page=200&"
                f"cursor={cursor}&"
                f"mailto={self.email}")
    

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
                            results.extend(data['results'])
                            retry_count = 0
                            
                            if cursor is None:
                                print('Reached end of pagination')
                                break
                                
                        elif response.status == 429:  # Rate limit hit
                            print('Rate limit exceeded - waiting...')
                            retry_after = int(response.headers.get('Retry-After', '5'))
                            await asyncio.sleep(retry_after)
                            continue
                            
                        else:
                            response.raise_for_status()
                            
                    # Rate limiting delay
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    print(f'Error retrieving data: {str(e)}')
                    retry_count += 1
                    if retry_count >= self.max_retries:
                        break
                    await asyncio.sleep(2 ** retry_count)
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
                    print('Timeout waiting for task - skipping')
                    break
                
                try: 
                    #process the task 
                    results = await self.retrieve_data(url)
                    if results: 
                        await self.result_queue.put((date_range, results))
                        remaining = self.task_queue.qsize()
                        print('Task completed for date range: ', date_range)
                        print('Status: ', f'{remaining} tasks remaining')

                except Exception as e: 
                    print(f'Error processing task: {e}')

                finally: 
                    self.task_queue.task_done()

            except asyncio.CancelledError: 
                break
            except Exception as e: 
                print(f'Unexpected error in worker task: {e}')
                break


    async def get_paginated_results(self, topic_id: str):
        """Process date ranges with controlled concurrency"""
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        try:
            
            for date_range in self.date_list:
                url = self._build_url(topic_id, 
                                    start_date=date_range['start'],
                                    end_date=date_range['end'])
                await self.task_queue.put((date_range, url))
                
            all_results = []

            completed_tasks = 0 
            task_count = self.task_queue.qsize()
            print(f'Starting with {task_count} tasks')

            worker_tasks = [
                asyncio.create_task(self._worker_task()) for _ in range(self.max_concurrent_requests)
            ]

            while completed_tasks < task_count: 
                try: 
                    results = await self.result_queue.get() 
                    if results: 

                        completed_tasks += 1
                        print(f'Completed {completed_tasks} of {task_count} tasks')
                        all_results.extend(results)
                    self.result_queue.task_done()
                except Exception as e: 
                    print(f'Error processing results: {e}')

            for task in worker_tasks: 
                task.cancel()
            
            await asyncio.gather(*worker_tasks) 
            return pl.DataFrame(all_results) if all_results else pl.DataFrame()
        
        finally: 
            if self.session: 
                await self.session.close()
                self.session = None


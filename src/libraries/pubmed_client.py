import dotenv
dotenv.load_dotenv()
import pandas as pd 
from datetime import datetime, timedelta
import asyncio 
import aiohttp
from typing import List, Dict
import os 
from libraries.convenience_func import ConvenienceFunc
from urllib.parse import urlparse, parse_qsl, urlencode
from metapub import PubMedFetcher
from metapub.exceptions import MetaPubError
from eutils import EutilsRequestError, EutilsNCBIError
from urllib3.exceptions import ConnectTimeoutError
from requests.exceptions import ConnectTimeout, RequestException
import random 
import traceback


class PubMedClient: 

    def __init__(self, max_concurrent_requests: int = 8, max_retries: int = 3, logger = None): 
        self.logger = logger
                # Debug: Before getting API key
        #date range generator to batch results - 2 year intervals from 1950 to 2022 
        self.request_semaphore = asyncio.Semaphore(max_concurrent_requests)
        #check metapub has api key 
        self.api_key = os.getenv('NCBI_API_KEY')
        os.environ['NCBI_API_KEY'] = self.api_key
        self.result_queue = asyncio.Queue()
        self.max_retries = max_retries
        self.task_queue = asyncio.Queue()
        self.max_concurrent_requests = max_concurrent_requests

        

    async def get_pubmed_search_results_batching(self, search_query : str):

        '''
        async wrapper around metapub implementation of the PubMed API 
        '''
        
        self.pubmed_retrieval = 'pmid'
        date_list = ConvenienceFunc(logger=self.logger).date_range_generator()

        #for every date batch create a task and add to queue 
        for date_batch in date_list:
            await self.task_queue.put((date_batch, search_query))

        completed_tasks = 0 
        task_count = self.task_queue.qsize()
        self.logger.info(f'Starting task queue for PubMed Boolean KW retrieval. {task_count} tasks to complete, batching by 2 year intervals')

        #create async tasks 
        worker_tasks = [
            asyncio.create_task(self._worker_task(client_function = self.async_retrieve_pubmed_data)) for _ in range(self.max_concurrent_requests)
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
            pmid_result_list = []
            self.logger.info('Processing PubMed results')
            for date_range, pmid_lists in all_results:
                pmid_result_list.extend(pmid_lists)
            all_results_df = pd.DataFrame(pmid_result_list, columns = ['pmid'])
            return all_results_df 
        else: 
            return pd.Series()
        
    
    async def _worker_task(self, client_function): 
        '''
        Worker task to process a single date range and return results, depending on queue and concurrency limits 
        '''

        while True: 
            try: 
                #grab available task 
                try: 
                    if self.pubmed_retrieval == 'pmid': 
                        date_range, query = await asyncio.wait_for(self.task_queue.get(), timeout=60)
                    if self.pubmed_retrieval == 'article_details': 
                        pmid = await asyncio.wait_for(self.task_queue.get(), timeout=60)

                except asyncio.TimeoutError: 
                    self.logger.warning('Timeout waiting for task - skipping')
                    break
                
                try: 
                    #process the task 
                    if self.pubmed_retrieval == 'pmid': 
                        results = await client_function(date_range, query)
                        await self.result_queue.put((date_range, results))

                    if self.pubmed_retrieval == 'article_details': 
                        results = await client_function(pmid)
                        await self.result_queue.put(results)
                    if results: 
                        remaining = self.task_queue.qsize()
                        self.logger.info(f'Status: {remaining} tasks remaining')

                except Exception as e: 
                    self.logger.error(f'Error processing task: {str(e)}')

                finally: 
                    self.task_queue.task_done()

            except asyncio.CancelledError: 
                break
            except Exception as e: 
                self.logger.error(f'Unexpected error in worker task: {str(e)}')
                self.logger.error(f"Traceback:\n{traceback.format_exc()}")
                break

    async def async_retrieve_pubmed_data(self, date_range: dict, query: str): 
        '''
        Async retrieve data with proper rate limiting
        '''
        start_date = date_range['start']
        end_date=date_range['end']

        results = [] 
        retstart = 0 
        #to allow pagination 
        chunk_size = 1000

        try: 
            while True: 

                async with self.request_semaphore: 
                    #put syncornous function on its own thread 
                    fetcher = PubMedFetcher()
                    self.logger.info(f'Sending PubMed Request with following params: Query: {query}, Since: {start_date}, Until : {end_date}, Current Chunk Start: {retstart}')
                    result_chunk = await asyncio.to_thread(
                        fetcher.pmids_for_query,
                    query=query,
                    since=start_date,
                    until=end_date,
                    retstart=retstart,
                    retmax=chunk_size
                    )

                if not result_chunk or len(result_chunk) < chunk_size: 
                    
                    if result_chunk:
                        self.logger.info('Reached end of pagination..') 
                        self.logger.info(f"Retrieved {len(results)} results so far for {start_date} to {end_date}")
                        results.extend(result_chunk)
                    break 

                results.extend(result_chunk)
                retstart += chunk_size
                self.logger.info(f"Retrieved {len(results)} results so far for {start_date} to {end_date}")

        except MetaPubError as e: 
            error_msg = str(e)
            self.logger.error(f"Caught MetaPubError: {error_msg}") 
            return None
        except Exception as e: 
            self.logger.error(f"Unexpected error: {str(e)}") 
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            raise

        return results 
    

    async def async_retrieve_pubmed_article_data(self, pmid: str): 

        retries = 0
        while retries < self.max_retries: 
            try: 
                async with self.request_semaphore: 
                    #add small delay 
                    await asyncio.sleep(random.uniform(0.1, 0.3))
                    fetcher = PubMedFetcher()
                    self.logger.info(f'Sending PubMed Request with following params: PMID: {pmid}')
                    result = await asyncio.to_thread(fetcher.article_by_pmid, pmid)
                return result 
            
            except (MetaPubError, EutilsRequestError) as e:  
                full_traceback = traceback.format_exc()
                if 'Too Many Requests (429)' in full_traceback: 
                    #check api key 
                    fetcher_apikey = fetcher.qs.api_key
                    self.logger.error(
                        f"API rate limit exceeded for PMID {pmid}, "
                        f"attempt {retries + 1}/{self.max_retries}, "
                        f"API key status: {'Set' if fetcher_apikey else 'Not Set'}"
                    )
                    #do error handling ie: exponential backoff 
                    retries += 1
                    if retries >= self.max_retries:
                        self.logger.error(f"Max retries reached for PMID {pmid} after rate limit errors")
                        return {"pmid": pmid, "error": "Rate limit exceeded"}
                    
                    wait_time = min(2**retries, 60)
                    await asyncio.sleep(wait_time)

            except (ConnectTimeout, RequestException, ConnectTimeoutError) as e:
                # Handle connection timeout errors
                full_traceback = traceback.format_exc()
                self.logger.warning(f"Connection timeout for PMID {pmid}, attempt {retries + 1}: {str(e)}, Full traceback: {full_traceback}")
                retries += 1
                if retries >= self.max_retries:
                    self.logger.error(f"Max retries reached for PMID {pmid} after timeout errors")
                    return {"pmid": pmid, "error": "Connection timeout"}
                
                
                wait_time = min(2**retries, 60)
                self.logger.info(f"Retrying for PMID {pmid} in {wait_time} seconds")
                await asyncio.sleep(wait_time)

            except EutilsNCBIError as e:
                self.logger.error(f"EutilsNCBIError for PMID {pmid}: {str(e)}")
                self.logger.error(f"Traceback:\n{traceback.format_exc()}")
                retries += 1
                wait_time = min(2**retries, 60)
                self.logger.warning(f"Retrying for PMID {pmid} in {wait_time} seconds")
                await asyncio.sleep(wait_time)
                if retries >= self.max_retries:
                    self.logger.error(f"Max retries reached for PMID {pmid} after EutilsNCBIError")
                    return {"pmid": pmid, "error": "EutilsNCBIError"}
                
            except Exception as e:
                self.logger.error(f"Unexpected error for PMID {pmid}: {str(e)}")
                self.logger.error(f"Traceback:\n{traceback.format_exc()}")
                retries += 1
                if retries >= self.max_retries:
                    self.logger.error(f"Max retries reached for PMID {pmid} after unexpected error")
                    return {"pmid": pmid, "error": "Unexpected error"}
    
    async def get_pubmed_article_details(self, pmid_list: List[str], batch_size: int = 1000): 
        '''
        Async retrieve article details for a list of PMIDs 
        '''
        self.pubmed_retrieval = 'article_details'
        worker_tasks = [
                asyncio.create_task(self._worker_task(client_function = self.async_retrieve_pubmed_article_data)) for _ in range(self.max_concurrent_requests)
            ]
        
        all_results = []
        for i in range(0, len(pmid_list), batch_size):
            batch = pmid_list[i:i + batch_size]
            for pmid in batch:
                await self.task_queue.put(pmid)
        
            task_count = self.task_queue.qsize()
            self.logger.info(f'Starting task queue for PubMed Article Details retrieval. {task_count} tasks to complete for batch: {i} to {i + batch_size}')

        #create a maximum number of workers tasks available to be used 

            complete_tasks = 0 
            batch_results = []
            while complete_tasks < task_count: 
                try: 
                    result = await self.result_queue.get()
                    if result: 
                        batch_results.append(result)
                        complete_tasks += 1
                        self.logger.info(f'Completed {complete_tasks} of {task_count} tasks. Current batch: {i} to {i + batch_size}')

                except Exception as e: 
                    self.logger.error(f'Error processing results: {e}')

            self.logger.info(f'All tasks for current batches arte complete')
            all_results.extend(batch_results)
                    # Log the status of worker tasks
            for idx, task in enumerate(worker_tasks):
                self.logger.info(f'Worker task {idx} status: {task._state}')


        #cancell all worker tasks now that all tasks and batches are complete 
        for task in worker_tasks: 
            task.cancel()

        #wait for all worker tasks to complete cancellation 
        await asyncio.gather(*worker_tasks, return_exceptions=True)

        if len(all_results) > 0: 
            #unpack article details 
            all_results_dct_list = [
                {
                    'pmid': getattr(result, 'pmid', None), 
                    'title': getattr(result, 'title', None), 
                    'abstract': getattr(result, 'abstract', None), 
                    'publication_year': getattr(result, 'year', None)
                } for result in all_results
            ]

            return pd.DataFrame(all_results_dct_list)
        else: 
            return pd.DataFrame()
















    





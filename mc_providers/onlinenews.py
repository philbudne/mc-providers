import datetime as dt
from typing import List, Dict
import dateparser
import logging
import numpy as np
import random
from waybacknews.searchapi import SearchApiClient
from collections import Counter
from .language import stopwords_for_language
from .provider import ContentProvider
from .cache import CachingManager


class OnlineNewsWaybackMachineProvider(ContentProvider):
    """
    All these endpoints accept a `domains: List[str]` keyword arg.
    """

    DEFAULT_COLLECTION = "mediacloud"
    MAX_QUERY_LENGTH = pow(2, 14)
    
    
    def __init__(self):
        super(OnlineNewsWaybackMachineProvider, self).__init__()
        self._client = SearchApiClient(self.DEFAULT_COLLECTION)
        self._logger = logging.getLogger(__name__)

    def everything_query(self) -> str:
        return '*'

    #Chunk'd
    #NB: it looks like the limit keyword here doesn't ever get passed into the query- something's missing here. 
    @CachingManager.cache()
    def sample(self, query: str, start_date: dt.datetime, end_date: dt.datetime, limit: int = 20,
               **kwargs) -> List[Dict]:
        results = []
        for subquery in self._assemble_and_chunk_query_str(query, **kwargs):
            this_results = self._client.sample(subquery, start_date, end_date, **kwargs)
            results.extend(this_results)
        
        if(len(results) > limit):
            results = random.sample(results, limit)
            
        return self._matches_to_rows(results)

    #Chunk'd
    @CachingManager.cache()
    def count(self, query: str, start_date: dt.datetime, end_date: dt.datetime, **kwargs) -> int:
        count = 0
        for subquery in self._assemble_and_chunk_query_str(query, **kwargs):
            count += self._client.count(subquery, start_date, end_date, **kwargs)
        return count

    #Chunk'd
    @CachingManager.cache()
    def count_over_time(self, query: str, start_date: dt.datetime, end_date: dt.datetime, **kwargs) -> Dict:
        
        counter = Counter()
        for subquery in self._assemble_and_chunk_query_str(query, **kwargs):
            results = self._client.count_over_time(subquery, start_date, end_date, **kwargs)
            countable = {i['date']:i['count'] for i in results}
            counter += Counter(countable)
        
        counter_dict = dict(counter)
        results = [{"date":date, "timestamp":date.timestamp(), "count":count} for date, count in counter_dict.items()]
        #Somehow the order of this list gets out of wack. Sorting before returning for the sake of testability
        sorted_results = sorted(results, key = lambda x: x["timestamp"]) 
        return {'counts': sorted_results}

    
    @CachingManager.cache()
    def item(self, item_id: str) -> Dict:
        return self._client.article(item_id)
    
    #Chunk'd
    def all_items(self, query: str, start_date: dt.datetime, end_date: dt.datetime, page_size: int = 1000, **kwargs):
        for subquery in self._assemble_and_chunk_query_str(query, **kwargs):
            for page in self._client.all_articles(subquery, start_date, end_date, **kwargs):
                yield self._matches_to_rows(page)

    #Chunk'd
    @CachingManager.cache()
    def words(self, query: str, start_date: dt.datetime, end_date: dt.datetime, limit: int = 100,
              **kwargs) -> List[Dict]:
        
        chunked_queries = self._assemble_and_chunk_query_str(query, **kwargs)
        
        
        # first figure out the dominant languages, so we can remove appropriate stopwords.
        # This method does chunking for you, so just pass the query 
        top_languages = self.languages(query, start_date, end_date, limit=100, **kwargs) 
        
        
        represented_languages = [i['language'] for i in top_languages if i['ratio'] > 0.1]
        stopwords = []
        for lang in represented_languages:
            try:
                stopwords += stopwords_for_language(lang)
            except RuntimeError:
                pass  # not stopwords for language, just let them all pass through
            
        # for now just return top terms in article titles
        sample_size = 5000
        
        #An accumulator for the subqueries
        results_counter = Counter({})
        for subquery in chunked_queries:
            this_results = self._client.terms(subquery, start_date, end_date,
                                     self._client.TERM_FIELD_TITLE, self._client.TERM_AGGREGATION_TOP)
            
            if "detail" not in this_results:
                results_counter += Counter(this_results)
        
        results = dict(results_counter)
            
        # and clean up results to return
        top_terms = [dict(term=t.lower(), count=c, ratio=c/sample_size) for t, c in results.items()
                     if t.lower() not in stopwords]
        top_terms = sorted(top_terms, key=lambda x:x["count"], reverse=True)
        return top_terms

    #Chunk'd
    @CachingManager.cache()
    def languages(self, query: str, start_date: dt.datetime, end_date: dt.datetime, limit: int = 10,
                  **kwargs) -> List[Dict]:
        
        matching_count = self.count(query, start_date, end_date, **kwargs)
        
        
        results_counter = Counter({})
        for subquery in self._assemble_and_chunk_query_str(query, **kwargs) :   
            this_languages = self._client.top_languages(subquery, start_date, end_date, **kwargs)
            countable = {item["name"]:item["value"] for item in this_languages}
            results_counter += Counter(countable)
            #top_languages.extend(this_languages)
        
        all_results = dict(results_counter)
        
        top_languages = [{'language':name, 'value':value} for name, value in all_results.items()]
        
        for item in top_languages:
            item['ratio'] = item['value'] / matching_count
        
        #Sort by count, then alphabetically
        top_languages = sorted(top_languages, key=lambda x:x['value'], reverse=True)
        return top_languages[:limit]

    #Chunk'd
    def sources(self, query: str, start_date: dt.datetime, end_date: dt.datetime, limit: int = 100,
                **kwargs) -> List[Dict]:
        
        #all_results = []
        
        results_counter = Counter({})
        for subquery in self._assemble_and_chunk_query_str(query, **kwargs):
            results = self._client.top_sources(subquery, start_date, end_date)
            countable = {source['name']:source['value'] for source in results}
            results_counter += Counter(countable)
            #all_results.extend(results)
        
        all_results = dict(results_counter)
        cleaned_sources = [{"source":source , "count":count} for source, count in all_results.items()]
        cleaned_sources = sorted(cleaned_sources, key= lambda x: x['count'], reverse=True)
        return cleaned_sources

    @classmethod
    def _assemble_and_chunk_query_str(cls, base_query: str, **kwargs) -> List:
        """
        If a query string is too long, we can attempt to run it anyway by splitting the domain substring (which is guaranteed 
        too be only a sequence of ANDs) into parts, to produce multiple smaller queries which are collectively equivalent 
        to the original. 
        """
        domains = kwargs.get('domains', [])
        
        
        
        if len(base_query) > cls.MAX_QUERY_LENGTH:
            ##of course there still is the possibility that the base query is too large, which 
            #cannot be fixed by this method
            raise RuntimeError(f"Base Query cannot exceed {cls.MAX_QUERY_LENGTH} characters")
        
        queries = [cls._assembled_query_str(base_query, domains=domains)]
        queries_too_big = any([len(q_) > cls.MAX_QUERY_LENGTH for q_ in queries])
        domain_divisor = 2
        
        if queries_too_big:
            while queries_too_big:
                chunked_domains = np.array_split(domains, domain_divisor)
                queries = [cls._assembled_query_str(base_query, domains=dom) for dom in chunked_domains]
                queries_too_big = any([len(q_) > cls.MAX_QUERY_LENGTH for q_ in queries])
                domain_divisor *= 2
            
        return queries
    
    @classmethod
    def _assembled_query_str(cls, query: str, **kwargs) -> str:
        domains = kwargs.get('domains', [])
        # need to put all those filters in single query string
        q = query
        if len(domains) > 0:
            q += " AND (domain:({}))".format(" OR ".join(domains))
        return q

    @classmethod
    def _matches_to_rows(cls, matches: List) -> List:
        return [OnlineNewsWaybackMachineProvider._match_to_row(m) for m in matches]

    @classmethod
    def _match_to_row(cls, match: Dict) -> Dict:
        return {
            'media_name': match['domain'],
            'media_url': "http://"+match['domain'],
            'id': match['archive_playback_url'].split("/")[4],  # grabs a unique id off archive.org URL
            'title': match['title'],
            'publish_date': dateparser.parse(match['publication_date']),
            'url': match['url'],
            'language': match['language'],
            'archived_url': match['archive_playback_url'],
            'article_url': match['article_url'],
        }

    def __repr__(self):
        # important to keep this unique among platforms so that the caching works right
        return "OnlineNewsWaybackMachineProvider"

import datetime as dt
from typing import List, Dict, Optional
import dateparser
import hashlib
import logging
import numpy as np
import random
from waybacknews.searchapi import SearchApiClient
from mediacloud_legacy.api import MediaCloud
import mcmetadata
from collections import Counter

from .errors import deprecated
from .language import stopwords_for_language
from .provider import ContentProvider
from .cache import CachingManager


class OnlineNewsAbstractProvider(ContentProvider):
    """
    All these endpoints accept a `domains: List[str]` keyword arg.
    """
    
    MAX_QUERY_LENGTH = pow(2, 14)

    def __init__(self, base_url: Optional[str]):
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self._base_url = base_url
        self._client = self.get_client()

    def get_client(self):
        raise NotImplementedError("Abstract provider class should not be implimented directly")

    @classmethod
    def domain_search_string(cls):
        raise NotImplementedError("Abstract provider class should not be implimented directly")

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

    def paged_items(self, query: str, start_date: dt.datetime, end_date: dt.datetime, page_size: int = 1000, **kwargs)\
            -> tuple[List[Dict], str] :
        '''
        Note - this is not chunk'd so you can't run giant queries page by page... use `all_items` instead.
        This kwargs should include `pagination_token`, which will get relayed in to the api client and fetch
        the right page of results.
        '''
        page, pagination_token = self._client.paged_articles(query, start_date, end_date, **kwargs)
        return self._matches_to_rows(page), pagination_token

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
    def _assemble_and_chunk_query_str(cls, base_query: str, **kwargs) :
        """
        If a query string is too long, we can attempt to run it anyway by splitting the domain substring (which is guaranteed 
        too be only a sequence of ANDs) into parts, to produce multiple smaller queries which are collectively equivalent 
        to the original. 
        
        
        Because we have this chunking thing implimented, and the filter behavior never interacts with the domain search behavior,
        we can just put the two different search fields into two different sets of behavior at the top. Theres obvi room to optimize,
        but this gets the done job.
        """
        
        
        domains = kwargs.get('domains', [])

        filters = kwargs.get('filters', [])
        
        if len(base_query) > cls.MAX_QUERY_LENGTH:
            ##of course there still is the possibility that the base query is too large, which 
            #cannot be fixed by this method
            raise RuntimeError(f"Base Query cannot exceed {cls.MAX_QUERY_LENGTH} characters")
        

        #Get Domain Queries
        domain_queries = []
        if len(domains) > 0:
            domain_queries = [cls._assembled_query_str(base_query, domains=domains)]
            domain_queries_too_big = any([len(q_) > cls.MAX_QUERY_LENGTH for q_ in domain_queries])

            domain_divisor = 2

            if domain_queries_too_big:
                while domain_queries_too_big:
                    chunked_domains = np.array_split(domains, domain_divisor)
                    domain_queries = [cls._assembled_query_str(base_query, domains=dom) for dom in chunked_domains]
                    domain_queries_too_big = any([len(q_) > cls.MAX_QUERY_LENGTH for q_ in domain_queries])
                    domain_divisor *= 2
                
        #Then Get Filter Queries
        filter_queries = []
        if len(filters) > 0:
            filter_queries = [cls._assembled_query_str(base_query, filters=filters)]
            filter_queries_too_big = any([len(q_) > cls.MAX_QUERY_LENGTH for q_ in filter_queries])

            filter_divisor = 2
            if filter_queries_too_big:
                while filter_queries_too_big:
                    chunked_filters = np.array_split(filters, filter_divisor)
                    filter_queries = [cls._assembled_query_str(base_query, filters=filt) for filt in chunked_filters]
                    filter_queries_too_big = any([len(q_) > cls.MAX_QUERY_LENGTH for q_ in filter_queries])
                    filter_divisor *= 2
            
        #There's a (probably not uncommon) edge case where we're searching against no collections at all,
        #so just do it manually here. 
        if len(domain_queries) == 0 and len(filter_queries) == 0:
            queries = [cls._assembled_query_str(base_query)]
        
        else:
            queries = domain_queries + filter_queries
        
        return queries
    
    @classmethod
    def _assembled_query_str(cls, query: str, **kwargs) -> str:
        
        domains = kwargs.get('domains', [])
        domain_clause = ""
        
        if len(domains) > 0:
            domain_string = " OR ".join(domains)
            domain_clause = f"{cls.domain_search_string()}:({domain_string})"
            
        filters = kwargs.get('filters', [])
        filter_clause = ""
        
        if len(filters) > 0:
            filter_clause = " OR ".join(filters)
        
        # need to put all those filters in single query string
        q = query
        
        if( (len(domains) > 0) and (len(filters) > 0) ):
            q += f" AND (({domain_clause}) OR ({filter_clause}))"
        
        elif len(domains) > 0:
            q += f" AND ({domain_clause})"
        
        elif len(filters) > 0:
            q += f" AND ({filter_clause})"
        return q


    @classmethod
    def _matches_to_rows(cls, matches: List) -> List:
        raise NotImplementedError()

    @classmethod
    def _match_to_row(cls, match: Dict) -> Dict:
        raise NotImplementedError()

    def __repr__(self):
        # important to keep this unique among platforms so that the caching works right
        return "OnlineNewsAbstractProvider"


class OnlineNewsWaybackMachineProvider(OnlineNewsAbstractProvider):
    """
    All these endpoints accept a `domains: List[str]` keyword arg.
    """

    def __init__(self, base_url: Optional[str] = None):
        super().__init__(base_url)

    def get_client(self):
        return SearchApiClient("mediacloud", self._base_url)

    @classmethod
    def domain_search_string(cls):
        return "domain"

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


class OnlineNewsMediaCloudProvider(OnlineNewsAbstractProvider):
    """
    Provider interface to access new mediacloud-news-search archive. 
    All these endpoints accept a `domains: List[str]` keyword arg.
    For now we just use the wayback client, but eventually we'll have our own fork
    """
    DEFAULT_API_BASE_URL = "https://news-search-api.mediacloud.org/v1/"
    DEFAULT_COLLECTION = "mediacloud_search_text_*"

    def __init__(self, base_url=Optional[str]):
        super().__init__(base_url)

    def get_client(self):
        #This seems cleaner than having a whole parallel client library
        api_client = SearchApiClient(collection=self.DEFAULT_COLLECTION, api_base_url=self._base_url)
        return api_client

    @classmethod
    def domain_search_string(cls):
        return "canonical_domain"

    @classmethod
    def _matches_to_rows(cls, matches: List) -> List:
        return [OnlineNewsMediaCloudProvider._match_to_row(m) for m in matches]

    @classmethod
    def _match_to_row(cls, match: Dict) -> Dict:
        story_info = {
            'media_name': match['canonical_domain'],
            'media_url': match['canonical_domain'],
            'id': hashlib.md5(match['normalized_url'].encode('utf8')).hexdigest(),
            'title': match['article_title'],
            'publish_date': dateparser.parse(match['publication_date']),
            'url': match['url'],
            'language': match['language'],
            'indexed_date': dateparser.parse(match['publication_date']),
        }
        if 'text_content' in match:
            story_info['text'] = match['text_content']
        return story_info

    def __repr__(self):
        return "OnlineNewsMediaCloudProvider"


@deprecated
class OnlineNewsMediaCloudLegacyProvider(ContentProvider):
    """
    Media Cloud is a platform for analyzing news media. It is a free, open-source platform for
    collecting, analyzing, and visualizing news media content. It is a project of the MIT Media
    Lab and the Berkman Klein Center for Internet & Society at Harvard University.

    This provider uses the Media Cloud API to search for articles. It is a wrapper around the
    Media Cloud Python client library, which is available here:
    """

    def __init__(self, api_key):
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self._api_key = api_key
        self._mc_client = MediaCloud(api_key)
        self._mc_client.TIMEOUT_SECS = 300  # give backend 5 mins to responsd

    def everything_query(self) -> str:
        return '*'

    @CachingManager.cache()
    def sample(self, query: str, start_date: dt.datetime, end_date: dt.datetime, limit: int = 20,
               **kwargs) -> List[Dict]:
        """
        Return a list of stories matching the query.
        :param query:
        :param start_date:
        :param end_date:
        :param limit:
        :param kwargs: sources and collections lists
        :return:
        """
        q, fq = self._format_query(query, start_date, end_date, **kwargs)
        story_list = self._mc_client.storyList(q, fq, rows=limit)
        return self._matches_to_rows(story_list)

    @CachingManager.cache()
    def count(self, query: str, start_date: dt.datetime, end_date: dt.datetime, **kwargs) -> int:
        """
        Count how many verified tweets match the query.
        :param query:
        :param start_date:
        :param end_date:
        :param kwargs: sources and collections lists
        :return:
        """
        q, fq = self._format_query(query, start_date, end_date, **kwargs)
        story_count_result = self._mc_client.storyCount(q, fq)
        return story_count_result['count']

    @CachingManager.cache()
    def count_over_time(self, query: str, start_date: dt.datetime, end_date: dt.datetime, **kwargs) -> List[Dict]:
        """
        How many verified tweets over time match the query.
        :param query:
        :param start_date:
        :param end_date:
        :param kwargs: sources and collections lists
        :return:
        """
        q, fq = self._format_query(query, start_date, end_date, **kwargs)
        story_count_result = self._mc_client.storyCount(q, fq, split=True)
        return story_count_result

    @CachingManager.cache()
    def item(self, item_id: str) -> Dict:
        story = self._mc_client.story(item_id)
        return self._match_to_row(story)

    def all_items(self, query: str, start_date: dt.datetime, end_date: dt.datetime, page_size: int = 1000, **kwargs):
        q, fq = self._format_query(query, start_date, end_date, **kwargs)
        last_id = 0
        more_stories = True
        stories = []
        while more_stories:
            page = self._matching_page(q, fq, last_processed_stories_id=last_id, page_size=page_size,
                                       sort='processed_stories_id')
            yield page
            if len(page) == 0:
                more_stories = False
            else:
                stories += page
                last_id = page[-1]['processed_stories_id']

    @CachingManager.cache()
    def _matching_page(self, q: str, fq: str, last_processed_stories_id: str, page_size: int, sort: str):
        page = self._mc_client.storyList(q, fq, last_processed_stories_id=last_processed_stories_id, rows=page_size,
                                         sort=sort)
        return page

    @CachingManager.cache()
    def words(self, query: str, start_date: dt.datetime, end_date: dt.datetime, limit: int = 100,
              **kwargs) -> List[Dict]:
        """
        Get the top words based on a sample
        :param query:
        :param start_date:
        :param end_date:
        :param limit:
        :param kwargs: sources and collections lists
        :return:
        """
        q, fq = self._format_query(query, start_date, end_date, **kwargs)
        top_words = self._mc_client.wordCount(q, fq)[:limit]
        return top_words

    @CachingManager.cache()
    def tags(self, query: str, start_date: dt.datetime, end_date: dt.datetime, **kwargs) -> List[Dict]:
        q, fq = self._format_query(query, start_date, end_date, **kwargs)
        tags_sets_id = kwargs.get('tags_sets_id', None)
        sample_size = kwargs.get('sample_size', None)
        top_tags = self._mc_client.storyTagCount(q, fq, tag_sets_id=tags_sets_id, limit=sample_size, http_method='POST')
        return top_tags

    @classmethod
    def _format_query(cls, query: str, start_date: dt.datetime, end_date: dt.datetime,
                      **kwargs) -> (str, str):
        """
        Take all the query params and return q and fq suitable for a media cloud solr-syntax query
        :param query:
        :param start_date:
        :param end_date:
        :param kwargs: sources and collections
        :return:
        """
        media_ids = kwargs['sources'] if 'sources' in kwargs else []
        tags_ids = kwargs['collections'] if 'collections' in kwargs else []
        q = cls._query_from_parts(query, media_ids, tags_ids)
        fq = MediaCloud.dates_as_query_clause(start_date, end_date)
        return q, fq

    @classmethod
    def _query_from_parts(cls, query: str, media_ids: List[int], tag_ids: List[int]) -> str:
        query = '({})'.format(query)
        if len(media_ids) > 0 or (len(tag_ids) > 0):
            clauses = []
            # add in the media sources they specified
            if len(media_ids) > 0:  # this format is a string of media_ids
                query_clause = "media_id:({})".format(" ".join([str(m) for m in media_ids]))
                clauses.append(query_clause)
            # add in the collections they specified
            if len(tag_ids) > 0:  # this format is a string of tags_id_medias
                query_clause = "tags_id_media:({})".format(" ".join([str(m) for m in tag_ids]))
                clauses.append(query_clause)
            # now add in any addition media query clauses (get OR'd together)
            if len(clauses) > 0:
                query += " AND ({})".format(" OR ".join(clauses))
        return query

    def __repr__(self):
        # important to keep this unique among platforms so that the caching works right
        return "OnlineNewsMediaCloudLegacyProvider"

    @classmethod
    def _matches_to_rows(cls, matches: List) -> List:
        return [OnlineNewsMediaCloudProvider._match_to_row(m) for m in matches]

    @classmethod
    def _match_to_row(cls, match: Dict) -> Dict:
        return {
            'media_name': mcmetadata.urls.canonical_domain(match['url']),
            'media_url': match['media_url'],
            'id': match['stories_id'],
            'title': match['title'],
            'publish_date': dateparser.parse(match['publish_date']),
            'url': match['url'],
            'language': match['language'],
        }

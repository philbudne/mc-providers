import datetime as dt
import logging
import random
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional

# PyPI
import ciso8601
import dateparser
import numpy as np              # for chunking
from waybacknews.searchapi import SearchApiClient

from .language import stopwords_for_language
from .provider import ContentProvider
from .cache import CachingManager
from .mediacloud import MCSearchApiClient

# don't need a logger per Provider instance
logger = logging.getLogger(__name__)

class OnlineNewsAbstractProvider(ContentProvider):
    """
    All these endpoints accept a `domains: List[str]` keyword arg.
    """
    
    MAX_QUERY_LENGTH = pow(2, 14)
    TOP_WORDS_THRESHOLD = 0.1
    DEBUG_WORDS = 0

    def __init__(self, base_url: Optional[str], timeout: Optional[int] = None, caching: bool = True):
        super().__init__(caching)
        self._base_url = base_url
        self._timeout = timeout
        self._client = self.get_client()

    def get_client(self):
        raise NotImplementedError("Abstract provider class should not be implemented directly")

    @classmethod
    def domain_search_string(cls):
        raise NotImplementedError("Abstract provider class should not be implemented directly")

    def everything_query(self) -> str:
        return '*'

    # Chunk'd
    # NB: it looks like the limit keyword here doesn't ever get passed into the query - something's missing here.
    def sample(self, query: str, start_date: dt.datetime, end_date: dt.datetime, limit: int = 20,
               **kwargs) -> List[Dict]:
        results = []
        for subquery in self._assemble_and_chunk_query_str(query, **kwargs):
            this_results = self._client.sample(subquery, start_date, end_date, **kwargs)
            results.extend(this_results)
        
        if len(results) > limit:
            results = random.sample(results, limit)
            
        return self._matches_to_rows(results)

    # Chunk'd
    def count(self, query: str, start_date: dt.datetime, end_date: dt.datetime, **kwargs) -> int:
        count = 0
        for subquery in self._assemble_and_chunk_query_str(query, **kwargs):
            count += self._client.count(subquery, start_date, end_date, **kwargs)
        return count

    # Chunk'd
    def count_over_time(self, query: str, start_date: dt.datetime, end_date: dt.datetime, **kwargs) -> Dict:
        counter: Counter = Counter()
        for subquery in self._assemble_and_chunk_query_str(query, **kwargs):
            results = self._client.count_over_time(subquery, start_date, end_date, **kwargs)
            countable = {i['date']: i['count'] for i in results}
            counter += Counter(countable)
        
        counter_dict = dict(counter)
        results = [{"date": date, "timestamp": date.timestamp(), "count": count} for date, count in counter_dict.items()]
        # Somehow the order of this list gets out of wack. Sorting before returning for the sake of testability
        sorted_results = sorted(results, key=lambda x: x["timestamp"])
        return {'counts': sorted_results}

    
    @CachingManager.cache()
    def item(self, item_id: str) -> Dict:
        one_item = self._client.article(item_id)
        return self._match_to_row(one_item)

    
    # Chunk'd
    def all_items(self, query: str, start_date: dt.datetime, end_date: dt.datetime, page_size: int = 1000, **kwargs):
        for subquery in self._assemble_and_chunk_query_str(query, **kwargs):
            for page in self._client.all_articles(subquery, start_date, end_date, **kwargs):
                yield self._matches_to_rows(page)

    def paged_items(self, query: str, start_date: dt.datetime, end_date: dt.datetime, page_size: int = 1000, **kwargs)\
            -> tuple[List[Dict], Optional[str]] :
        """
        Note - this is not chunk'd so you can't run giant queries page by page... use `all_items` instead.
        This kwargs should include `pagination_token`, which will get relayed in to the api client and fetch
        the right page of results.
        """
        updated_kwargs = {**kwargs, 'chunk': False}
        query = self._assemble_and_chunk_query_str(query, **updated_kwargs)[0]
        page, pagination_token = self._client.paged_articles(query, start_date, end_date, **kwargs)
        return self._matches_to_rows(page), pagination_token

    def _word_counts(self, query: str, start_date: dt.datetime, end_date: dt.datetime, **kwargs) -> Mapping:
        logger.debug("AP._word_counts %s %s %s %r", query, start_date, end_date, kwargs)

        chunked_queries = self._assemble_and_chunk_query_str(query, **kwargs)

        # An accumulator for the subqueries
        totals: Counter = Counter()

        for subquery in chunked_queries:
            logger.debug("AP.words subquery %s %s %s", subquery, start_date, end_date)

            this_results = self._client.terms(subquery, start_date, end_date,
                                              self._client.TERM_FIELD_TITLE,
                                              self._client.TERM_AGGREGATION_TOP)
            if "detail" not in this_results:
                if self.DEBUG_WORDS:
                    logger.debug("AP.words results %s", this_results)
                totals += Counter(this_results)
        return totals

    # Chunk'd
    @CachingManager.cache()
    def words(self, query: str, start_date: dt.datetime, end_date: dt.datetime, limit: int = 100,
              **kwargs) -> List[Dict]:
        logger.debug("AP.words %s %s %s %r", query, start_date, end_date, kwargs)

        # first figure out the dominant languages, so we can remove appropriate stopwords.
        # This method does chunking for you, so just pass the query 
        top_languages = self.languages(query, start_date, end_date, limit=100, **kwargs) 

        stopwords = set()
        for lang_ent in top_languages:
            lang = lang_ent['language']
            ratio = lang_ent['ratio']

            # skips stopwords for less represented languages,
            # but doesn't eliminate them from sampling!!
            if self.DEBUG_WORDS:
                logger.debug("lang %s ratio %.6f", lang, ratio)
            if ratio < self.TOP_WORDS_THRESHOLD:
                continue
            if self.DEBUG_WORDS:
                logger.debug("getting stop words for %s", lang)
            try:
                if len(lang) == 2:
                    for word in stopwords_for_language(lang):
                        stopwords.add(word)
            except KeyboardInterrupt:
                raise
            except RuntimeError as e:
                # explicitly raises RuntimeError if len(lang) != 2!!
                logger.warning("error getting stop words for %s: %e", lang, e)

        # for now just return top terms in article titles
        sample_size = 5000      # PB: only used for scaling???
        
        results_counter = self._word_counts(query, start_date, end_date, **kwargs)
        if self.DEBUG_WORDS > 1:
            logger.debug("AP.words total %r", results_counter)
            for t, c in results_counter.items():
                logger.debug("%s %d %.6f %s", t, c, c/sample_size, t.lower() not in stopwords)

        # and clean up results to return
        top_terms = [dict(term=t.lower(), count=c, ratio=c/sample_size) for t, c in results_counter.items()
                     if t.lower() not in stopwords]
        top_terms = sorted(top_terms, key=lambda x:x["count"], reverse=True)
        return top_terms

    # Chunk'd
    def languages(self, query: str, start_date: dt.datetime, end_date: dt.datetime, limit: int = 10,
                  **kwargs) -> List[Dict]:
        
        matching_count = self.count(query, start_date, end_date, **kwargs)

        results_counter: Counter = Counter({})
        for subquery in self._assemble_and_chunk_query_str(query, **kwargs) :   
            this_languages = self._client.top_languages(subquery, start_date, end_date, **kwargs)
            countable = {item["name"]: item["value"] for item in this_languages}
            results_counter += Counter(countable)
            # top_languages.extend(this_languages)
        
        all_results = dict(results_counter)
        
        top_languages = [{'language': name, 'value': value, 'ratio': 0.0} for name, value in all_results.items()]
        
        for item in top_languages:
            item['ratio'] = item['value'] / matching_count
        
        # Sort by count, then alphabetically
        top_languages = sorted(top_languages, key=lambda x: x['value'], reverse=True)
        return top_languages[:limit]

    # Chunk'd
    def sources(self, query: str, start_date: dt.datetime, end_date: dt.datetime, limit: int = 100,
                **kwargs) -> List[Dict]:
        
        # all_results = []
        
        results_counter: Counter = Counter({})
        for subquery in self._assemble_and_chunk_query_str(query, **kwargs):
            results = self._client.top_sources(subquery, start_date, end_date)
            countable = {source['name']: source['value'] for source in results}
            results_counter += Counter(countable)
            # all_results.extend(results)
        
        all_results = dict(results_counter)
        cleaned_sources = [{"source": source , "count": count} for source, count in all_results.items()]
        cleaned_sources = sorted(cleaned_sources, key=lambda x: x['count'], reverse=True)
        return cleaned_sources

    @classmethod
    def _assemble_and_chunk_query_str(cls, base_query: str, chunk: bool = True, **kwargs):
        """
        If a query string is too long, we can attempt to run it anyway by splitting the domain substring (which is
        guaranteed to be only a sequence of ANDs) into parts, to produce multiple smaller queries which are collectively
        equivalent to the original.

        Because we have this chunking thing implemented, and the filter behavior never interacts with the domain search
        behavior, we can just put the two different search fields into two different sets of behavior at the top.
        There's obvious room to optimize, but this gets the done job.
        """
        logger.debug("AP._assemble_and_chunk_query_str %s %s %r", base_query, chunk, kwargs)
        domains = kwargs.get('domains', [])

        filters = kwargs.get('filters', [])

        if chunk and (len(base_query) > cls.MAX_QUERY_LENGTH):
            # of course there still is the possibility that the base query is too large, which
            # cannot be fixed by this method
            raise RuntimeError(f"Base Query cannot exceed {cls.MAX_QUERY_LENGTH} characters")

        # Get Domain Queries
        domain_queries = []
        if len(domains) > 0:
            domain_queries = [cls._assembled_query_str(base_query, domains=domains)]
            domain_queries_too_big = any([len(q_) > cls.MAX_QUERY_LENGTH for q_ in domain_queries])

            domain_divisor = 2

            if chunk and domain_queries_too_big:
                while domain_queries_too_big:
                    chunked_domains = np.array_split(domains, domain_divisor)
                    domain_queries = [cls._assembled_query_str(base_query, domains=dom) for dom in chunked_domains]
                    domain_queries_too_big = any([len(q_) > cls.MAX_QUERY_LENGTH for q_ in domain_queries])
                    domain_divisor *= 2
                
        # Then Get Filter Queries
        filter_queries = []
        if len(filters) > 0:
            filter_queries = [cls._assembled_query_str(base_query, filters=filters)]
            filter_queries_too_big = any([len(q_) > cls.MAX_QUERY_LENGTH for q_ in filter_queries])

            filter_divisor = 2
            if chunk and filter_queries_too_big:
                while filter_queries_too_big:
                    chunked_filters = np.array_split(filters, filter_divisor)
                    filter_queries = [cls._assembled_query_str(base_query, filters=filt) for filt in chunked_filters]
                    filter_queries_too_big = any([len(q_) > cls.MAX_QUERY_LENGTH for q_ in filter_queries])
                    filter_divisor *= 2
            
        # There's a (probably not uncommon) edge case where we're searching against no collections at all,
        # so just do it manually here.
        if len(domain_queries) == 0 and len(filter_queries) == 0:
            queries = [cls._assembled_query_str(base_query)]
        
        else:
            queries = domain_queries + filter_queries
        
        return queries

    @staticmethod
    def _sanitize_query(fstr: str) -> str:
        """
        noop: (/ quoting done in mediacloud.py sanitize_query)
        """
        return fstr

    @classmethod
    def _assembled_query_str(cls, query: str, **kwargs) -> str:
        logger.debug("_assembled_query_str IN: %s %r", query, kwargs)
        # filter and/or domain clauses (selectors to be OR'ed together)
        selector_clauses = []

        domains = kwargs.get('domains', [])
        if len(domains) > 0:
            domain_strings = " OR ".join(domains)
            selector_clauses.append(f"{cls.domain_search_string()}:({domain_strings})")
            
        # put all filters in single query string
        # they're additive, so "selectors" might have been a clearer name?
        filters = kwargs.get('filters', [])
        if len(filters) > 0:
            for filter in filters:
                f = cls._sanitize_query(filter)
                if "AND" in f:
                    # parenthesize if any chance it has a grabby AND.
                    selector_clauses.append(f"({f})")
                else:
                    selector_clauses.append(f)

        # PB: experimental to get web-search out of query formatting biz
        # (to be able to put selectors in a filter context)
        url_search_strings: list[tuple[str,str]] = kwargs.get('url_search_strings', [])
        if url_search_strings:
            for cdom, sstr in url_search_strings:
                usf = rf"(canonical_domain:{cdom} AND url:(http\://{sstr} OR https\://{sstr}))"
                # NOTE: unclear if canonical_domain check of any benefit.  simplified:
                #usf = rf"url:(http\://{sstr} OR https\://{sstr})"
                selector_clauses.append(cls._sanitize_query(usf))

        if len(selector_clauses) > 0:
            # Add parens around user query to protect against against grabby ANDs
            # (individual selector_clauses have been parenthesized if needed)
            clauses_string = " OR ".join(selector_clauses)
            q = f"({query}) AND ({clauses_string})"
        else:
            q = query
        logger.debug("_assembled_query_str OUT: %s", q)
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

    def __init__(self, base_url: Optional[str] = None, timeout: Optional[int] = None, caching: bool = True):
        super().__init__(base_url, timeout, caching)  # will call get_client

    def get_client(self):
        client = SearchApiClient("mediacloud", self._base_url)
        if self._timeout:
            client.TIMEOUT_SECS = self._timeout
        return client

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
    """
    
    DEFAULT_COLLECTION = "mc_search-*"

    def __init__(self, base_url=Optional[str], timeout: Optional[int] = None, caching: bool = True):
        super().__init__(base_url, timeout, caching)

    def get_client(self):
        api_client = MCSearchApiClient(collection=self.DEFAULT_COLLECTION, api_base_url=self._base_url)
        if self._timeout:
            api_client.TIMEOUT_SECS = self._timeout
        return api_client

    @classmethod
    def domain_search_string(cls):
        return "canonical_domain"

    @classmethod
    def _matches_to_rows(cls, matches: List) -> List:
        return [cls._match_to_row(m) for m in matches]

    @staticmethod
    def _match_to_row(match: Dict) -> Dict:
        story_info = {
            'id': match['id'],
            'media_name': match['canonical_domain'],
            'media_url': match['canonical_domain'],
            'title': match['article_title'],
            'publish_date': dt.date.fromisoformat(match['publication_date']),
            'url': match['url'],
            'language': match['language'],
            'indexed_date': ciso8601.parse_datetime(match['indexed_date']+"Z"),
        }
        if 'text_content' in match:
            story_info['text'] = match['text_content']
        return story_info

    def __repr__(self):
        return "OnlineNewsMediaCloudProvider"

    def _is_no_results(self, results) -> bool:
        """
        used to test _overview_query results
        """
        return self._client._is_no_results(results)

    def count(self, query: str, start_date: dt.datetime, end_date: dt.datetime, **kwargs) -> int:
        logger.debug("MC.count %s %s %s %r", query, start_date, end_date, kwargs)
        # no chunking on MC
        results = self._overview_query(query, start_date, end_date, **kwargs)
        if self._is_no_results(results):
            logger.debug("MC.count: no results")
            return 0
        count = results['total']
        logger.debug("MC.count: %s", count)
        return count

    def count_over_time(self, query: str, start_date: dt.datetime, end_date: dt.datetime, **kwargs) -> Dict:
        logger.debug("MC.count_over_time %s %s %s %r", query, start_date, end_date, kwargs)
        results = self._overview_query(query, start_date, end_date, **kwargs)
        to_return: List[Dict] = []
        if not self._is_no_results(results):
            data = results['dailycounts']
            to_return = []
            # transform to list of dicts for easier use: process in sorted order
            for day_date in sorted(data):  # date is in 'YYYY-MM-DD' format
                dt = ciso8601.parse_datetime(day_date) # PB: is datetime!!
                to_return.append({
                    'date': dt.date(), # PB: was returning datetime!
                    'timestamp': dt.timestamp(), # PB: conversion may be to local time!!
                    'count': data[day_date]
                })
        logger.debug("MC.count_over_time %d items", len(to_return))
        return {'counts': to_return}

    # NB: limit argument ignored, but included to keep mypy quiet
    def sample(self, query: str, start_date: dt.datetime, end_date: dt.datetime, limit: int = 20, **kwargs) -> List[Dict]:
        logger.debug("MC.sample %s %s %s %r", query, start_date, end_date, kwargs)
        results = self._overview_query(query, start_date, end_date, **kwargs)
        if self._is_no_results(results):
            rows = []
        else:
            rows = self._matches_to_rows(results['matches'])
        logger.debug("MC.sample: %d rows", len(rows))
        return rows

    def languages(self, query: str, start_date: dt.datetime, end_date: dt.datetime, limit: int = 10,
                  **kwargs) -> List[Dict]:
        logger.debug("MC.languages %s %s %s %r", query, start_date, end_date, kwargs)
        results = self._overview_query(query, start_date, end_date, **kwargs)
        if self._is_no_results(results):
            return []
        top_languages = [{'language': name, 'value': value, 'ratio': 0.0}
                         for name, value in results['toplangs'].items()]
        logger.debug("MC.languages: _overview returned %d items", len(top_languages))

        # now normalize
        matching_count = self.count(query, start_date, end_date, **kwargs)
        for item in top_languages:
            item['ratio'] = item['value'] / matching_count
        # Sort by count
        top_languages = sorted(top_languages, key=lambda x: x['value'], reverse=True)
        items = top_languages[:limit]
        logger.debug("MC.languages: returning %d items", len(items))
        return items

    def sources(self, query: str, start_date: dt.datetime, end_date: dt.datetime, limit: int = 100,
                **kwargs) -> List[Dict]:
        logger.debug("MC.sources %s %s %s %r", query, start_date, end_date, kwargs)
        results = self._overview_query(query, start_date, end_date, **kwargs)
        if self._is_no_results(results):
            items = []
        else:
            cleaned_sources = [{"source": source, "count": count} for source, count in results['topdomains'].items()]
            items = sorted(cleaned_sources, key=lambda x: x['count'], reverse=True)
        logger.debug("MC.sources: %d items", len(items))
        return items

    @CachingManager.cache('overview')
    def _overview_query(self, query: str, start_date: dt.datetime, end_date: dt.datetime, **kwargs) -> Dict:
        logger.debug("MC._overview %s %s %s %r", query, start_date, end_date, kwargs)

        # no chunking on MC
        q = self._assembled_query_str(query, **kwargs)

        return self._client._overview_query(q, start_date, end_date, **kwargs)

    @classmethod
    def _assemble_and_chunk_query_str(cls, base_query: str, chunk: bool = True, **kwargs):
        """
        PB: added for words
        """
        logger.debug("MC._assemble_and_chunk_query_str %s %s %r", base_query, chunk, kwargs)
        return [cls._assembled_query_str(base_query, **kwargs)]

################################################################
# code dragged up from mediacloud.py and news-search-api.py
#
import base64
import time

import elasticsearch
import mcmetadata.urls as urls
from elasticsearch_dsl import Search
from elasticsearch_dsl.aggs import RareTerms, Sampler, SignificantTerms, Terms
from elasticsearch_dsl.query import Match, QueryString
#from elasticsearch_dsl.types import FieldSort, SortOptions # not in 8.15 (yet?)
from elasticsearch_dsl.utils import AttrDict

# These are all the characters used in elastic search queries, so they should NOT be included in your search str
_ALL_RESERVED_CHARS = set(r'+\-!():^[]"{}~*?|&/')
# However, most query strings are using these characters on purpose, so let's only automatically escape some of them
_RARE_RESERVED_CHARS = set('/')

def _get(ad: AttrDict, key: str, default: Any = None) -> Any:
    """
    AttrDict doesn't have a "get" method; hide that here make it was
    to switch back to _exec returning res.to_dict()
    """
    return getattr(ad, key, default)

def _sanitize_es_query(query: str, all: bool = False) -> str:
    """
    from mc-providers/mc_providers/mediacloud.py

    Make sure we properly escape any reserved characters in an elastic search query
    @see https://www.elastic.co/guide/en/elasticsearch/reference/7.17/query-dsl-query-string-query.html#_reserved_characters
    :param query: a full query string
    :param all: True to escape all reserved characters
    :return: str

    NOTE! right now quoting may be being done in web-search??  At the
    very least, The ContentProvider class should expose a (static?)
    method, so there is less provider specific knowledge in client code.
    """
    if all:
        reserved = _ALL_RESERVED_CHARS
    else:
        reserved = _RARE_RESERVED_CHARS
    sanitized = []
    for char in query:
        if char in reserved:
            sanitized.append("\\") # r"\" doesn't work!
        sanitized.append(char)
    ret = ''.join(sanitized)
    return ret

def _format_match(hit: AttrDict, expanded: bool = False) -> dict:
    src = hit["_source"]
    res = {
        "article_title": _get(src, "article_title"),
        "publication_date": _get(src, "publication_date", "")[:10] or None,
        "indexed_date": _get(src, "indexed_date", None),
        "language": _get(src, "language", None),
        "full_langauge": _get(src, "full_language", None),
        "url": _get(src, "url", None),
        "original_url": _get(src, "original_url", None),
        "canonical_domain": _get(src, "canonical_domain", None),
        "id": hit["_id"]        # PB: was re-hash of url!
    }
    if expanded:
        res["text_content"] = _get(src, "text_content")
    return res

def _format_day_counts(bucket: list) -> dict[str, int]:
    """
    from news-search-api/api.py;
    used to format "dailycounts"

    takes [{"key": key, "doc_count": doc_count}, ....]
    and returns {key: count, ....}
    """
    return {item["key_as_string"][:10]: item["doc_count"] for item in bucket}


def _format_counts(bucket: list) -> dict[str, int]:
    """
    from news-search-api/api.py;
    used to format "topdomains" & "toplangs"

    takes [{"key": key, "doc_count": doc_count}, ....]
    and returns {key: count, ....}
    """
    return {item["key"]: item["doc_count"] for item in bucket}

# was published_date, but it's awful for pagination
# (can only return a single page per day!)
_DEF_PAGE_SORT_FIELD = "indexed_date"
_DEF_PAGE_SORT_ORDER = "desc"

PageTokenType = str       # ints returned as str

def _b64_encode_page_token(strng: str) -> str:
    return base64.b64encode(strng.encode(), b"-_").decode().replace("=", "~")

def _b64_decode_page_token(strng: str) -> str:
    return base64.b64decode(strng.replace("~", "=").encode(), b"-_").decode()

def _get_hits(res: AttrDict) -> list[AttrDict]:
    """
    retrieve hits array from _exec results
    """
    h1 = _get(res, "hits")
    if not h1:
        return []
    return _get(h1, "hits", [])

_ES_MAXPAGE = 1000

class OnlineNewsMediaCloudESProvider(OnlineNewsMediaCloudProvider):
    """
    experimental/temporary version of MC Provider going
    direct to Elastic (skipping both the news-search-api server
    and mediacloud.py, the client library that talks to n-s-a)
    """
    LOG_FULL_RESULTS = False    # log full _exec results @ debug

    def get_client(self):
        # called from OnlineNewsAbstractProvider, to set _client, so
        # can't raise exceptions, but want _client to be None, to
        # catch any attempts to use it!
        return None

    @staticmethod
    def _sanitize_query(fstr: str) -> str:
        """
        Do quoting done by _sanitize_es_query in mediacloud.py
        """
        return fstr.replace("/", r"\/")

    def _fields(self, expanded) -> list[str]:
        fields = ["article_title", "publication_date", "indexed_date",
                  "language", "full_language", "canonical_domain", "url", "original_url"]
        if expanded:
            fields.append("text_content")
        return fields

    def _basic_search(self, user_query: str, start_date: dt.datetime, end_date: dt.datetime,
                     expanded: bool = False, source: bool = True, **kwargs) -> Search:
        """
        from news-search-api/api.py cs_basic_query
        create a elasticsearch_dsl query from user_query date range and kwargs
        """
        # only sanitize user query
        sq = _sanitize_es_query(user_query)
        logger.debug("_basic_query %s sanitize %s", user_query, sq)

        # adds in canonical_domain/url query terms
        # XXX TODO: move to a .filter!
        q = self._assembled_query_str(sq, **kwargs)

        # works for date or datetime!
        start = start_date.strftime("%Y-%m-%d")
        end = end_date.strftime("%Y-%m-%d")

        s = Search(index=self._index_from_dates(start_date, end_date))\
            .query(QueryString(query=q, default_field="text_content", default_operator="and"))\
            .filter("range", publication_date={'gte': start, "lte": end})

        if source:
            return s.source(self._fields(expanded))
        else:
            return s.source(False)

    def _is_no_results(self, results) -> bool:
        """
        used to test _overview_query results
        """
        return not results

    def _index_from_dates(self, start_date: dt.datetime | None, end_date: dt.datetime | None) -> list[str]:
        """
        return list of indices to search for a given date range.
        if indexing goes being split by published_date (by year or quarter?)
        this could limit the number of shards that need to be queried
        """
        return [self.DEFAULT_COLLECTION]

    def _exec(self, search: Search) -> AttrDict:
        """
        one place to send queries to ES, for logging
        """
        logger.debug("MC._exec %r", search.to_dict())

        # Will always (re)start at first server?  No state keeping
        # is possible with web-search (creates a new Provider instance
        # for each query).  Maybe shuffle the list??
        eshosts = (self._base_url or "").split(",") # comma separated list of http://SERVER:PORT
        if not eshosts:
            raise ValueError("no ES url(s)")
        es = elasticsearch.Elasticsearch(eshosts, request_timeout=self._timeout, max_retries=3)

        t0 = time.monotonic()
        if self._caching < 0:
            # Here to try to force ES not to use cached results (for testing).
            # .execute(ignore_cache=True) only effects in-library caching; this
            # puts ?request_cache=false on the request URL, which
            # https://www.elastic.co/guide/en/elasticsearch/reference/current/shard-request-cache.html
            # says "The request_cache query-string parameter can be used to
            # enable or disable caching on a per-request basis. If set, it
            # overrides the index-level setting"
            search = search.params(request_cache=False)

        res = search.using(es).execute()
        elapsed = time.monotonic() - t0
        logger.debug("MC._exec ES time %s ms (%.3f elapsed)", _get(res, "took", -1), elapsed*1000)
        if self.LOG_FULL_RESULTS:
            logger.debug("MC._exec returning %r", res.to_dict()) # can be VERY big for paged_articles!!!
        return res

    @CachingManager.cache('overview')
    def _overview_query(self, q: str, start_date: dt.datetime, end_date: dt.datetime, **kwargs) -> dict:
        """
        from news-search-api/api.py
        returns empty dict when no hits
        """

        logger.debug("MC._overview %s %s %s %r", q, start_date, end_date, kwargs)

        AGG_DAILY = "daily"
        AGG_LANG = "lang"
        AGG_DOMAIN = "domain"

        search = self._basic_search(q, start_date, end_date, **kwargs)
        search.aggs.bucket(AGG_DAILY, "date_histogram", field="publication_date",
                           calendar_interval="day", min_doc_count=1)
        search.aggs.bucket(AGG_LANG, "terms", field="language.keyword", size=100)
        search.aggs.bucket(AGG_DOMAIN, "terms", field="canonical_domain", size=100)
        search = search.extra(track_total_hits=True)
        res = self._exec(search)
        hits = _get_hits(res)
        if not hits:
            return {}           # checked by _is_no_results

        aggs = res["aggregations"]
        return {
            "query": q,
            "total": res["hits"]["total"]["value"],
            "topdomains": _format_counts(aggs[AGG_DOMAIN]["buckets"]),
            "toplangs": _format_counts(aggs[AGG_LANG]["buckets"]),
            "dailycounts": _format_day_counts(aggs[AGG_DAILY]["buckets"]),
            "matches": [_format_match(h) for h in hits], # _match_to_row called in .sample()
        }

    def _terms(self, query: str, start_date: dt.datetime, end_date: dt.datetime,
               field: str, aggr: str, **kwargs) -> Dict:
        """
        only called internally, so no field/aggr checks
        """
        agg_name = "topterms"
        sampler_name = "sample"

        search = self._basic_search(query, start_date, end_date, source=False, **kwargs)
        if aggr == "top":
            agg_terms = Terms(field=field, size=200,
                              min_doc_count=10, shard_min_doc_count=5)
            shard_size = 500
        elif aggr == "significant":
            agg_terms = SignificantTerms(field=field, size=200,
                                         min_doc_count=10, shard_min_doc_count=5)
            shard_size = 500
        elif aggr == "rare":
            agg_terms = RareTerms(field=field, exclude="[0-9].*")
            shard_size = 10
        else:
            raise ValueError(aggr)

        search.aggs.bucket(sampler_name,
                           "sampler",
                           shard_size=shard_size).aggs[agg_name] = agg_terms

        res = self._exec(search)
        if (not _get_hits(res) or
            not (buckets := res["aggregations"][sampler_name][agg_name]["buckets"])):
            return {}

        return _format_counts(buckets)

    def _word_counts(self, query: str, start_date: dt.datetime, end_date: dt.datetime, **kwargs) -> Mapping:
        """
        called by OnlineNewsAbstractProvider.words
        """
        field = "article_title"
        agg = "top"             # also: significant & rare

        logger.debug("MC._words %s %s %s", query, start_date, end_date)
        # XXX doing direct ES queries, is it possible to put stop words filter into query??
        return self._terms(query, start_date, end_date, field, agg, **kwargs)

    @CachingManager.cache()
    def item(self, item_id: str) -> Dict:
        s = Search(index=self.DEFAULT_COLLECTION)\
            .query(Match(_id=item_id))\
            .source(includes=self._fields(expanded=True)) # always includes full_text!!
        res = self._exec(s)
        hits = _get_hits(res)
        if len(hits) == 0:
            return {}

        # double conversion!
        return self._match_to_row(_format_match(hits[0], True))

    def paged_items(
            self, query: str,
            start_date: dt.datetime, end_date: dt.datetime,
            page_size: int = _ES_MAXPAGE,
            **kwargs
    ) -> tuple[list[dict], Optional[PageTokenType]]:
        """
        return a single page of data (with `page_size` items).
        Pass `None` as first `pagination_token`, after that pass
        value returned by previous call, until `None` returned.

        `kwargs` may contain: `sort_field` (str), `sort_order` (str)
        """
        logger.debug("MC._paged_articles q: %s: %s e: %s ps: %d kw: %r",
                     query, start_date, end_date, page_size, kwargs)

        page_size = max(page_size, _ES_MAXPAGE)
        expanded = kwargs.pop("expanded", False)
        page_sort_field = kwargs.pop("page_sort_field", _DEF_PAGE_SORT_FIELD)
        page_sort_order = kwargs.pop("page_sort_order", _DEF_PAGE_SORT_ORDER)
        pagination_token = kwargs.pop("pagination_token", None)

        if page_sort_field not in self._fields(expanded):
            raise ValueError(page_sort_field)

        if page_sort_order not in ["asc", "desc"]:
            raise ValueError(page_sort_order)

        if kwargs:
            extra_keys = set(kwargs) - {'domains', 'filters'}
            if extra_keys:
                exstring = ", ".join(extra_keys)
                raise TypeError(f"unknown keyword args: {exstring}")

        page_sort_format = None
        if page_sort_field == "publication_date":
            page_sort_format = "basic_date" # YYYYMMDD (no need for encoding)
        elif page_sort_field == "indexed_date":
            # "date" fields are _supposed_ to be stored as milliseconds,
            # but supplied values have microseconds, and stored
            # values seem to be returned with them?!
            page_sort_format = "epoch_millis"

        if page_sort_format:
            # numeric string: no encoding needed
            # (unless obfuscation is the goal)
            _encode_page_token = _decode_page_token = lambda x: x
        else:
            _encode_page_token = _b64_encode_page_token
            _decode_page_token = _b64_decode_page_token

        if page_sort_format:
            sort_opts = {
                # XXX types.SortOptions (not in 8.15.4)
                page_sort_field: {
                    # XXX types.FieldSort (not in 8.15.4)
                    "order": page_sort_order,
                    "format": page_sort_format
                }
            }
        else:
            sort_opts = {page_sort_field: page_sort_order}

        search = self._basic_search(query, start_date, end_date, expanded=expanded, **kwargs)\
                     .extra(size=page_size, track_total_hits=True)\
                     .sort(sort_opts)
        if pagination_token:
            # important to use `search_after` instead of 'from' for
            # memory reasons related to paging through more than 10k
            # results.
            search = search.extra(search_after=[_decode_page_token(pagination_token)])

        res = self._exec(search)
        hits = _get_hits(res)
        if not hits:
            return ([], "")

        if len(hits) == page_size:
            # token is made from first/only sort key value for last item
            new_pt = _encode_page_token(hits[-1]["sort"][0])
        else:
            new_pt = ""

        # double conversion!
        rows = self._matches_to_rows([_format_match(h, expanded) for h in hits])
        return (rows, new_pt)

    def all_items(self, query: str,
                  start_date: dt.datetime, end_date: dt.datetime,
                  page_size: int = _ES_MAXPAGE, **kwargs):
        """
        returns generator of pages (lists) of items
        """
        next_page_token: PageTokenType | None = None
        while True:
            page, next_page_token = self.paged_items(
                query, start_date, end_date,
                page_size=page_size,
                pagination_token=next_page_token,
                **kwargs)

            if not page:
                break

            yield page

            if not next_page_token:
                break

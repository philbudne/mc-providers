# XXX maybe make _assemble.... take dict, and _prune (pop) args there???

import datetime as dt
import json
import logging
import os
import random
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional

# PyPI
import ciso8601
import dateparser     # used for publication_date in IA match_to_row
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
        for subquery in self._assemble_and_chunk_query_str_kw(query, kwargs):
            this_results = self._client.sample(subquery, start_date, end_date, **kwargs)
            results.extend(this_results)
        
        if len(results) > limit:
            results = random.sample(results, limit)
            
        return self._matches_to_rows(results)

    # Chunk'd
    def count(self, query: str, start_date: dt.datetime, end_date: dt.datetime, **kwargs) -> int:
        count = 0
        for subquery in self._assemble_and_chunk_query_str_kw(query, kwargs):
            count += self._client.count(subquery, start_date, end_date, **kwargs)
        return count

    # Chunk'd
    def count_over_time(self, query: str, start_date: dt.datetime, end_date: dt.datetime, **kwargs) -> Dict:
        counter: Counter = Counter()
        for subquery in self._assemble_and_chunk_query_str_kw(query, kwargs):
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
        for subquery in self._assemble_and_chunk_query_str(query, kwargs):
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
        query = self._assemble_and_chunk_query_str_kw(query, updated_kwargs)[0]
        page, pagination_token = self._client.paged_articles(query, start_date, end_date, **kwargs)
        return self._matches_to_rows(page), pagination_token

    def _word_counts(self, query: str, start_date: dt.datetime, end_date: dt.datetime, **kwargs) -> tuple[int, Mapping]:
        logger.debug("AP._word_counts %s %s %s %r", query, start_date, end_date, kwargs)

        kwargs.pop("exclude", None)
        chunked_queries = self._assemble_and_chunk_query_str_kw(query, kwargs)

        # An accumulator for the subqueries
        totals: Counter = Counter()

        for subquery in chunked_queries:
            logger.debug("AP.words subquery %s %s %s", subquery, start_date, end_date)

            # for now just return top terms in article titles
            this_results = self._client.terms(subquery, start_date, end_date,
                                              self._client.TERM_FIELD_TITLE,
                                              self._client.TERM_AGGREGATION_TOP)
            if "detail" not in this_results:
                if self.DEBUG_WORDS:
                    logger.debug("AP.words results %s", this_results)
                totals += Counter(this_results)
        # PB: 5000 from "sample_size" in .words method
        return (5000, totals)

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

        sample_size, results_counter = self._word_counts(query, start_date, end_date,
                                                         exclude=list(stopwords), **kwargs)
        if self.DEBUG_WORDS > 1:
            logger.debug("AP.words total %r", results_counter)
            for t, c in results_counter.items():
                logger.debug("%s %d %.6f %s", t, c, c/sample_size, t.lower() not in stopwords)

        # and clean up results to return
        top_terms = self._get_top_terms(results_counter, stopwords, sample_size)
        top_terms = sorted(top_terms, key=lambda x:x["count"], reverse=True)
        return top_terms

    def _get_top_terms(self, results_counter: Mapping, stopwords: Iterable[str], sample_size: int) -> Dict:
        """
        eliminate any stopwords and format for return
        """
        return [dict(term=t.lower(), count=c, ratio=c/sample_size) for t, c in results_counter.items()
                if t.lower() not in stopwords]

    # Chunk'd
    def languages(self, query: str, start_date: dt.datetime, end_date: dt.datetime, limit: int = 10,
                  **kwargs) -> List[Dict]:
        
        matching_count = self.count(query, start_date, end_date, **kwargs)

        results_counter: Counter = Counter({})
        for subquery in subqself._assemble_and_chunk_query_str_kw(query, kwargs):
            this_languages = self._client.top_languages(subquery, start_date, end_date, **kwargs)
            countable = {item["name"]: item["value"] for item in this_languages}
            results_counter += Counter(countable)
            # top_languages.extend(this_languages)
        
        top_languages = [{'language': name, 'value': value, 'ratio': 0.0} for name, value in results_counter.items()]
        
        for item in top_languages:
            item['ratio'] = item['value'] / matching_count
        
        # Sort by count, then alphabetically
        top_languages = sorted(top_languages, key=lambda x: x['value'], reverse=True)
        return top_languages[:limit]

    # Chunk'd
    def sources(self, query: str, start_date: dt.datetime, end_date: dt.datetime, limit: int = 100,
                **kwargs) -> List[Dict]:
        
        results_counter: Counter = Counter({})
        for subquery in self._assemble_and_chunk_query_str_kw(query, kwargs):
            results = self._client.top_sources(subquery, start_date, end_date)
            countable = {source['name']: source['value'] for source in results}
            results_counter += Counter(countable)
        
        cleaned_sources = [{"source": source , "count": count} for source, count in results_counter.items()]
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
    def _prune_kwargs(kwargs: dict[str, Any]) -> None:
        """
        takes a query **kwargs dict and removes keys that
        are processed in this library, and should not be passed to clients.
        """
        # remove Chunked?
        kwargs.pop("domains", None) # can be a set
        kwargs.pop("filters", None) # can be a set
        kwargs.pop("url_search_strings", None) # can be a dict[str, set]
        kwargs.pop("url_search_string_domain", None) # TEMP

    @classmethod
    def _assemble_and_chunk_query_str_kw(cls, base_query: str, chunk: bool = True, kwargs: dict = {}):
        """
        takes kwargs as dict, removes items that shouldn't be sent to _client
        """
        queries = cls._assemble_and_chunk_query_str(base_query, chunk=chunk, **kwargs)
        cls._prune_kwargs(kwargs)
        return queries

    @staticmethod
    def _sanitize_query(fstr: str) -> str:
        """
        noop: (/ quoting done in mediacloud.py sanitize_query),
        presumably waybacknews also does??
        """
        return fstr

    @classmethod
    def _selector_query_clauses(cls, kwargs: dict) -> list[str]:
        """
        take domains and filters kwargs and
        returns a list of query_strings to be OR'ed together
        (to be AND'ed with user query *or* used as a filter)
        """
        logger.debug("AP._selector_query_clauses IN: %r", kwargs)
        selector_clauses = []

        domains = kwargs.get('domains', [])
        if len(domains) > 0:
            domain_strings = " OR ".join(domains)
            selector_clauses.append(f"{cls.domain_search_string()}:({domain_strings})")
            
        # put all filters in single query string (NOTE: additive)
        filters = kwargs.get('filters', [])
        if len(filters) > 0:
            for filter in filters:
                f = cls._sanitize_query(filter)
                if "AND" in f:
                    # parenthesize if any chance it has a grabby AND.
                    selector_clauses.append(f"({f})")
                else:
                    selector_clauses.append(f)
        logger.debug("AP._selector_query_clauses OUT: %s", selector_clauses)
        return selector_clauses

    @classmethod
    def _selector_query_string_from_clauses(cls, clauses: list[str]) -> str:
        return " OR ".join(clauses)

    @classmethod
    def _selector_query_string(cls, kwargs: dict) -> str:
        """
        takes kwargs (as dict) return a query_string to be AND'ed with
        user query or used as a filter.
        """
        return cls._selector_query_string_from_clauses(cls._selector_query_clauses(kwargs))

    @classmethod
    def _assembled_query_str(cls, query: str, **kwargs) -> str:
        logger.debug("_assembled_query_str IN: %s %r", query, kwargs)
        sqs = cls._selector_query_string(kwargs) # takes dict
        if sqs:
            q = f"({query}) AND ({sqs})"
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


################
# helpers for formatting url_search_strings (only enabled for MC)
# the helpers are only needed because of the TEMP url_search_string_domain

def format_and_append_uss(uss: str, url_list: list[str]) -> None:
    """
    The ONE place that knows how to format a url_search_string!!!
    (ie; what to put before and after one).

    Appends to `url_list` argument!

    NOTE! generates "unsanitized" (unsanitary?) strings!!

    Currently (11/2024) A URL Search String should:
    1. Start with fully qualified domain name WITHOUT http:// or https://
    2. End with "*"
    """
    # currently url_search_strings MUST start with fully
    # qualified domain name (FQDN) without scheme or
    # leading slashes, and MUST end with a *!
    if not uss.endswith("*"):
        uss += "*"
    url_list.append(f"http\\://{uss}")
    url_list.append(f"https\\://{uss}")

def match_formatted_search_strings(fuss: list[str]) -> str:
    """
    takes list of url search_string formatted by `format_and_append_uss`
    returns query_string fragment
    """
    assert fuss
    urls_str = " OR ".join(fuss)
    return f"url:({urls_str})"


class OnlineNewsMediaCloudProvider(OnlineNewsAbstractProvider):
    """
    Provider interface to access new mediacloud-news-search archive. 
    All these endpoints accept a `domains: List[str]` keyword arg.
    """
    
    DEFAULT_COLLECTION = os.environ.get(
        "ELASTICSEARCH_INDEX_NAME_PREFIX", "mc_search") + "-*"

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


    @classmethod
    def _selector_query_clauses(cls, kwargs: dict) -> str:
        """
        take domains, filters, url_search_strings as kwargs
        return a list of query_strings to be OR'ed together
        (to be AND'ed with user query or used as a filter)
        """
        logger.debug("MC._selector_query_clauses IN: %r", kwargs)
        selector_clauses = super()._selector_query_clauses(kwargs)

        # Here to try to get web-search out of query
        # formatting biz.  Accepts a Mapping indexed by
        # domain_string, of lists (or sets!) of search_strings.
        url_search_strings: Mapping[str, Iterable[str]] = kwargs.get('url_search_strings', [])
        if url_search_strings:
            # Unclear if domain field check actually helps at all,
            # so make it optional for testing.
            if kwargs.get("url_search_string_domain", True): # TEMP: include canonincal_domain:
                domain_field = cls.domain_search_string()

                # here with mapping of cdom => iterable[search_string]
                for cdom, search_strings in url_search_strings.items():
                    fuss = [] # formatted url_search_strings
                    for sstr in search_strings:
                        format_and_append_uss(sstr, fuss)

                    mfuss = match_formatted_search_strings(fuss)
                    selector_clauses.append(
                        cls._sanitize_query(
                            f"({domain_field}:{cdom} AND {mfuss})"))

                    format_and_append_uss(cdom, fuss)
            else: # make query without domain (name) field check
                # collect all the URL search strings
                fuss = []
                for cdom, search_strings in url_search_strings.items():
                    for sstr in search_strings:
                        format_and_append_uss(sstr, fuss)

                # check all the urls in one swell foop!
                selector_clauses.append(
                    cls._sanitize_query(
                        match_formatted_search_strings(fuss)))

        logger.debug("MC._selector_query_clauses OUT: %s", selector_clauses)
        return selector_clauses

    def count(self, query: str, start_date: dt.datetime, end_date: dt.datetime, **kwargs) -> int:
        logger.debug("MC.count %s %s %s %r", query, start_date, end_date, kwargs)
        # no chunking on MC
        results = self._overview_query(query, start_date, end_date, **kwargs)
        return self._count_from_overview(results)

    def _count_from_overview(self, results: Dict) -> int:
        """
        used in .count() and .languages()
        """
        if self._is_no_results(results):
            logger.debug("MC._count_from_overview: no results")
            return 0
        count = results['total']
        logger.debug("MC._count_from_overview: %s", count)
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
        matching_count = self._count_from_overview(results)
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
        self._prune_kwargs(kwargs)
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
import importlib.metadata       # for default opaque_id
import json
import os
import time
from typing import Callable

import elasticsearch
import mcmetadata.urls as urls
from elasticsearch_dsl import Search, Response
from elasticsearch_dsl.aggs import RareTerms, Sampler, SignificantTerms, SignificantText, Terms
from elasticsearch_dsl.query import Match, QueryString
#from elasticsearch_dsl.types import FieldSort, SortOptions # not in 8.15
from elasticsearch_dsl.utils import AttrDict

# from api-client/.../api.py
try:
    VERSION = "v" + importlib.metadata.version("mc-providers")
except importlib.metadata.PackageNotFoundError:
    VERSION = "dev"

OPAQUE_ID = f"mc-providers {VERSION}"

def _get(ad: AttrDict, key: str, default: Any = None) -> Any:
    """
    AttrDict doesn't have a "get" method; hide that here to make it
    easy to switch back to _search returning res.to_dict() if needed...
    """
    return getattr(ad, key, default)

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

    takes [{"key_as_string": "YYYY-MM-DDT00:00:00.000Z", "doc_count": count}, ....]
    and returns {"YYYY-MM-DD": count, ....}
    """
    return {item["key_as_string"][:10]: item["doc_count"] for item in bucket}


def _format_counts(bucket: list) -> dict[str, int]:
    """
    from news-search-api/api.py
    used to format "topdomains" & "toplangs"

    takes [{"key": key, "doc_count": doc_count}, ....]
    and returns {key: count, ....}
    """
    return {item["key"]: item["doc_count"] for item in bucket}

# was publication_date, but it's awful for pagination
# (can only return a single page per day!)
_DEF_PAGE_SORT_FIELD = "indexed_date"
_DEF_PAGE_SORT_ORDER = "desc"

def _b64_encode_page_token(strng: str) -> str:
    return base64.b64encode(strng.encode(), b"-_").decode().replace("=", "~")

def _b64_decode_page_token(strng: str) -> str:
    return base64.b64decode(strng.replace("~", "=").encode(), b"-_").decode()

def _get_hits(res: Response) -> list[AttrDict]:
    """
    retrieve hits array from _search results
    here to check Response in MOST cases
    """
    # Response.success() wants
    # `self._shards.total == self._shards.successful and not self.timed_out`
    # _search method will have already logged any failed shards.
    if not res.success():
        logger.warn("res.success() is False!")
        return []

    try:
        return res.hits.hits
    except AttributeError:
        logger.warn("res.hits.hits failed!")
        return []

_ES_MAXPAGE = 1000

class OnlineNewsMediaCloudESProvider(OnlineNewsMediaCloudProvider):
    """
    version of MC Provider going direct to ES.

    Consolidates query formatting/creation previously spread
    across multiple files, including:

    * web-search/mcweb/backend/search/utils.py (url_search_strings)
    * this file (domain search string)
    * mc-providers/mc_providers/mediacloud.py (date ranges)
    * news-search-api/api.py
    * news-search-api/client.py (DSL, including aggegations)
    """
    def __init__(self, *args, **kwargs):
        # CAN pass string (filename) here, but feeding all the
        # resulting JSON files to es-tools/collapse-esperf.py for
        # flamegraphing could get you a mish-mash of different
        # queries' results.
        self._profile = kwargs.pop("profile", False)

        # total seconds from the last profiled query:
        self._last_es_seconds = -1.0

        # https://www.elastic.co/guide/en/elasticsearch/reference/current/api-conventions.html
        # says:
        #   The X-Opaque-Id header accepts any arbitrary
        #   value. However, we recommend you limit these values to a
        #   finite set, such as an ID per client. Don’t generate a
        #   unique X-Opaque-Id header for every request. Too many
        #   unique X-Opaque-Id values can prevent Elasticsearch from
        #   deduplicating warnings in the deprecation logs.
        # Which probaly REALLY means one string per client LIBRARY!
        # See preference below.
        opaque_id = kwargs.pop("client_id", None) # more generic public name
        if not opaque_id:
            opaque_id = OPAQUE_ID

        # https://www.elastic.co/guide/en/elasticsearch/reference/7.17/search-search.html#search-preference
        #   "Any string that does not start with _. If the cluster
        #   state and selected shards do not change, searches using
        #   the same <custom-string> value are routed to the same
        #   shards in the same order.
        # pass user-id or a session id
        self._preference = kwargs.pop("preference", None)

        # after pop-ing any local-only args:
        super().__init__(*args, **kwargs)

        eshosts = (self._base_url or "").split(",") # comma separated list of http://SERVER:PORT

        # Retries without delay (never mind backoff!)
        # web-search creates new Provider for each API request,
        # so randomize the pool.
        self._es = elasticsearch.Elasticsearch(eshosts,
                                               max_retries=3,
                                               opaque_id=opaque_id,
                                               request_timeout=self._timeout,
                                               randomize_nodes_in_pool=True)


    def get_client(self):
        """
        called from OnlineNewsAbstractProvider constructor to set _client
        so must pretend we've done it.
        """
        # tempting to stash Elasticsearch object here, BUT, at least
        # for initial work wanted to make sure any existing code path
        # that tried using the _client object would blow up.  It
        # probably helps mypy to have a dedicated _es member for
        # Elasticsearch object.
        return None

    @staticmethod
    def _sanitize_query(fstr: str) -> str:
        """
        Do quoting done by _sanitize_es_query in mediacloud.py client library
        """
        return fstr.replace("/", r"\/")

    def _fields(self, expanded) -> list[str]:
        """
        from news-search-api/client.py QueryBuilder constructor
        """
        fields = ["article_title", "publication_date", "indexed_date",
                  "language", "full_language", "canonical_domain", "url", "original_url"]
        if expanded:
            fields.append("text_content")
        return fields

    # relative costs of days vs sources
    # just guesses for now
    SOURCE_WEIGHT = 1
    DAY_WEIGHT = 25

    def _basic_search(self, user_query: str, start_date: dt.datetime, end_date: dt.datetime,
                     expanded: bool = False, source: bool = True, **kwargs) -> Search:
        """
        from news-search-api/api.py cs_basic_query
        create a elasticsearch_dsl query from user_query date range and kwargs
        """
        # only sanitize user query
        sq = self._sanitize_query(user_query)
        logger.debug("_basic_query %s sanitize %s", user_query, sq)

        # works for date or datetime!
        start = start_date.strftime("%Y-%m-%dT00:00:00Z") # paranoia
        end = end_date.strftime("%Y-%m-%d") # T23:59:59:999_999_999Z implied?

        s = Search(index=self._index_from_dates(start_date, end_date), using=self._es)\
            .query(QueryString(query=sq, default_field="text_content", default_operator="and"))

        if self._preference:
            s = s.params(preference=self._preference)

        # Evaluating selectors (domains/filters/url_search_strings) in "filter context";
        # Supposed to be faster, and enable caching of which documents to look at.
        # https://www.elastic.co/guide/en/elasticsearch/reference/current/query-filter-context.html#filter-context

        # Apply filter with the smallest result set first.
        # could include languages etc here

        filters : list[tuple[int, Callable]] = [
            ((end_date - start_date).days * self.DAY_WEIGHT,
             lambda s : s.filter("range", publication_date={'gte': start, "lte": end}))
        ]

        selector_clauses = self._selector_query_clauses(kwargs)
        if selector_clauses:
            # Maybe someday construct DSL rather than resorting to string we format
            # only to have ES have to parse it??
            sqs = self._selector_query_string_from_clauses(selector_clauses)
            filters.append((len(selector_clauses) * self.SOURCE_WEIGHT,
                            lambda s : s.filter(QueryString(query=sqs))))

        filters.sort()          # smallest weight first
        for weight, func in filters:
            s = func(s)

        if source:              # return source?
            return s.source(self._fields(expanded))
        else:
            return s.source(False) # no source fields in hits

    def _is_no_results(self, results) -> bool:
        """
        used to test _overview_query results
        """
        return not results

    def _index_from_dates(self, start_date: dt.datetime | None, end_date: dt.datetime | None) -> list[str]:
        """
        return list of indices to search for a given date range.
        if indexing goes back to being split by publication_date (by year or quarter?)
        this could limit the number of shards that need to be queried

        I
        """
        return [self.DEFAULT_COLLECTION]

    def _search(self, search: Search, profile: str | bool = False) -> Response:
        """
        one place to send queries to ES, for logging
        """
        logger.debug("MC._search %r", search.to_dict())

        t0 = time.monotonic()
        if self._caching < 0:
            # Here to try to force ES not to use cached results (for testing).
            # .execute(ignore_cache=True) only effects in-library caching.
            # This puts ?request_cache=false on the request URL, which
            # https://www.elastic.co/guide/en/elasticsearch/reference/current/shard-request-cache.html
            # says "The request_cache query-string parameter can be
            # used to enable or disable caching on a per-request
            # basis. If set, it overrides the index-level setting"
            search = search.params(request_cache=False)

        profile = profile or self._profile
        if profile:
            search = search.extra(profile=True)

        res = search.execute()
        elapsed = time.monotonic() - t0
        logger.info("MC._search ES time %s ms (%.3f elapsed)", _get(res, "took", -1), elapsed*1000)

        if profile and (pdata := _get(res, "profile")):
            self._profile_data(pdata, profile)

        try:                    # look for circuit breaker trips
            # Response.success() wants success
            # self._shards.total == self._shards.successful and not self.timed_out
            # but we can't be choosey right now because of top words (mis)behavior
            # the logging below WILL happen often!
            shards = res._shards
            if shards:
                failed = shards.failed
                total = shards.total
                if failed:
                    # hundreds of shards, so summarize...
                    # (almost always circuit breakers)
                    reasons = Counter()
                    for shard in shards.failures:
                        rt = shard.reason.type
                        if rt:
                            reasons[rt] += 1
                    logger.info("MC._search %d/%d shards failed; reasons: %r", failed, total, dict(reasons))
        except (ValueError, KeyError) as e:
            logger.debug("error looking at results: %r", e)
        return res

    def _profile_data(self, pdata, profile: str | bool) -> None:
        """
        digest profiling data
        """
        if isinstance(profile, str):
            fname = time.strftime(f"{profile}-%Y-%m-%d-%H-%M-%S.json")
            with open(fname, "w") as f:
                json.dump(pdata.to_dict(), f)
            logger.debug("wrote profiling data to %s", fname)

        # sum up ES internal times
        query_ns = rewrite_ns = coll_ns = agg_ns = 0
        for shard in pdata.shards: # AttrList
            for search in shard.searches: # AttrList
                for q in search.query:    # AttrList
                    query_ns += q.time_in_nanos
                for coll in search.collector: # list
                    coll_ns += coll.time_in_nanos
                rewrite_ns += search.rewrite_time
            # XXX sum by aggregation name?
            for agg in shard.aggregations:
                agg_ns += agg.time_in_nanos
        es_nanos = query_ns + rewrite_ns + coll_ns + agg_ns
        self._last_es_seconds = es_nanos / 1e9
        # XXX save components???
        # XXX sum over Provider lifetime??

        # avoid floating point divisions that may not be displayed:
        logger.info("ES time: %.6f sec (ns: query: %d rewrite: %d collectors: %d aggs: %d)",
                     self._last_es_seconds, query_ns, rewrite_ns, coll_ns, agg_ns)

    @CachingManager.cache('overview')
    def _overview_query(self, q: str, start_date: dt.datetime, end_date: dt.datetime, **kwargs) -> dict:
        """
        from news-search-api/api.py
        returns empty dict when no hits
        """

        logger.debug("MC._overview %s %s %s %r", q, start_date, end_date, kwargs)

        # these are arbitrary, but match news-search-api/client.py
        # so that top-queries.py can recognize this is an overview query:
        AGG_DAILY = "dailycounts"
        AGG_LANG = "toplangs"
        AGG_DOMAIN = "topdomains"

        profile = kwargs.pop("profile", None)

        search = self._basic_search(q, start_date, end_date, **kwargs)
        search.aggs.bucket(AGG_DAILY, "date_histogram", field="publication_date",
                           calendar_interval="day", min_doc_count=1)
        search.aggs.bucket(AGG_LANG, "terms", field="language.keyword", size=100)
        search.aggs.bucket(AGG_DOMAIN, "terms", field="canonical_domain", size=100)
        search = search.extra(track_total_hits=True)
        res = self._search(search, profile=profile)
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
               field: str, aggr: str, exclude: list[str] = [], **kwargs) -> tuple[int, Dict]:
        """
        only called internally, so no field/aggr checks
        """
        agg_name = "topterms"
        sampler_name = "sample"

        profile = kwargs.pop("profile", None)
        search = self._basic_search(query, start_date, end_date, source=False, **kwargs)
        if aggr == "top":
            # To get more accurate results, the terms agg fetches more
            # than the top size terms from each shard. It fetches the
            # top `shard_size` terms, which defaults to `size` * 1.5 + 10.

            # Terms must appear in `min_doc_count` documents (default is 1).

            # The parameter `shard_min_doc_count` regulates the
            # certainty a shard has if the term should actually be
            # added to the candidate list or not with respect to the
            # min_doc_count. Terms will only be considered if their
            # local shard frequency within the set is higher than the
            # shard_min_doc_count.

            agg_terms = Terms(field=field, size=200,
                              min_doc_count=10, # default is 1
                              shard_min_doc_count=5, # default is 0
                              exclude=exclude) # stop words!
            shard_size = 500
        elif aggr == "significant":
            # Returns interesting or unusual occurrences of terms in a
            # set.  The terms selected are not simply the most popular
            # terms in a set. They are the terms that have undergone a
            # significant change in popularity measured between a
            # foreground and background set.

            agg_terms = SignificantTerms(field=field, size=200,
                                         min_doc_count=10, shard_min_doc_count=5,
                                         exclude=exclude) # stop words!
            shard_size = 500

            # See also "significant text" aggregation:
            # Designed for use on `text` fields.
            # Does not require field data.
            # Re-analyzes text on the fly.
        elif aggr == "rare":
            # "rare" terms are at the long-tail of the distribution
            # and are not frequent. Conceptually, this is like a terms
            # aggregation that is sorted by _count ascending.

            # from https://github.com/internetarchive/web_collection_search/blob/2e189b5/api.py#L179??
            agg_terms = RareTerms(field=field, exclude="[0-9].*")
            shard_size = 10
        elif aggr == "significant_text": # phil's addition
            # Returns interesting or unusual occurrences of terms in a
            # set.  The terms selected are not simply the most popular
            # terms in a set. They are the terms that have undergone a
            # significant change in popularity measured between a
            # foreground and background set.

            # Can be used on any `text` field; does NOT require
            # "fielddata" (and the memory problems that entails) BUT
            # means text needs to be analyzed on the fly, with higher
            # CPU usage.
            agg_terms = SignificantText(field=field,
                                         exclude=exclude) # stop words!
            shard_size = 500
        else:
            raise ValueError(aggr)

        # The `shard_size` parameter limits how many top-scoring
        # documents are collected in the sample processed on each
        # shard. The default value is 100.

        search.aggs.bucket(sampler_name,
                           "sampler",
                           shard_size=shard_size).aggs[agg_name] = agg_terms
 
        # Try asking for no "hits" array; This means we can't use
        # _get_hits (which uses res.success() which checks for failed
        # shards (which often happen here due to circuit breakers
        # tripping) here! track_total_hits=False is the default,
        # but being extra careful.
        search = search.extra(size=0, track_total_hits=False)

        res = self._search(search, profile=profile)

        # Response.success() wants _shards.success == _shards.total,
        # but we need to be happy with what we can get!
        # _search will already have logged a summary of failed shards.
        if not res.timed_out:
            samples = res["aggregations"][sampler_name]
            buckets = samples[agg_name]["buckets"]
            doc_count = samples["doc_count"]
            logger.debug("scanned %d docs, %d buckets returned", doc_count, len(buckets))
            if buckets and doc_count:
                return (doc_count, _format_counts(buckets))

        return (0, {})


    def _word_counts(self, query: str, start_date: dt.datetime, end_date: dt.datetime, **kwargs) -> tuple[int, Mapping]:
        """
        called by OnlineNewsAbstractProvider.words.
        NOTE! uses "exclude" in kwargs, with list of stop words!
        """
        logger.debug("MC._words %s %s %s %r", query, start_date, end_date, kwargs)

        field = "article_title" # or "text"
        #field = "text"
        aggr = "top"
        # aggr = "significant_text"
        return self._terms(query, start_date, end_date, field=field, aggr=aggr, **kwargs)

    def _get_top_terms(self, results_counter: Mapping, stopwords: Iterable[str], sample_size: int) -> Dict:
        """
        called by OnlineNewsAbstractProvider.words.
        NOTE! Not using stopwords (we've asked ES to exclude them)
        XXX SHOULD BE TESTED!!!
        """
        return [dict(term=t.lower(), count=c, ratio=c/sample_size) for t, c in results_counter.items()]

    @CachingManager.cache()
    def item(self, item_id: str) -> Dict:
        s = Search(index=self.DEFAULT_COLLECTION, using=self._es)\
            .query(Match(_id=item_id))\
            .source(includes=self._fields(expanded=True)) # always includes full_text!!
        res = self._search(s)
        hits = _get_hits(res)
        if not hits:
            return {}

        # double conversion!
        return self._match_to_row(_format_match(hits[0], True))

    def paged_items(
            self, query: str,
            start_date: dt.datetime, end_date: dt.datetime,
            page_size: int = 1000,
            **kwargs
    ) -> tuple[list[dict], Optional[str]]:
        """
        return a single page of data (with `page_size` items).
        Pass `None` as first `pagination_token`, after that pass
        value returned by previous call, until `None` returned.

        `kwargs` may contain: `sort_field` (str), `sort_order` (str)
        """
        logger.debug("MC._paged_articles q: %s: %s e: %s ps: %d kw: %r",
                     query, start_date, end_date, page_size, kwargs)

        page_size = min(page_size, _ES_MAXPAGE)
        expanded = kwargs.pop("expanded", False)
        page_sort_field = kwargs.pop("page_sort_field", _DEF_PAGE_SORT_FIELD)
        page_sort_order = kwargs.pop("page_sort_order", _DEF_PAGE_SORT_ORDER)
        pagination_token = kwargs.pop("pagination_token", None)

        if page_sort_field not in self._fields(expanded):
            raise ValueError(page_sort_field)

        if page_sort_order not in ["asc", "desc"]:
            raise ValueError(page_sort_order)

        profile = kwargs.pop("profile", None)
        kwcopy = kwargs.copy()
        self._prune_kwargs(kwcopy)
        if kwcopy:
            exstring = ", ".join(kwcopy) # join key names
            raise TypeError(f"unknown keyword args: {exstring}")

        # XXX types.SortOptions (not in 8.15.4)
        sort_opts = {page_sort_field: page_sort_order}

        search = self._basic_search(query, start_date, end_date, expanded=expanded, **kwargs)\
                     .extra(size=page_size, track_total_hits=True)\
                     .sort(sort_opts)

        if pagination_token:
            # important to use `search_after` instead of 'from' for
            # memory reasons related to paging through more than 10k
            # results.
            after = [_b64_decode_page_token(pagination_token)] # list of keys
            search = search.extra(search_after=after)

        res = self._search(search, profile=profile)
        hits = _get_hits(res)
        if not hits:
            return ([], "")

        if len(hits) == page_size:
            # get paging token from first sort key of last item returned.
            # str() needed for dates, which are returned as integer milliseconds
            new_pt = _b64_encode_page_token(str(hits[-1]["sort"][0]))
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
        next_page_token: str | None = None
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

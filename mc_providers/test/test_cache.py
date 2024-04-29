import unittest
import datetime as dt
import pytest
import os
from mc_providers import (provider_for, PLATFORM_ONLINE_NEWS, PLATFORM_SOURCE_MEDIA_CLOUD)
from mc_providers.cache import CachingManager

MEDIA_CLOUD_API_KEY = os.getenv('MEDIA_CLOUD_API_KEY', None)

IN_GITHUB_CI_WORKFLOW = os.getenv("GITHUB_ACTIONS") == "true"

cache = {}


def _cache_get_key(*args, **kwargs):
    import hashlib
    serialise = []
    for arg in args:
        serialise.append(str(arg))
    for key, arg in kwargs.items():
        serialise.append(str(key))
        serialise.append(str(arg))
    key = hashlib.md5("".join(serialise).encode("UTF8")).hexdigest()
    return key


def _cache_handler(fn, cache_prefix, *args, **kwargs) -> tuple:
    was_cached = False
    key = _cache_get_key(cache_prefix, *args, **kwargs)
    results = cache.get(key)
    if not results:
        was_cached = False
        results = fn(*args, **kwargs)
        cache[key] = results
    return results, was_cached


@pytest.mark.skipif(IN_GITHUB_CI_WORKFLOW, reason="requires VPN tunnel to Media Cloud News Search API server")
class CacheMediaCloudTest(unittest.TestCase):

    def setUp(self) -> None:
        self._provider = provider_for(PLATFORM_ONLINE_NEWS, PLATFORM_SOURCE_MEDIA_CLOUD, None,
                                      "http://localhost:8010/v1/")
        CachingManager.cache_function = _cache_handler

    def _reset_cache(self):
        global cache
        cache = {}
        assert len(cache.keys()) == 0

    def _verify_cache_size(self, expected_size):
        assert len(cache.keys()) == expected_size

    def tearDown(self):
        self._reset_cache()
        CachingManager.cache_function = None

    def test_count(self):
        self._reset_cache()
        self._verify_cache_size(0)
        results = self._provider.count("coronavirus", dt.datetime(2023, 11, 1),
                                       dt.datetime(2023, 12, 1))
        self._verify_cache_size(1)

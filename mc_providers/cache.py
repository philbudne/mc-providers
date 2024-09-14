from typing import Callable, Optional

from .provider import ContentProvider

class CachingManager():
    """
    Use this to set caching functionality on the providers library for whatever system your using
    something like:
    `CachingManager.cache_function = your_cache_function`
    your_cache_function should have a signature like (function_to_cache, cache_prefix, args, kwargs) -> tuple[Any, bool]
    (second item in tuple indicatin if it was cached or not already)
    """
    cache_function = None

    @classmethod
    def cache(cls, custom_prefix_for_key: Optional[str] = None):
        """
        @param custom_prefix_for_key: if specified, will be used in place of function name for cache_key generation
        """
        def decorator(fn: Callable):
            def wrapper(*args, **kwargs):
                cache_prefix = custom_prefix_for_key if custom_prefix_for_key is not None else fn.__name__
                # see if first argument is a ContentProvider (invocation of ContentProvider method)
                # this could be less ugly if "cache" was ONLY EVER used on ContentProvider methods!
                caching = len(args) < 1 or not isinstance(args[0], ContentProivider) or args[0]._caching
                if caching and cls.cache_function is not None:
                    # use the function name
                    results, was_cached = cls.cache_function(fn, cache_prefix, *args, **kwargs)
                    return results
                else:
                    return fn(*args, **kwargs)
            return wrapper
        return decorator

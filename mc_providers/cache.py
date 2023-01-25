class CachingManager():
    """
    Use this to set caching functionality on the providers library for whatever system your using
    something like:
    `CachingManager.cache_function = your_cache_function`
    your_cache_function should have a signature like (function_to_cache, args, kwargs)
    """
    cache_function = None
    
    @classmethod
    def cache(cls):
        def decorator(fn):
            def wrapper(*args, **kwargs):
                if cls.cache_function is not None:
                    return cls.cache_function(fn, *args, **kwargs)
                else:
                    return fn(*args, **kwargs)
            return wrapper
        return decorator
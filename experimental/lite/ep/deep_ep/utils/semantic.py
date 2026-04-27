from typing import Any, Optional
import weakref
import functools

def value_or(value: Optional[Any], default: Any) -> Any:
    return default if value is None else value


def weak_lru(maxsize: Optional[int] = 128, typed: bool = False):
    """
    LRU Cache decorator that keeps a weak reference to `self`
    Useful for caching methods in classes that may cause memory leaks if `functools.lru_cache` is used directly.
    From https://stackoverflow.com/a/68052994/16569836
    """
    def wrapper(func):

        @functools.lru_cache(maxsize, typed)
        def _func(_self, *args, **kwargs):
            return func(_self(), *args, **kwargs)

        @functools.wraps(func)
        def inner(self, *args, **kwargs):
            return _func(weakref.ref(self), *args, **kwargs)

        return inner

    return wrapper

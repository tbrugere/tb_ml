from typing import TypeVar, ParamSpec, Callable, Optional

T = TypeVar('T')
P = ParamSpec('P')


def take_annotation_from(this: Callable[P, T]) -> Callable[[Callable], Callable[P, T]]:
    """Inspired from https://stackoverflow.com/a/71262408/4948719"""
    def decorator(real_function: Callable) -> Callable[P, T]:
        real_function.__annotations__ = this.__annotations__
        return real_function #type: ignore 
    return decorator



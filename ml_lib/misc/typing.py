import types
import typing
from typing import TypeVar, ParamSpec, Callable, Optional, Any, Literal

T = TypeVar('T')
P = ParamSpec('P')
Q = ParamSpec('Q')


def take_annotation_from(this: Callable[P, T]) -> Callable[[Callable], Callable[P, T]]:
    """Inspired from https://stackoverflow.com/a/71262408/4948719"""
    def decorator(real_function: Callable) -> Callable[P, T]:
        real_function.__annotations__ = this.__annotations__
        return real_function #type: ignore 
    return decorator

# would be great but doesn't work yet
# def kwargs_passed_down_to(target: Callable[P, Any]) -> \
#         Callable[[Callable[Q, T]], Callable[Concatenate[Q, P], T]]:
#     raise NotImplementedError

def get_type_origin(t):
    """recursively looks at a type's __origin__ until we can't"""
    if not hasattr(t, "__origin__"):
        return t
    else: 
        return get_type_origin(t.__origin__)

def advanced_type_check(value, t):
    """Todo edit, to check for subscripted types for example (like list[int])"""

    origin = typing.get_origin(t)
    if t is None:
        return value is None
    if origin is None:
        return isinstance(value, t)
    if origin == typing.Annotated:
        new_t, *_ = typing.get_args(t)
        return advanced_type_check(value, new_t)
    if origin == typing.Union or origin == types.UnionType:
        for possibility in typing.get_args(t):
            if advanced_type_check(value, possibility): return True
        else: return False
    if origin == Literal:
        args = typing.get_args(t)
        return value in args
    return isinstance(value, origin)

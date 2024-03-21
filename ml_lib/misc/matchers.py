"""ABCs used to make match-case statements more readable
A lot of this is very dark magic, 
basically subclasses of Checker hijack insinsatnce calls:
they all define a check() classmethod, 
and A.check(x) is called when trying to check if isinstance(x, A)"""
from typing import Callable, Any

class CheckerMeta(type):
    check: Callable[[Any], bool]

    def __instancecheck__(self, instance):
        return self.check(instance)

class Checker(metaclass=CheckerMeta):
    def __init__(self):
        raise ValueError("Class with metaclass CheckerMeta shouldn't be instantiated")

    @classmethod
    def check(cls, instance) -> bool:
        del instance
        return False

class EmptySet(Checker):
    @classmethod
    def check(cls, instance):
        return isinstance(instance, set) and len(instance) == 0

# one day I'll make one that checks for regexes… with __match_args__... 
# might not work actually... we'll see

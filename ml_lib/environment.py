"""
This modules contains stuff on environments.

Environments are just a way for me to pass around a bunch of variables
(kinda like dictionaries)
in an implicit way.

What I did before:

```python
    def forward(self, x, return_loss=True, 
                return_mse=False, #added because I wanted to plot intermediate
                return_kl=False, #losses
                return_intermediate_encoding=False #added because 
                                    #I wanted to display it
                ):
        x_ = self.encode(x)
        kl = None
        if self.training:
            z, kl = self.reparameterize(x_, return_kl=true)
        else:
            z = self.reparameterize(x_)
        y = self.decode(z)
        
        if not return_loss:
            return y
        assert kl is not None, "computing loss, but not in training mode"
        mse = MSE(x, y)

        loss = mse + kl
        mse_ = [mse] if return_mse else []
        kl_ = [kl] if return_kl else []
        inter_enc = [z] if return_intermediate_encoding else []

        return (loss, *mse_, *kl_, *inter_enc)


```

What I do now:

```python
    def forward(self, x, return_loss=True):
        x_ = self.encode(x)
        kl=None
        if self.training:
            z, kl = self.reparameterize(x_, return_kl=true)
            self.env.record("kl", kl)
        else:
            z = self.reparameterize(x_, return_kl=true)
        self.env.record("z", z)

        y = self.decode(z)
        
        if not return_loss :
            return y

        mse = MSE(x)
        self.env.record("mse", mse)
        
        loss = mse + kl
        return loss

```
and if I want to get the value of kl, mse, 

## What it should be used for

- stuff that will be passed to hooks in the optimizer. in particular:
- stuff that should be logged (intermediate values, parts of losses)
- 

## What it shouldn t be used for

- Basically anything else. You don’t want to pull a 
- in particular, full loss should also be returned

Instead of passing around a humo
"""

from typing import TypeAlias, TypeVar, Any, Optional, Callable

from collections import defaultdict
from collections.abc import Collection
from dataclasses import dataclass, field
from inspect import signature, Parameter
from io import StringIO

Scope: TypeAlias = tuple[str, ...]

T = TypeVar("T") 

def scopevar_of_str(s: str) -> tuple[Scope, str] :
    path = s.split("/")
    return tuple(path[:-1]), path[-1]

def str_of_scopevar(scope: Scope, var: str) -> str:
    return "/".join((*scope, var))
    

class Environment():
    data: defaultdict[str, dict[Scope, Any]]

    def __init__(self):
        self.reset()

    def reset(self):
        self.data = defaultdict(dict)

    def record(self, key: str, value: Any, scope: Scope=()):
        self.data[key][scope] = value

    def record_dict(self, d: dict, scope: Scope = ()):
        for key, value in d.items():
            self.record(key, value, scope)

    def get(self, key:str, scope: Scope=()):
        """
        Returns the value for key `key` under scope `scope`.

        This function also searches subscopes of `scope`.
        The value with the highest (shortest) scope is returned

        Returns None if no value is found
        """
        values: dict[Scope, Any] = self.data[key]
        scope_len = len(scope)
        starts_with_scope = lambda s: s[:scope_len] == scope
        #otherwise return the item with the highest scope
        max_scope: Optional[Scope] = None
        best_item: Any = None
        for item_scope, item in values.items():
            if not starts_with_scope(item_scope):
                continue
            if max_scope is not None and len(item_scope) >= len(max_scope):
                continue 
            max_scope = item_scope
            best_item = item
        return best_item

    def contains(self, key:str, scope: Scope=()) -> bool:
        if key not in self.data:
            return False
        values = self.data[key]
        scope_len = len(scope)
        starts_with_scope = lambda s: s[:scope_len] == scope
        for item_scope in values:
            if starts_with_scope(item_scope):
                return True
        return False


    def __getattr__(self, key: str):
        return self.get(key)

    def __contains__(self, key:str):
        return self.contains(key)


    def run_function(self, f: Callable[..., T], 
                     record_result_as:str | tuple[str] | None = None, 
                     *args, **kwargs) -> T:
        """
        runs function f gathering its arguments from different sources
        (from higher to lower priority):

        - the given `args` and `kwargs`
        - variables with the same name in the environment
        - the default variables from the function
        """
        sig = signature(f)

        arguments = sig.bind_partial(*args, **kwargs)
        for param in sig.parameters.values():
            if (param.name not in arguments.arguments 
                    and param.name in self
                    and param.kind != Parameter.POSITIONAL_ONLY
                ):
                arguments.arguments[param.name] = self.get(param.name)
            

        f_args, f_kwargs = arguments.args, arguments.kwargs

        ret_val = f(*f_args, **f_kwargs)

        match record_result_as:
            case None:
                pass
            case (*record_names,):
                assert isinstance(ret_val, Collection), f"record_result_as is a tuple {record_result_as}, but only one value was returned by f"
                assert len(ret_val) == len(record_names), f"record_result_as is not the same length as the number of returned values {ret_val=}, {record_result_as=}"
                for key, value in zip(record_names, ret_val):
                    self.record(key=key, value=value)
            case str() as record_name:
                self.record(key=record_name, value=ret_val)

        return ret_val

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.data})"

    def __str__(self) -> str:
        sio = StringIO(f"{self.__class__.__name__} with\n")
        scoped_key_values: list[tuple[tuple[str, ...], Any]] = []
        for key, values in self.data.items():
            for scope, value in values.items():
                scoped_key_values.append((
                    (*scope, key),
                    value
                    ))
        scoped_key_values.sort()
        for scoped_key, value in scoped_key_values:
            sio.write(f' - {"/".join(scoped_key)}: {value}')

        return sio.getvalue()

    # no setattr. This is intended, there is no reason you should setattr
    # in a non-scoped environment

class HierarchicEnvironment(Environment):
    """An environment that has read-access to data from another 'parent' 
    environment
    """
    parent: Optional[Environment]

    def __init__(self, parent:Optional[Environment]):
        super().__init__()
        self.parent = parent
    
    def get(self, key:str, scope: Scope= ()):
        res = Environment.get(self, key, scope)
        if res is not None or self.parent is None:
            return res
        return self.parent.get(key, scope)

    def contains(self, key:str, scope: Scope = ()) -> bool:
        return Environment.contains(self, key, scope)\
                or (self.parent is not None 
                    and  self.parent.contains(key, scope))


@dataclass
class ScopedEnvironment():
    scope: Scope = ()
    environment: Environment = field(default_factory=Environment)

    def record(self, key:str, value: Any):
        self.environment.record(key, value, self.scope)

    def get(self, key:str):
        return self.environment.get(key, self.scope)

    def __setattr__(self, key: str, value: Any):
        if key in ('scope', 'environment'):
            self.__dict__[key] = value
        self.record(key, value)

    def __getattr__(self, key: str):
        return self.get(key)


class HasEnvironmentMixin():

    env: ScopedEnvironment

    def __init__(self):
        self.env = ScopedEnvironment() #set a dummy env

    def set_environment(self, env: Environment, scope: Scope=()):
        r"""
        Sets the environment for the class and all its subclasses,
        with the right (subclass) scopes 

        For example, if you have an object `a`. So that `a`, `a.b`, `a.b.c`
        and `a.d` all inherit HasEnvironmentMixin, 

        Then if you call 
        ```
        a.set_environment(env)
        ```

        Then `a`, `a.b`, `a.b.c` and `a.d`’s environments are all set to env.
        and 

        - the scope of `a` is set  to `()`
        - the scope of `a.b` is set  to `("b")`
        - the scope of `a.b.c` is set  to `("b", "c")`
        - the scope of `a.d` is set  to `("d")`
        """
        self.env = ScopedEnvironment(scope=scope, environment=env)

        for name, item in vars(self).items():
            if not isinstance(item, HasEnvironmentMixin):
                continue
            item.set_environment(env, (*scope, name))

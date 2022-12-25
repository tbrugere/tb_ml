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

from typing import TypeAlias, Any, Optional

from collections import defaultdict
from dataclasses import dataclass, field

Scope: TypeAlias = tuple[str, ...]

class Environment():


    data: defaultdict[str, dict[Scope, Any]]

    def __init__(self):
        self.reset()

    def reset(self):
        self.data = defaultdict(dict)

    def record(self, key: str, value: Any, scope: Scope=()):
        self.data[key][scope] = value

    def get(self, key:str, scope: Optional[Scope]=None):
        values: dict[Scope, Any] = self.data[key]
        if scope is not None:
            return values.get(scope, None)
        #otherwise return the item with the highest scope
        max_scope: Optional[Scope] = None
        best_item: Any = None
        for item_scope, item in values.items():
            if max_scope is None or len(item_scope) < len(max_scope):
                max_scope = item_scope
                best_item = item
        return best_item

    def __getattr__(self, key: str):
        return self.get(key)

    # no setattr. This is intended, there is no reason you should setattr
    # in a non-scoped environment
        

@dataclass
class ScopedEnvironment():
    scope: Scope = ()
    environment: Environment = field(default_factory=Environment)

    def record(self, key:str, value: Any):
        self.environment.record(key, value, self.scope)

    def get(self, key:str):
        return self.environment.get(key, self.scope)

    def __setattr__(self, key: str, value: Any):
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

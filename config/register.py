import inspect
import importlib

import logging  
import logging.config
logging.config.fileConfig("logging.conf")
logger = logging.getLogger()

GLOBAL_CONFIG = dict()

def register(cls:type):
    if cls.__name__ in GLOBAL_CONFIG:
        raise ValueError(f"{cls.__name__} already registered")
    if inspect.isfunction(cls):
        GLOBAL_CONFIG[cls.__name__] = cls
    if inspect.isclass(cls):
        GLOBAL_CONFIG[cls.__name__]= extract_schema(cls) #extract_schema(cls)
    else:
        raise ValueError(f'register {cls} failed')
    return cls

def extract_schema(cls:type):
    schame = dict()
    schame['_name'] = cls.__name__
    schame['_pymodule'] = importlib.import_module(cls.__module__)
    return schame

# def extract_schema(cls: type):
#     '''
#     Args:
#         cls (type),
#     Return:
#         Dict, 
#     '''
#     argspec = inspect.getfullargspec(cls.__init__)
#     arg_names = [arg for arg in argspec.args if arg != 'self']
#     num_defualts = len(argspec.defaults) if argspec.defaults is not None else 0
#     num_requires = len(arg_names) - num_defualts

#     schame = dict()
#     schame['_name'] = cls.__name__
#     schame['_pymodule'] = importlib.import_module(cls.__module__)
#     schame['_inject'] = getattr(cls, '__inject__', [])
#     schame['_share'] = getattr(cls, '__share__', [])

#     for i, name in enumerate(arg_names):
#         if name in schame['_share']:
#             assert i >= num_requires, 'share config must have default value.'
#             value = argspec.defaults[i - num_requires]
        
#         elif i >= num_requires:
#             value = argspec.defaults[i - num_requires]

#         else:
#             value = None 

#         schame[name] = value
        
#     return schame

# class Register:

#     def __init__(self, registry_name):
#         self._dict = {}
#         self._name = registry_name

#     def __setitem__(self, key, value):
#         if not callable(value):
#             raise Exception(f"Value of a Registry must be a callable!\nValue: {value}")
#         if key is None:
#             key = value.__name__
#         if key in self._dict:
#             logging.warning("Key %s already in registry %s." % (key, self._name))
#         self._dict[key] = value

#     def register(self, target):
#         """Decorator to register a function or class."""

#         def add(key, value):
#             self[key] = value
#             return value

#         if callable(target):
#             # @reg.register
#             return add(None, target)
#         # @reg.register('alias')
#         return lambda x: add(target, x)

#     def __getitem__(self, key):
#         return self._dict[key]

#     def __contains__(self, key):
#         return key in self._dict

#     def keys(self):
#         """key"""
#         return self._dict.keys()
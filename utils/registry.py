CLASS_REGISTRY = {}

def register_class(cls):
    CLASS_REGISTRY[cls.__name__] = cls
    return cls
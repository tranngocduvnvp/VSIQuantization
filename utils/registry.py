# Global registry for class registration
CLASS_REGISTRY = {}


def register_class(cls):
    """
    Decorator to register a class in the global registry.
    
    Used to register observers, quantizers, and other components so they
    can be instantiated by name at runtime.
    
    Args:
        cls: Class to register
        
    Returns:
        cls: The registered class (unchanged)
        
    Example:
        @register_class
        class MyQuantizer:
            pass
            
        # Later:
        quantizer = CLASS_REGISTRY["MyQuantizer"]()
    """
    CLASS_REGISTRY[cls.__name__] = cls
    return cls
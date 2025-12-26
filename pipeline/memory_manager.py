"""
Memory Manager - Critical for running multiple large models on 36GB RAM
Ensures only one model is loaded at a time with proper cleanup.
"""
import gc
import torch
from typing import Optional, Any
from functools import wraps
import time


class MemoryManager:
    """
    Manages GPU/MPS memory by ensuring only one heavy model is loaded at a time.
    Uses aggressive garbage collection between model loads.
    """
    
    _instance = None
    _current_model: Optional[str] = None
    _model_cache: dict = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_memory_usage(cls) -> dict:
        """Get current memory usage statistics."""
        stats = {
            "allocated": 0,
            "reserved": 0,
            "device": "unknown"
        }
        
        if torch.backends.mps.is_available():
            stats["device"] = "mps"
            stats["allocated"] = torch.mps.current_allocated_memory() / (1024**3)  # GB
            stats["reserved"] = torch.mps.driver_allocated_memory() / (1024**3)  # GB
        elif torch.cuda.is_available():
            stats["device"] = "cuda"
            stats["allocated"] = torch.cuda.memory_allocated() / (1024**3)  # GB
            stats["reserved"] = torch.cuda.memory_reserved() / (1024**3)  # GB
        
        return stats
    
    @classmethod
    def clear_memory(cls, verbose: bool = True) -> None:
        """Aggressively clear GPU/MPS memory."""
        if verbose:
            before = cls.get_memory_usage()
            print(f"🧹 Clearing memory... (Before: {before['allocated']:.2f}GB allocated)")
        
        # Clear Python garbage
        gc.collect()
        
        # Clear PyTorch caches
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Additional garbage collection passes
        for _ in range(3):
            gc.collect()
        
        if verbose:
            after = cls.get_memory_usage()
            print(f"✅ Memory cleared (After: {after['allocated']:.2f}GB allocated)")
    
    @classmethod
    def unload_model(cls, model_name: str) -> None:
        """Unload a specific model from memory."""
        if model_name in cls._model_cache:
            print(f"📤 Unloading model: {model_name}")
            model = cls._model_cache.pop(model_name)
            
            # Move to CPU first (helps with MPS cleanup)
            if hasattr(model, 'to'):
                try:
                    model.to('cpu')
                except:
                    pass
            
            # Delete the model
            del model
            
            # Clear memory
            cls.clear_memory()
            
            if cls._current_model == model_name:
                cls._current_model = None
    
    @classmethod
    def unload_all(cls) -> None:
        """Unload all cached models."""
        print("📤 Unloading all models...")
        model_names = list(cls._model_cache.keys())
        for name in model_names:
            cls.unload_model(name)
        cls.clear_memory()
    
    @classmethod
    def register_model(cls, name: str, model: Any) -> None:
        """Register a loaded model for tracking."""
        cls._model_cache[name] = model
        cls._current_model = name
        print(f"📥 Registered model: {name}")
        
        # Log memory usage
        stats = cls.get_memory_usage()
        print(f"   Memory: {stats['allocated']:.2f}GB allocated")
    
    @classmethod
    def prepare_for_model(cls, model_name: str, required_gb: float = 8.0) -> bool:
        """
        Prepare memory for loading a new model.
        Unloads other models if necessary.
        """
        print(f"\n{'='*60}")
        print(f"🔄 Preparing to load: {model_name} (needs ~{required_gb}GB)")
        print(f"{'='*60}")
        
        # Unload all other models first
        cls.unload_all()
        
        # Wait a moment for memory to settle
        time.sleep(0.5)
        
        # Check available memory
        stats = cls.get_memory_usage()
        print(f"✅ Ready to load {model_name}")
        print(f"   Current memory: {stats['allocated']:.2f}GB allocated")
        
        return True


def model_loader(model_name: str, required_gb: float = 8.0):
    """
    Decorator for model loading functions.
    Automatically manages memory before loading.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Check if model is already loaded
            if model_name in MemoryManager._model_cache:
                print(f"✅ {model_name} already loaded, reusing...")
                return MemoryManager._model_cache[model_name]
            
            # Prepare memory
            MemoryManager.prepare_for_model(model_name, required_gb)
            
            # Load the model
            model = func(*args, **kwargs)
            
            # Register it
            MemoryManager.register_model(model_name, model)
            
            return model
        return wrapper
    return decorator


# Convenience function for use in pipeline
def cleanup():
    """Quick cleanup function."""
    MemoryManager.unload_all()

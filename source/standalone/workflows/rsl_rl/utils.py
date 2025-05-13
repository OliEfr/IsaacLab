import torch
import numpy as np
from functools import wraps
import traceback
import logging

def setup_tensor_tracking():
    """Set up logging for tensor operations"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('tensor_tracker')
    return logger

class TensorTracker:
    def __init__(self):
        self.logger = setup_tensor_tracking()
        
    def __enter__(self):
        # Patch torch.Tensor methods to track device changes
        self.original_to = torch.Tensor.to
        self.original_cpu = torch.Tensor.cpu
        self.original_numpy = torch.Tensor.numpy
        
        @wraps(torch.Tensor.to)
        def tracked_to(tensor, *args, **kwargs):
            result = self.original_to(tensor, *args, **kwargs)
            if args and isinstance(args[0], (torch.device, str)):
                new_device = str(args[0])
                if str(tensor.device) != new_device:
                    stack = traceback.extract_stack()
                    caller = stack[-2]  # Get caller information
                    self.logger.info(f"Device change detected: {tensor.device} -> {new_device} at {caller.filename}:{caller.lineno}")
            return result
            
        @wraps(torch.Tensor.cpu)
        def tracked_cpu(tensor):
            result = self.original_cpu(tensor)
            if tensor.device.type != 'cpu':
                stack = traceback.extract_stack()
                caller = stack[-2]
                self.logger.info(f"Moving tensor to CPU from {tensor.device} at {caller.filename}:{caller.lineno}")
            return result
            
        @wraps(torch.Tensor.numpy)
        def tracked_numpy(tensor):
            stack = traceback.extract_stack()
            caller = stack[-2]
            self.logger.info(f"Converting tensor to numpy at {caller.filename}:{caller.lineno}")
            return self.original_numpy(tensor)
        
        torch.Tensor.to = tracked_to
        torch.Tensor.cpu = tracked_cpu
        torch.Tensor.numpy = tracked_numpy
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original methods
        torch.Tensor.to = self.original_to
        torch.Tensor.cpu = self.original_cpu
        torch.Tensor.numpy = self.original_numpy
"""Device detection utilities"""


def get_device():
    """Detect and return the best available device (GPU or CPU)"""
    try:
        import torch
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return 'cpu'
    except Exception:
        # If torch is not available, default to CPU
        return 'cpu'


import os
import json
from loguru import logger
from typing import List, Union, Tuple, Dict


def SetOptions(options, backend) -> Tuple[Union[List[str], Tuple[str, ...], Dict[str, str]], str]:
    if options is not None:
        options = options
    if backend is not None:
        backend = backend
    if options is None and backend is None:
        options = ('-O1', '-v') 
        backend = 'nvrtc'
    logger.info(f"WCV: Using compiler options: {options} with backend: {backend}")

    return options, backend

def GetWCVEnv():
    options_str = os.getenv('WCV_COMPILE_OPTIONS', None)
    backend_str = os.getenv('WCV_BACKEND', None)
    
    if options_str:
        try:
            options = json.loads(options_str)
        except json.JSONDecodeError:
            logger.info(f"WCV: Invalid JSON in WCV_COMPILE_OPTIONS, setting to None")
            options = None
    else:
        options = None
    
    if backend_str:
        try:
            backend = backend_str.strip()
            if not backend:
                logger.info(f"WCV: Empty WCV_BACKEND value, setting to None")
                backend = None
        except Exception as e:
            logger.info(f"WCV: Error processing WCV_BACKEND: {e}, setting to None")
            backend = None
    else:
        backend = None
    
    return options, backend

options, backend = SetOptions(*GetWCVEnv())



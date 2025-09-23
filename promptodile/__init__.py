"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
This material is based upon work supported by the Department of the Air Force under Air Force Contract No. FA8702-15-D-0001 or FA8702-25-D-B002. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the Department of the Air Force.
© 2025 Massachusetts Institute of Technology.

Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
The software/firmware is provided to you on an As-Is basis
Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other than as specifically authorized by the U.S. Government may violate any copyrights that exist in this work.
"""

import os
import random
import logging


# Set the RANDOM_SEED environment variable to set your own seed.
random.seed(os.getenv('RANDOM_SEED', 42))

LOGGING_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

logging.basicConfig(
    level=logging.INFO,
    format=LOGGING_FORMAT
)

def configure_logging(level: int=logging.INFO, format: str | None=None):
    """Added for convenience to set logging levels.
    
    logging.CRITICAL = 50
    logging.FATAL = 50      # Alias for CRITICAL
    logging.ERROR = 40
    logging.WARNING = 30
    logging.WARN = 30       # Deprecated alias for WARNING
    logging.INFO = 20
    logging.DEBUG = 10
    logging.NOTSET = 0
    """
    if format is None:
        format = LOGGING_FORMAT
    
    logging.basicConfig(level=level, format=format, force=True)
        
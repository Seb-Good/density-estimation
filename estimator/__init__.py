# Import 3rd party libraries
import os

# Set working directory
WORKING_PATH = (
    os.path.dirname(
        os.path.dirname(
            os.path.realpath(__file__)
        )
    )
)

# Set data directory
DATA_PATH = os.path.join(WORKING_PATH, 'data')

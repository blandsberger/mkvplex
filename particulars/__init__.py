"""mkvplex implementation package.

The public facade re-exports the historical single-file API while implementation
now lives in logical modules. New code should import the specific module it uses.
"""

from .models import *
from .common import *
from .discovery import *
from .naming import *
from .media import *
from .discs import *
from .tmdb import *
from .fsops import *
from .movie import *
from .collection import *
from .volume import *
from .tvplan import *
from .tv import *
from .cli import *

from .models import VERSION

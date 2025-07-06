"""HyperCubeX EAN – Core package.

Only re-export primitives needed by the rest of the system to avoid
leaking internal modules.
"""

from .neuron import Neuron  # noqa: F401
from .assembly import Assembly  # noqa: F401
from .network import Network  # noqa: F401
from .strategies import PruningStrategy, CompetitiveEnergyPruning  # noqa: F401
from .scheduler import Scheduler, SimpleScheduler  # noqa: F401
from .workspace import GlobalWorkspace  # noqa: F401
# Teachers live in separate namespace but convenient re-export
from hx_teachers import RecursiveTeacher  # noqa: F401

"""by lyuwenyu
"""


from .rtdetr import *

from .hybrid_encoder import *
from .rtdetr_decoder import *
from .rtdetr_postprocessor import *
from .rtdetr_criterion import *

from .matcher import *

# Optional second-host adapter modules. Importing them registers distinct
# classes; native configs continue to instantiate the untouched official
# RTDETRTransformer, SetCriterion and RTDETRPostProcessor classes.
import sys as _sys
from . import auxiliary_geometry as _auxiliary_geometry
from . import competition_mapping as _competition_mapping

# Existing artifact and configuration identifiers remain valid while the
# physical files use the same functional names as the main host.
_sys.modules[__name__ + '.hilor_ufd'] = _auxiliary_geometry
_sys.modules[__name__ + '.hilor_qccr'] = _competition_mapping
from . import candidate_value as _candidate_value
_sys.modules[__name__ + '.hilor_umr'] = _candidate_value

from .hilor_gca import *
from .hilor_postprocessor import *

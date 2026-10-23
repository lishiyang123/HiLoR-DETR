"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""


import sys as _sys

# Functional responsibilities within the two method stages:
# GSA: auxiliary geometry and classification-role adjustment.
# CCVR: competition-conditioned mapping and candidate-value reranking.
# Retain import aliases for existing callers and serialized class identifiers;
# these identifiers do not define additional independent method contributions.
from . import competition_mapping as qccr
_sys.modules[__name__ + ".qccr"] = qccr
from . import candidate_value as umr
_sys.modules[__name__ + ".umr"] = umr
from . import auxiliary_geometry as criterion_ufd
_sys.modules[__name__ + ".criterion_ufd"] = criterion_ufd
from . import classification_role as criterion_lcct
_sys.modules[__name__ + ".criterion_lcct"] = criterion_lcct

from .deim import DEIM

from .matcher import HungarianMatcher
from .hybrid_encoder import HybridEncoder
from .dfine_decoder import DFINETransformer
from .rtdetrv2_decoder import RTDETRTransformerv2

from .postprocessor import PostProcessor
from .deim_criterion import DEIMCriterion

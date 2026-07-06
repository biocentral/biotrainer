from enum import Enum
from typing import Optional, Union

from .core import AutoEvalFramework
from .frameworks import PBCSupervisedFramework, FLIPFramework, PGYMFramework, PBCZeroShotContactFramework, \
    PBCSupervisedContactFramework


class AvailableFramework(Enum):
    FLIP = "FLIP"
    PBC_SUPERVISED = "PBC_SUPERVISED"
    PGYM = "PGYM"
    PBC_ZEROSHOT_CONTACT = "PBC_ZEROSHOT_CONTACT"
    PBC_SUPERVISED_CONTACT = "PBC_SUPERVISED_CONTACT"


available_frameworks = {AvailableFramework.FLIP: FLIPFramework(),
                        AvailableFramework.PBC_SUPERVISED: PBCSupervisedFramework(),
                        AvailableFramework.PGYM: PGYMFramework(),
                        AvailableFramework.PBC_ZEROSHOT_CONTACT: PBCZeroShotContactFramework(),
                        AvailableFramework.PBC_SUPERVISED_CONTACT: PBCSupervisedContactFramework(),
                        }


def framework_factory(framework_name: Union[str, AvailableFramework]) -> Optional[AutoEvalFramework]:
    try:
        av_framework = AvailableFramework(framework_name.upper()) if isinstance(framework_name, str) else framework_name
        return available_frameworks.get(av_framework, None)
    except ValueError:
        return None

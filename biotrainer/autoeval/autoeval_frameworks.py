from enum import Enum
from typing import Optional, Union

from .core import AutoEvalFramework
from .frameworks import PBCSupervisedFramework, FLIPFramework, PGYMFramework, PBCZeroShotContactFramework


class AvailableFramework(Enum):
    FLIP = "FLIP"
    PBC = "PBC"
    PGYM = "PGYM"
    ZEROSHOT_CONTACT = "ZEROSHOT_CONTACT"


available_frameworks = {AvailableFramework.FLIP: FLIPFramework(),
                        AvailableFramework.PBC: PBCSupervisedFramework(),
                        AvailableFramework.PGYM: PGYMFramework(),
                        AvailableFramework.ZEROSHOT_CONTACT: PBCZeroShotContactFramework()
                        }


def framework_factory(framework_name: Union[str, AvailableFramework]) -> Optional[AutoEvalFramework]:
    try:
        av_framework = AvailableFramework(framework_name.upper()) if isinstance(framework_name, str) else framework_name
        return available_frameworks.get(av_framework, None)
    except ValueError:
        return None
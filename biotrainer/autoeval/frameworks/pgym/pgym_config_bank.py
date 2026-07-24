from typing import Dict, Any
from biotrainer_core.data_classes.autoeval import AutoEvalTask

from ...core import AutoEvalConfigBank


class PGYMConfigBank(AutoEvalConfigBank):

    def get_task_config(self, task: AutoEvalTask) -> Dict[str, Any]:
        return {}

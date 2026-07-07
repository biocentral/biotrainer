from typing import Dict, Any
from biotrainer_core.data_classes.autoeval import AutoEvalTask

from ...core import AutoEvalConfigBank


class ContactConfigBank(AutoEvalConfigBank):  # can be common for zeroshot and supervised contact tasks!

    def get_task_config(self, task: AutoEvalTask) -> Dict[str, Any]:
        return {}

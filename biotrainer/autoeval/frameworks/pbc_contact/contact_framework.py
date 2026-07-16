from ...core import AutoEvalFramework, AutoEvalMode
from .contact_config_bank import ContactConfigBank
from .contact_data_handler import ZeroShotContactDataHandler, SupervisedContactDataHandler


class PBCZeroShotContactFramework(AutoEvalFramework):
    @staticmethod
    def get_name():
        return "PBC_ZEROSHOT_CONTACT"

    @staticmethod
    def get_mode() -> AutoEvalMode:
        return AutoEvalMode.ZERO_SHOT_CONTACT

    def get_data_handler(self):
        return ZeroShotContactDataHandler()

    def get_config_bank(self):
        return ContactConfigBank()


class PBCSupervisedContactFramework(AutoEvalFramework):
    @staticmethod
    def get_name():
        return "PBC_SUPERVISED_CONTACT"

    @staticmethod
    def get_mode() -> AutoEvalMode:
        return AutoEvalMode.SUPERVISED_CONTACT_ATTENTION

    def get_data_handler(self):
        return SupervisedContactDataHandler()

    def get_config_bank(self):
        return ContactConfigBank()
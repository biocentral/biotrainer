from junban import PipelineStep

from ..pipeline_context import BiotrainerPipelineContext

from ...validations import InputValidator


class InputValidationStep(PipelineStep[BiotrainerPipelineContext]):

    def _check_entry_assumptions(self, context: BiotrainerPipelineContext) -> bool:
        return True

    def _check_exit_assumptions(self, context: BiotrainerPipelineContext) -> bool:
        return True

    def get_start_message(self) -> str:
        return "Validating input..."

    def get_end_message(self) -> str:
        return "Input validation complete!"

    def _execute(self, context: BiotrainerPipelineContext) -> BiotrainerPipelineContext:
        if context.config.get("validate_input", True):
            protocol = context.config["protocol"]
            input_validator = InputValidator(protocol=protocol)
            validated_input_data = input_validator.validate(context.input_data)
            # No errors - set validated input data as input data
            context.input_data = validated_input_data
            # Log hash2id for remapping predictions
            context.hash2id = {data_point.get_hash(): data_point.seq_id for data_point in validated_input_data}
        return context

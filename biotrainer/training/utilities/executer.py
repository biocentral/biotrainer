from pathlib import Path
from junban import Pipeline
from typing import Union, Dict, Any, Optional, List
from biotrainer_core.data_classes import BiotrainerModelResult, SequenceData

from ..trainers import Trainer
from ..config import Configurator
from ..trainers.pipeline_context import BiotrainerPipelineContext
from ..output_files import OutputManager, output_observer_factory, BiotrainerOutputObserver

from ...shared.logging import clear_logging


def parse_config_file_and_execute_run(config: Union[str, Path, Dict[str, Any]],
                                      input_data: Optional[List[SequenceData]] = None,
                                      custom_pipeline: Optional[Pipeline[BiotrainerPipelineContext]] = None,
                                      custom_output_observers: Optional[List[BiotrainerOutputObserver]] = None,
                                      write_to_file: Optional[bool] = True) -> BiotrainerModelResult:
    # Verify config via configurator
    configurator = None
    if isinstance(config, str):
        configurator = Configurator.from_config_path(config)
    elif isinstance(config, Path):
        configurator = Configurator.from_config_path(str(config))
    elif isinstance(config, dict):
        configurator = Configurator.from_config_dict(config)

    assert configurator is not None, f"Config could not be read, incorrect type: {type(config)}"

    config = configurator.get_verified_config(input_data=input_data, ignore_file_checks=False)

    if input_data is not None and len(input_data) > 0:
        config["input_data"] = input_data
        assert "input_file" not in config, (f"Cannot have both input_file and input_data, "
                                            f"should have been caught by the config verification!")

    output_dir = Path(config["output_dir"])

    output_observers = output_observer_factory(output_dir=output_dir, config=config)
    if custom_output_observers and len(custom_output_observers) > 0:
        output_observers.extend(custom_output_observers)

    output_manager = OutputManager(observers=output_observers)
    output_manager.add_config(config=config)

    trainer: Trainer
    if custom_pipeline:
        output_manager.update_derived_values(custom_pipeline=True)

    # Run biotrainer pipeline
    trainer = Trainer(config=config,
                      output_manager=output_manager,
                      custom_pipeline=custom_pipeline
                      )

    output_manager = trainer.run()

    # Save output_variables in out.yml
    if write_to_file:
        output_manager.write_to_file(output_dir=Path(output_manager.model_result.config["log_dir"]))

    output_result = output_manager.model_result

    clear_logging()

    return output_result

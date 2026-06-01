from typing import Optional
from biotrainer_core.data_classes import Protocol

from .solver import Solver
from .gp_solver import GPSolver
from .residue_solvers import ResidueClassificationSolver, ResidueRegressionSolver
from .residues_solvers import ResiduesClassificationSolver, ResiduesRegressionSolver
from .sequence_solvers import SequenceClassificationSolver, SequenceRegressionSolver

from ...shared import METRIC_CALCULATORS

__SOLVERS = {
    Protocol.residue_to_class: ResidueClassificationSolver,
    Protocol.residue_to_value: ResidueRegressionSolver,
    Protocol.residues_to_class: ResiduesClassificationSolver,
    Protocol.residues_to_value: ResiduesRegressionSolver,
    Protocol.sequence_to_class: SequenceClassificationSolver,
    Protocol.sequence_to_value: SequenceRegressionSolver,
}


def get_solver(protocol: Protocol, name: str,
               network: Optional = None, optimizer: Optional = None, loss_function: Optional = None,
               device: Optional = None, number_of_epochs: Optional = None,
               patience: Optional = None, epsilon: Optional = None, output_manager: Optional = None,
               log_dir: Optional = None, n_classes: Optional[int] = 0,
               **kwargs
               ) -> Solver:
    if network.__class__.__name__ == "GPModelAdapter":
        solver_class = GPSolver
    else:
        solver_class = __SOLVERS.get(protocol)
    metrics_calc = get_metrics_calculator(protocol=protocol, device=device, n_classes=n_classes)

    if not solver_class:
        raise NotImplementedError
    else:
        return solver_class(
            split_name=name, protocol=protocol,
            network=network, optimizer=optimizer, loss_function=loss_function, metrics_calculator=metrics_calc,
            device=device, number_of_epochs=number_of_epochs,
            patience=patience, epsilon=epsilon, output_manager=output_manager,
            log_dir=log_dir, n_classes=n_classes
        )


def get_metrics_calculator(protocol: Protocol, device: Optional = None, n_classes: Optional[int] = 0):
    metrics_calc = METRIC_CALCULATORS.get(protocol)

    if not metrics_calc:
        raise NotImplementedError
    else:
        return metrics_calc(device=device, n_classes=n_classes)


__all__ = [
    'Solver',
    'get_solver',
    'get_metrics_calculator',
]

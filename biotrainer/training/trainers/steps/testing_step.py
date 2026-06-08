from typing import List
from biotrainer_core.data_classes import Protocol, BiotrainerPrediction, BiotrainerInferenceResult

from .training_factory import TrainingFactory

from junban import PipelineStep

from ..pipeline_context import BiotrainerPipelineContext

from ...solvers import get_metrics_calculator
from ...validations import SanityCheckerForTestSets

from ....shared import get_logger, Bootstrapper

logger = get_logger(__name__)


class TestingStep(PipelineStep[BiotrainerPipelineContext]):

    def _check_entry_assumptions(self, context: BiotrainerPipelineContext) -> bool:
        assert context.test_datasets is not None, f"test_datasets cannot be None at the testing step!"
        assert context.best_split is not None, f"best_split cannot be None at the testing step!"
        return True

    def _check_exit_assumptions(self, context: BiotrainerPipelineContext) -> bool:
        return True

    def get_start_message(self) -> str:
        return "Testing model..."

    def get_end_message(self) -> str:
        return "Testing complete!"

    @staticmethod
    def _do_and_log_evaluation(context: BiotrainerPipelineContext, solver, test_loader,
                               test_set_id: str) -> BiotrainerInferenceResult:
        # re-initialize the model to avoid any undesired information leakage and only load checkpoint weights
        solver.load_checkpoint(resume_training=False)
        test_results = solver.inference(test_loader, calculate_test_metrics=True)

        if context.config.get("save_split_ids", False):
            test_results_to_log = test_results.revert_mappings(protocol=context.config["protocol"],
                                                               class_int2str=context.target_manager.class_int2str)
            context.output_manager.add_test_result(test_set_id=test_set_id,
                                                   inference_result=test_results_to_log)
        else:
            test_results_to_log = test_results.omit_predictions()
            context.output_manager.add_test_result(test_set_id=test_set_id,
                                                   inference_result=test_results_to_log)

        logger.info(f"Test set {test_set_id} metrics: {test_results.metrics}")
        return test_results

    @staticmethod
    def _do_and_log_prediction(context: BiotrainerPipelineContext, solver, pred_loader):
        protocol = context.config["protocol"]
        class_int2str = context.target_manager.class_int2str
        # re-initialize the model to avoid any undesired information leakage and only load checkpoint weights
        solver.load_checkpoint(resume_training=False)

        # Monte Carlo Dropout Prediction Output only enabled for per-sequence protocols at the moment
        if protocol in Protocol.per_sequence_protocols() and solver.model_has_dropout():
            mcd_results: List[BiotrainerPrediction] = solver.inference_monte_carlo_dropout(pred_loader,
                                                                                           n_forward_passes=30)
            predictions = [result.revert_mappings(protocol=protocol, class_int2str=class_int2str) for result in
                           mcd_results]
        else:
            pred_results = solver.inference(pred_loader, calculate_test_metrics=False)
            pred_results = pred_results.revert_mappings(protocol=protocol, class_int2str=class_int2str)
            predictions = pred_results.predictions

        # Remap hashes to actual ids
        predictions = [pred.replace_seq_id(context.hash2id.get(pred.seq_id, pred.seq_id)) for pred in predictions]
        context.output_manager.add_predictions(predictions=predictions)

        logger.info(f"Calculated predictions for {len(context.prediction_dataset)} samples!")
        return predictions

    @staticmethod
    def _do_and_log_bootstrapping_evaluation(context: BiotrainerPipelineContext,
                                             metrics_calculator,
                                             test_results: BiotrainerInferenceResult,
                                             test_loader, test_set_id: str):
        logger.info(f'Running bootstrapping evaluation on the best model for test set ({test_set_id})')
        bootstrapped_metrics = Bootstrapper.bootstrap(protocol=context.config["protocol"],
                                                      device=context.config["device"],
                                                      bootstrapping_iterations=context.config[
                                                          "bootstrapping_iterations"],
                                                      metrics_calculator=metrics_calculator,
                                                      predictions=test_results.predictions,
                                                      test_loader=test_loader)
        context.output_manager.add_test_result(test_set_id=test_set_id,
                                               bootstrapped_metrics=bootstrapped_metrics)
        logger.info(f'Bootstrapping results for test set ({test_set_id}): {bootstrapped_metrics}')

    def _execute(self, context: BiotrainerPipelineContext) -> BiotrainerPipelineContext:
        # TESTING
        test_datasets = context.test_datasets
        best_split = context.best_split

        finetuning = "finetuning_config" in context.config
        for test_set_id, test_dataset in test_datasets.items():
            logger.info('Running final evaluation on the best model')
            test_dataset_embeddings = TrainingFactory.create_dataset(context=context,
                                                                     split=test_dataset,
                                                                     mode="test",
                                                                     finetuning=finetuning)
            test_loader = TrainingFactory.create_dataloader(context=context, dataset=test_dataset_embeddings,
                                                            hyper_params=best_split.hyper_params, finetuning=finetuning)
            test_results = self._do_and_log_evaluation(context=context,
                                                       solver=best_split.solver,
                                                       test_loader=test_loader,
                                                       test_set_id=test_set_id)

            # ADDITIONAL EVALUATION
            metrics_calculator = get_metrics_calculator(protocol=context.config["protocol"],
                                                        device=context.config["device"],
                                                        n_classes=context.n_classes)
            # BOOTSTRAPPING
            if context.config["bootstrapping_iterations"] > 0:
                self._do_and_log_bootstrapping_evaluation(context=context,
                                                          metrics_calculator=metrics_calculator,
                                                          test_results=test_results,
                                                          test_loader=test_loader,
                                                          test_set_id=test_set_id)

            # SANITY CHECKER
            if context.config.get("sanity_check", True):
                baseline_test_dataset = context.baseline_test_datasets[test_set_id]
                baseline_test_dataset_embeddings = TrainingFactory.create_dataset(context=context,
                                                                                  split=baseline_test_dataset,
                                                                                  mode="test",
                                                                                  finetuning=False)
                baseline_test_loader = TrainingFactory.create_dataloader(context=context,
                                                                         dataset=baseline_test_dataset_embeddings,
                                                                         hyper_params=best_split.hyper_params,
                                                                         finetuning=False)
                sanity_checker = SanityCheckerForTestSets(training_config=context.config,
                                                          n_classes=context.n_classes,
                                                          n_features=context.n_features,
                                                          train_dataset=context.train_dataset,
                                                          val_dataset=context.val_dataset,
                                                          test_dataset=baseline_test_dataset,
                                                          test_loader=baseline_test_loader,
                                                          metrics_calculator=metrics_calculator,
                                                          test_results=test_results,
                                                          class_weights=context.class_weights,
                                                          mode="warn")
                baseline_results, warnings = sanity_checker.check_test_results(test_set_id=test_set_id)
                if baseline_results is not None and len(baseline_results) > 0:
                    context.output_manager.add_test_result(test_set_id=test_set_id,
                                                           baselines=baseline_results)
                if len(warnings) > 0:
                    context.output_manager.add_test_result(test_set_id=test_set_id,
                                                           sanity_check_warnings=warnings)

        # PREDICTION
        prediction_dataset = context.prediction_dataset
        if prediction_dataset and len(prediction_dataset) > 0:
            logger.info(f'Calculating predictions for {len(prediction_dataset)} samples!')
            pred_dataset_embeddings = TrainingFactory.create_dataset(context=context,
                                                                     split=prediction_dataset, mode="pred")
            pred_loader = TrainingFactory.create_dataloader(context=context, dataset=pred_dataset_embeddings,
                                                            hyper_params=best_split.hyper_params, finetuning=finetuning)

            _ = self._do_and_log_prediction(context=context,
                                            solver=best_split.solver,
                                            pred_loader=pred_loader
                                            )

        return context

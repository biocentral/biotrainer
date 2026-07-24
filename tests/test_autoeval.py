import os
import unittest
import tempfile

from pathlib import Path
from biotrainer.autoeval import AutoEval
from biotrainer_core.data_classes import ZeroShotMethod
from biotrainer.bioengineer import BioEngineer, BioEngineerBaseline


class AutoevalTests(unittest.TestCase):

    @unittest.skipUnless(os.getenv('CI'), "Slow test - only run in CI")
    def test_autoeval_pbc_supervised_ohe(self):
        """ Checks that autoeval pipeline runs correctly with one hot encoding """
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            print("Starting AutoEval pipeline...")

            autoeval = AutoEval(embedder_name="one_hot_encoding",
                                output_dir=tmp_dir_name,
                                min_seq_length=10,
                                max_seq_length=450)
            report = autoeval.pbc_supervised().run()

            self.assertTrue(report is not None)
            self.assertTrue(len(report.supervised_results) > 0)

    def test_autoeval_zeroshot_contact_baseline(self):
        """ Checks that autoeval pipeline runs correctly with zero-shot contact baseline """
        TEST_CONTACT_STORAGE = Path(__file__).parent / "test_input_files"

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            print("Starting AutoEval pipeline...")

            bio_engineer = BioEngineer.from_baseline(baseline=BioEngineerBaseline.RANDOM_BASELINE)
            autoeval = AutoEval(embedder_name="bioengineer_random_baseline",
                                output_dir=tmp_dir_name,
                                custom_bioengineer=bio_engineer,
                                custom_storage_path=TEST_CONTACT_STORAGE, )
            report = autoeval.pbc_zeroshot_contact(zero_shot_method=ZeroShotMethod.JACOBIAN_CONTACT).run()

            self.assertTrue(report is not None)
            self.assertTrue(len(report.zeroshot_contact_results) > 0)


    @unittest.skip(reason="Large test that should only be executed on demand")
    def test_autoeval_zeroshot_contact_ESM2_8M_UR50D(self):
        """ Checks that autoeval pipeline runs correctly with zero-shot contact ESM2_8M_UR50D """
        TEST_CONTACT_STORAGE = Path(__file__).parent / "test_input_files"

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            print("Starting AutoEval pipeline...")

            autoeval = AutoEval(embedder_name="facebook/esm2_t6_8M_UR50D",
                                output_dir=tmp_dir_name,
                                custom_storage_path=TEST_CONTACT_STORAGE)
            report = autoeval.pbc_zeroshot_contact(zero_shot_method=ZeroShotMethod.JACOBIAN_CONTACT).run()

            self.assertTrue(report is not None)
            self.assertTrue(len(report.zeroshot_contact_results) > 0)

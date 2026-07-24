import torch
import unittest
import tempfile
import numpy as np

from copy import deepcopy
from biotrainer_core.data_classes import SequenceData, BiotrainerModelResult

from biotrainer.training import BiotrainerModel

s2c_config = {'protocol': 'sequence_to_class',
              'model_choice': 'FNN',
              'embedder_name': 'one_hot_encoding',
              'input_file': "test_input_files/scl_subset/scl_rand.fasta"}


class HashingTests(unittest.TestCase):

    def test_hash_is_same(self):
        """ Same input data should always result in the same hash"""
        input_data = [
            SequenceData(seq_id="Seq1", seq="MMALSLALM", attributes={"TARGET": "Membrane", "SET": "train"},
                         embedding=[1, 2, 3]),
            SequenceData(seq_id="Seq2", seq="PRTEIN", attributes={"TARGET": "Membrane", "SET": "train"},
                         embedding=[4, 5, 6]),
            SequenceData(seq_id="Seq3", seq="PRT", attributes={"TARGET": "Soluble", "SET": "train"},
                         embedding=[7, 8, 9]),
            SequenceData(seq_id="Seq4", seq="SEQWENCE", attributes={"TARGET": "Membrane", "SET": "val"},
                         embedding=[10, 11, 12]),
            SequenceData(seq_id="Seq5", seq="PRTE", attributes={"TARGET": "Soluble", "SET": "val"},
                         embedding=[13, 14, 15]),
            SequenceData(seq_id="Seq6", seq="MMALSM", attributes={"TARGET": "Membrane", "SET": "test"},
                         embedding=torch.tensor([16, 17, 18])),
            SequenceData(seq_id="Seq7", seq="PRSEQ", attributes={"TARGET": "Soluble", "SET": "test"},
                         embedding=np.array([19, 20, 21])),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config = deepcopy(s2c_config)
            config.pop("input_file")
            result1 = BiotrainerModel().train(config=config, input_data=input_data)
            result2 = BiotrainerModel().train(config=config, input_data=input_data)
            self.assertEqual(result1.derived_values.model_hash, result2.derived_values.model_hash)

    def test_hash_is_different_config(self):
        """ Different config should result in different hash """
        input_data = [
            SequenceData(seq_id="Seq1", seq="MMALSLALM", attributes={"TARGET": "Membrane", "SET": "train"},
                         embedding=[1, 2, 3]),
            SequenceData(seq_id="Seq2", seq="PRTEIN", attributes={"TARGET": "Membrane", "SET": "train"},
                         embedding=[4, 5, 6]),
            SequenceData(seq_id="Seq3", seq="PRT", attributes={"TARGET": "Soluble", "SET": "train"},
                         embedding=[7, 8, 9]),
            SequenceData(seq_id="Seq4", seq="SEQWENCE", attributes={"TARGET": "Membrane", "SET": "val"},
                         embedding=[10, 11, 12]),
            SequenceData(seq_id="Seq5", seq="PRTE", attributes={"TARGET": "Soluble", "SET": "val"},
                         embedding=[13, 14, 15]),
            SequenceData(seq_id="Seq6", seq="MMALSM", attributes={"TARGET": "Membrane", "SET": "test"},
                         embedding=torch.tensor([16, 17, 18])),
            SequenceData(seq_id="Seq7", seq="PRSEQ", attributes={"TARGET": "Soluble", "SET": "test"},
                         embedding=np.array([19, 20, 21])),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config = deepcopy(s2c_config)
            config.pop("input_file")
            result1 = BiotrainerModel().train(config=config, input_data=input_data)
            config["model_choice"] = "LogReg"
            result2 = BiotrainerModel().train(config=config, input_data=input_data)
            self.assertNotEqual(result1.derived_values.model_hash, result2.derived_values.model_hash)

    def test_hash_is_different_input(self):
        """ Different config should result in different hash """
        input_data = [
            SequenceData(seq_id="Seq1", seq="MMALSLALM", attributes={"TARGET": "Membrane", "SET": "train"},
                         embedding=[1, 2, 3]),
            SequenceData(seq_id="Seq2", seq="PRTEIN", attributes={"TARGET": "Membrane", "SET": "train"},
                         embedding=[4, 5, 6]),
            SequenceData(seq_id="Seq3", seq="PRT", attributes={"TARGET": "Soluble", "SET": "train"},
                         embedding=[7, 8, 9]),
            SequenceData(seq_id="Seq4", seq="SEQWENCE", attributes={"TARGET": "Membrane", "SET": "val"},
                         embedding=[10, 11, 12]),
            SequenceData(seq_id="Seq5", seq="PRTE", attributes={"TARGET": "Soluble", "SET": "val"},
                         embedding=[13, 14, 15]),
            SequenceData(seq_id="Seq6", seq="MMALSM", attributes={"TARGET": "Membrane", "SET": "test"},
                         embedding=torch.tensor([16, 17, 18])),
            SequenceData(seq_id="Seq7", seq="PRSEQ", attributes={"TARGET": "Soluble", "SET": "test"},
                         embedding=np.array([19, 20, 21])),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config = deepcopy(s2c_config)
            config.pop("input_file")
            result1 = BiotrainerModel().train(config=config, input_data=input_data)
            input_data[0] = SequenceData(seq_id="Seq1",
                                         seq="MMALSLAL",  # Minor change in input sequence
                                         attributes={"TARGET": "Membrane", "SET": "train"},
                                         embedding=[1, 2, 3])
            result2 = BiotrainerModel().train(config=config, input_data=input_data)
            self.assertNotEqual(result1.derived_values.model_hash, result2.derived_values.model_hash)

    def test_hash_is_different_embeddings(self):
        """ Different config should result in different hash """
        input_data = [
            SequenceData(seq_id="Seq1", seq="MMALSLALM", attributes={"TARGET": "Membrane", "SET": "train"},
                         embedding=[1, 2, 3]),
            SequenceData(seq_id="Seq2", seq="PRTEIN", attributes={"TARGET": "Membrane", "SET": "train"},
                         embedding=[4, 5, 6]),
            SequenceData(seq_id="Seq3", seq="PRT", attributes={"TARGET": "Soluble", "SET": "train"},
                         embedding=[7, 8, 9]),
            SequenceData(seq_id="Seq4", seq="SEQWENCE", attributes={"TARGET": "Membrane", "SET": "val"},
                         embedding=[10, 11, 12]),
            SequenceData(seq_id="Seq5", seq="PRTE", attributes={"TARGET": "Soluble", "SET": "val"},
                         embedding=[13, 14, 15]),
            SequenceData(seq_id="Seq6", seq="MMALSM", attributes={"TARGET": "Membrane", "SET": "test"},
                         embedding=torch.tensor([16, 17, 18])),
            SequenceData(seq_id="Seq7", seq="PRSEQ", attributes={"TARGET": "Soluble", "SET": "test"},
                         embedding=np.array([19, 20, 21])),
        ]
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            config = deepcopy(s2c_config)
            config.pop("input_file")
            result1 = BiotrainerModel().train(config=config, input_data=input_data)
            input_data[0] = SequenceData(seq_id="Seq1",
                                         seq="MMALSLALM",
                                         attributes={"TARGET": "Membrane", "SET": "train"},
                                         embedding=[1, 2, 3.01]) # Minor change in embedding
            result2 = BiotrainerModel().train(config=config, input_data=input_data)
            self.assertNotEqual(result1.derived_values.model_hash, result2.derived_values.model_hash)

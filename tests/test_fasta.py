import unittest

from biotrainer_core.input_files import read_FASTA

scl_fasta = "test_input_files/scl_subset/scl_rand.fasta"


class FastaTests(unittest.TestCase):

    def test_read_fasta(self):
        seqs = read_FASTA(scl_fasta)
        self.assertEqual(len(seqs), 134)
        self.assertTrue(all(seq.label is not None for seq in seqs))


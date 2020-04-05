import unittest

import numpy as np

from trans.sed import StochasticEditDistance, Sub, Ins, Del


class TestTransducer(unittest.TestCase):

    def setUp(self) -> None:

        self.source_alphabet1 = list("abcdefg")
        self.target_alphabet1 = list("fghijk")

        self.smart_sed = StochasticEditDistance(
            self.source_alphabet1, self.target_alphabet1, smart_init=True)


    def test_sed_random_initialization(self):

        sed = StochasticEditDistance(
            self.source_alphabet1, self.target_alphabet1, smart_init=False)
        eos_weight = sed.delta_eos

        for weight_dict in ("delta_del", "delta_ins", "delta_sub"):
            for weight in getattr(sed, weight_dict).values():
                self.assertTrue(np.isclose(eos_weight, weight))

    def test_sed_copy_biased_initialization(self):

        sed = StochasticEditDistance(
            self.source_alphabet1, self.target_alphabet1, smart_init=True)
        eos_weight = sed.delta_eos

        for weight_dict in ("delta_del", "delta_ins"):
            for weight in getattr(sed, weight_dict).values():
                self.assertTrue(np.isclose(eos_weight, weight))

        for (x, y), weight in sed.delta_sub.items():
            if x == y:
                self.assertFalse(np.isclose(eos_weight, weight))
            else:
                self.assertTrue(np.isclose(eos_weight, weight))

    def test_viterbi_decoding(self):

        best_edits, distance = self.smart_sed.viterbi_distance(
            source="affa", target="iffig", with_alignment=True)

        expected_edits = [
            Sub(old='a', new='i'), Sub(old='f', new='f'), Sub(old='f', new='f'),
            Ins(new='i'), Sub(old='a', new='g')
        ]

        self.assertTrue(np.isclose(-26.7633, distance))
        self.assertListEqual(expected_edits, best_edits)

    def test_stochastic_decoding(self):

        distance = self.smart_sed.stochastic_distance(
            source="affa", target="iffig")

        self.assertTrue(np.isclose(-26.05741, distance))

    def test_em(self):

        input_pairs = [
            ("abby", "a b i"), ("abidjan", "a b i d ʒ ɑ"),
            ("abject", "a b ʒ ɛ k t"), ("abolir", "a b ɔ l i ʁ"),
            ("abonnement", "a b ɔ n m ɑ")
        ]

        sources, targets = zip(*input_pairs)

        source_alphabet = {c for source in sources for c in source}
        target_alphabet = {c for target in targets for c in target}

        sed = StochasticEditDistance(
            source_alphabet, target_alphabet, smart_init=True)

        o = sed.stochastic_distance(sources[1], targets[1])
        print(o)

        sed.update_model(sources, targets, em_iterations=1)

    def test_fit_from_data(self):

        input_lines = [
            "abby\ta b i", "abidjan\ta b i d ʒ ɑ", "abject\ta b ʒ ɛ k t",
            "abolir\ta b ɔ l i ʁ", "abonnement\ta b ɔ n m ɑ"
        ]

        sed = StochasticEditDistance.fit_from_data(input_lines, em_iterations=1)
        parameter_dictionaries = sed.parameter_dictionaries
        print(parameter_dictionaries)  # @TODO


if __name__ == "__main__":
    unittest.main()

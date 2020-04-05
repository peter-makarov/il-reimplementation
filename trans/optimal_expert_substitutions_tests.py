import heapq
import unittest

from trans import optimal_expert
from trans import optimal_expert_substitutions
from trans import sed

class OptimalSubstitutionExpertTests(unittest.TestCase):

    def setUp(self) -> None:
        aligner = optimal_expert_substitutions.NoSubstitutionAligner()
        self.optimal_expert = \
            optimal_expert_substitutions.OptimalSubstitutionExpert(aligner)

    def test_score_end(self):
        x = "walk"
        i = 4
        y = "walked"
        t = "walked"
        action_scores = self.optimal_expert.score(x, t, i, y)
        expected_action_scores = {optimal_expert.END: 0}
        self.assertEqual(expected_action_scores, action_scores)

    def test_score_empty_strings(self):
        x = ""
        i = 0
        t = ""
        y = "a"
        action_scores = self.optimal_expert.score(x, t, i, y)
        expected_action_scores = {optimal_expert.END: 0}
        self.assertEqual(expected_action_scores, action_scores)

    def test_score(self):
        x = ""
        i = 0
        t = "abbbbbbb"
        y = "bbbbbbb"
        action_scores = self.optimal_expert.score(x, t, i, y)
        expected_action_scores = {optimal_expert.END: 0, sed.Ins("b"): 1}
        self.assertEqual(expected_action_scores, action_scores)

    def test_correct_end(self):
        x = "walk"
        i = 4
        y = "walk"
        t = "walked"
        action_scores = self.optimal_expert.score(x, t, i, y)
        expected_action_scores = {sed.Ins("e"): 2}
        self.assertEqual(expected_action_scores, action_scores)

    def test_sed_aligner(self):

        input_lines = [
            "abby\ta b i", "abidjan\ta b i d ʒ ɑ", "abject\ta b ʒ ɛ k t",
            "abolir\ta b ɔ l i ʁ", "abonnement\ta b ɔ n m ɑ"
        ]

        aligner = sed.StochasticEditDistance.fit_from_data(
            input_lines, em_iterations=5)
        expert = optimal_expert_substitutions.OptimalSubstitutionExpert(aligner)

        x = "abject"
        t = "a b ʒ ɛ k t"
        i = 3
        y = "a b ʒ e"
        action_scores = expert.score(x, t, i, y)
        #print(action_scores)

    def test_sed_aligner_real_data(self):

        with open("test_data/fre_train.tsv") as f:
            input_lines = f.readlines()

        aligner = sed.StochasticEditDistance.fit_from_data(
            input_lines, em_iterations=1)
        expert = optimal_expert_substitutions.OptimalSubstitutionExpert(aligner)

        x = "abject"
        t = "a b ʒ ɛ k t"
        i = 3
        y = "a b ʒ e"
        while True:
            action_scores = expert.score(x, t, i, y)
            print(action_scores)
            action, score = max_dict(action_scores)
            print(f"optimal action: {action, score}\n")
            print()
            if action == optimal_expert.END:
                break
            if isinstance(action, sed.Del):
                i += 1
            elif isinstance(action, sed.Ins):
                y += action.new
            elif isinstance(action, optimal_expert_substitutions.Copy):
                i += 1
                y += action.old
            elif isinstance(action, sed.Sub):
                i += 1
                y += action.new
            else:
                raise ValueError(f"action: {action}")


def max_dict(d):
    x = [(-v, i, k) for i, (k, v) in enumerate(d.items())]
    heapq.heapify(x)
    v, _, k = heapq.heappop(x)
    return k, -v

if __name__ == "__main__":
    unittest.main()

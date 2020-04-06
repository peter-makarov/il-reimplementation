import heapq
import unittest

from trans import optimal_expert
from trans import optimal_expert_substitutions
from trans import sed
from trans.actions import Copy, Del, Ins, Sub


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
        expected_action_scores = {optimal_expert.END: 0, Ins("b"): 1}
        self.assertEqual(expected_action_scores, action_scores)

    def test_correct_end(self):
        x = "walk"
        i = 4
        y = "walk"
        t = "walked"
        action_scores = self.optimal_expert.score(x, t, i, y)
        expected_action_scores = {Ins("e"): 2}
        self.assertEqual(expected_action_scores, action_scores)

    def test_sed_aligner(self):

        input_lines = [
            "abba\tabba", "bababa\tbababa", "bba\tbba",
            "bbbb\tbbb", "bbbbb\tbbbb"
        ]

        # learns to copy even when not initialized with copy bias
        aligner = sed.StochasticEditDistance.fit_from_data(
            input_lines, em_iterations=5)
        expert = optimal_expert_substitutions.OptimalSubstitutionExpert(aligner)

        x = ""
        i = 0
        t = "abbbbbbb"
        y = "bbbbbbb"
        action_scores = expert.score(x, t, i, y)
        optimal_action, _ = max_dict(action_scores)
        expected_actions = {optimal_expert.END, Ins("b")}
        self.assertSetEqual(expected_actions, set(action_scores.keys()))
        self.assertEqual(optimal_expert.END, optimal_action)

    def test_sed_aligner_real_data(self):

        verbose = False
        input_lines = []
        with open("test_data/fre_train.tsv") as f:
            try:
                for _ in range(100):
                    input_lines.append(next(f))
            except StopIteration:
                pass

        aligner = sed.StochasticEditDistance.fit_from_data(
            input_lines, em_iterations=1)
        expert = optimal_expert_substitutions.OptimalSubstitutionExpert(aligner)

        x = "abject"
        t = "a b ʒ ɛ k t"
        i = 3
        y = "a b ʒ e"

        optimal_actions = iter(
            (Sub(old='e', new=' '), Sub(old='c', new='k'), Ins(new=' '),
             Copy(old='t', new='t'), optimal_expert.END)
        )

        while True:
            action_scores = expert.score(x, t, i, y)
            action, score = max_dict(action_scores)
            if verbose:
                print(action_scores)
                print(f"optimal action: {action, score}\n")
                print()
            if action == optimal_expert.END:
                break
            if isinstance(action, Del):
                i += 1
            elif isinstance(action, Ins):
                y += action.new
            elif isinstance(action, Sub):
                i += 1
                y += action.new
            else:
                raise ValueError(f"action: {action}")
            self.assertEqual(next(optimal_actions), action)

    def test_actions(self):

        self.assertTrue(isinstance(Copy(1, 1), Sub))
        self.assertFalse(isinstance(Sub(1, 2), Copy))


def max_dict(d):
    x = [(-v, i, k) for i, (k, v) in enumerate(d.items())]
    heapq.heapify(x)
    v, _, k = heapq.heappop(x)
    return k, -v


if __name__ == "__main__":
    unittest.main()

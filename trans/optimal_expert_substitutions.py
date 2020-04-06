from typing import Any, Iterable, List, Sequence, Union
import collections

from trans.defaults import END_WORD as END
from trans.optimal_expert import ActionsPrefix, OptimalExpert, Prefix, edit_distance
from trans.actions import Aligner, Copy, Del, Ins, Sub

import numpy as np


Action = Union[collections.namedtuple, int]


class EditDistanceAligner(Aligner):

    def __init__(self, del_cost=1., ins_cost=1., sub_cost=1.):
        self.del_cost = del_cost
        self.ins_cost = ins_cost
        self.sub_cost = sub_cost

    def action_sequence_cost(self, x: Sequence[Any], y: Sequence[Any],
                             x_offset: int, y_offset: int) -> float:
        ed = edit_distance(
            x, y,
            del_cost=self.del_cost, ins_cost=self.ins_cost,
            sub_cost=self.sub_cost,
            x_offset=x_offset, y_offset=y_offset
        )
        return ed[-1, -1]

    def action_cost(self, action: Action):
        if isinstance(action, Copy) or action == END:
            return 0
        else:
            return 1


class NoSubstitutionAligner(EditDistanceAligner):

    def __init__(self):
        super().__init__(del_cost=1., ins_cost=1., sub_cost=1.)

    def action_cost(self, action: Action):
        if isinstance(action, Sub):
            return np.inf
        else:
            return super().action_cost(action)


class OptimalSubstitutionExpert(OptimalExpert):

    def __init__(self, aligner: Aligner, maximum_output_length: int = 150):
        super().__init__(maximum_output_length)
        self.aligner = aligner

    def find_valid_actions(self, x: Sequence[Any], i: int, y: Sequence[Any],
                           prefixes: Iterable[Prefix]):
        if len(y) >= self.maximum_output_length:
            return {END}
        input_not_empty = i < len(x)
        attention = x[i] if input_not_empty else None
        actions_prefixes: List[ActionsPrefix] = []
        for prefix in prefixes:
            prefix_insert = prefix.leftmost_of_suffix
            if prefix_insert == END:
                valid_actions = {END}
            else:
                valid_actions = {Ins(prefix_insert)}
            if input_not_empty:
                if prefix_insert == attention:
                    valid_actions.add(Copy(attention, prefix_insert))
                elif prefix_insert != END:
                    # substitutions
                    valid_actions.add(Sub(old=attention, new=prefix_insert))
                valid_actions.add(Del(attention))
            actions_prefix = ActionsPrefix(valid_actions, prefix)
            actions_prefixes.append(actions_prefix)
        return actions_prefixes

    def roll_out(self, x: Sequence[Any], t: Sequence[Any], i: int,
                 actions_prefixes: Iterable[ActionsPrefix]):
        costs_to_go = dict()
        for actions_prefix in actions_prefixes:
            suffix_begin = actions_prefix.prefix.j
            for action in actions_prefix.actions:
                if isinstance(action, Del):
                    x_offset = i + 1
                    t_offset = suffix_begin
                elif isinstance(action, Ins):
                    x_offset = i
                    t_offset = suffix_begin + 1
                elif isinstance(action, Sub):
                    x_offset = i + 1
                    t_offset = suffix_begin + 1
                elif action == END:
                    x_offset = i
                    t_offset = suffix_begin
                else:
                    raise ValueError(f"Unknown action: {action}")
                sequence_cost = self.aligner.action_sequence_cost(
                    x, t, x_offset, t_offset)
                action_cost = self.aligner.action_cost(action)
                cost = action_cost + sequence_cost
                if action not in costs_to_go or costs_to_go[action] > cost:
                    costs_to_go[action] = cost
        return costs_to_go

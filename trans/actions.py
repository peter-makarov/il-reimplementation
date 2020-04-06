from typing import Any, Sequence
import abc
import dataclasses


@dataclasses.dataclass(frozen=True, eq=True)
class Sub(object):
    old: Any
    new: Any


@dataclasses.dataclass(frozen=True, eq=True)
class Copy(Sub):
    old: Any
    new: Any

    def __post_init__(self):
        if self.old != self.new:
            raise ValueError(f"Copy: old={self.old} != new={self.new}")


@dataclasses.dataclass(frozen=True, eq=True)
class Del(object):
    old: Any


@dataclasses.dataclass(frozen=True, eq=True)
class Ins(object):
    new: Any


class Aligner(abc.ABC):

    @abc.abstractmethod
    def action_sequence_cost(self, x: Sequence[Any], y: Sequence[Any],
                             x_offset: int, y_offset: int) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def action_cost(self, action: Any) -> float:
        raise NotImplementedError

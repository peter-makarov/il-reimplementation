"""Defines a neural transducer."""
from typing import Any, Dict, List, Optional, Tuple, Union
import dataclasses
import functools
import heapq

import torch
import numpy as np

from trans import optimal_expert
from trans import vocabulary
from trans.actions import ConditionalCopy, ConditionalDel, ConditionalIns, \
    ConditionalSub, Edit, EndOfSequence, GenerativeEdit, BeginOfSequence
from trans.vocabulary import BEGIN_WORD, COPY, DELETE, END_WORD, PAD

MAX_ACTION_SEQ_LEN = 150


@functools.total_ordering
@dataclasses.dataclass
class Output:
    action_history: List[Any]
    output: Union[str, List[str]]
    log_p: float
    losses: torch.Tensor = None

    def __lt__(self, other):
        return self.log_p < other.log_p

    def __eq__(self, other):
        return self.log_p == other.log_p


@dataclasses.dataclass
class Hypothesis:
    action_history: List[Any]
    alignment: int
    decoder: Tuple[torch.Tensor, torch.Tensor]
    negative_log_p: float
    output: List[str]


@functools.total_ordering
@dataclasses.dataclass
class Expansion:
    action: Any
    decoder: Tuple[torch.Tensor, torch.Tensor]
    from_hypothesis: Hypothesis
    negative_log_p: float

    def __lt__(self, other):
        return self.negative_log_p < other.negative_log_p

    def __eq__(self, other):
        return self.negative_log_p == other.negative_log_p


class Transducer(torch.nn.Module):
    def __init__(self, vocab: vocabulary.Vocabularies,
                 expert: optimal_expert.Expert, char_dim: int, action_dim: int,
                 enc_hidden_dim: int, enc_layers: int, dec_hidden_dim: int,
                 dec_layers: int, device: str = 'cpu', **kwargs):

        super().__init__()
        self.device = torch.device(device)

        self.vocab = vocab
        self.optimal_expert = expert

        self.number_characters = len(vocab.characters)
        self.number_actions = len(vocab.actions)
        self.substitutions = self.vocab.substitutions
        self.inserts = self.vocab.insertions

        self.dec_layers = dec_layers
        self.dec_hidden_dim = dec_hidden_dim

        # encoder
        self.char_lookup = torch.nn.Embedding(
            num_embeddings=self.number_characters,
            embedding_dim=char_dim,
            device=self.device,
            padding_idx=PAD
        )

        self.enc = torch.nn.LSTM(
            input_size=char_dim,
            hidden_size=enc_hidden_dim,
            num_layers=enc_layers,
            bidirectional=True,
            device=self.device,
        )

        # decoder
        self.act_lookup = torch.nn.Embedding(
            num_embeddings=self.number_actions,
            embedding_dim=action_dim,
            device=self.device,
        )

        decoder_input_dim = enc_hidden_dim * 2 + action_dim

        self.dec = torch.nn.LSTM(
            input_size=decoder_input_dim,
            hidden_size=dec_hidden_dim,
            num_layers=dec_layers,
            device=self.device,
        )

        self._h0_c0 = None

        # classifier
        self.W = torch.nn.Linear(
            in_features=dec_hidden_dim,
            out_features=self.number_actions,
            device=self.device,

        )

    @property
    def h0_c0(self):
        return self._h0_c0

    @h0_c0.setter
    def h0_c0(self, batch_size):
        if not self._h0_c0 or \
                (batch_size and self._h0_c0[0].size(1) != batch_size):
            self._h0_c0 = (
                torch.zeros((self.dec_layers, batch_size, self.dec_hidden_dim), device=self.device),
                torch.zeros((self.dec_layers, batch_size, self.dec_hidden_dim), device=self.device),
            )

    def input_embedding(self, input_: Union[List[int], List[List[int]]], is_training: bool):
        """Returns a list of character embeddings for the input."""
        input_tensor = torch.tensor(
            input_,
            dtype=torch.int,
            device=self.device,
        )
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(dim=0)
        emb = self.char_lookup(input_tensor)

        if not is_training:

            unk_indices = input_tensor >= self.number_characters
            if unk_indices.sum().item() > 0:
                # UNK is the average of trained embeddings (excluding UNK)
                ids_tensor = torch.tensor(
                    range(1, self.number_characters),
                    dtype=torch.int,
                    device=self.device,
                )
                unk = self.char_lookup(ids_tensor).mean(dim=0)
                emb[unk_indices] = unk

        return emb

    def compute_valid_actions(self, length_encoder_suffix: int) -> torch.tensor:
        valid_actions = torch.full((self.number_actions,), False,
                                   dtype=torch.bool, device=self.device)
        valid_actions[END_WORD] = True
        valid_actions[self.inserts] = True
        if length_encoder_suffix > 1:
            valid_actions[[COPY, DELETE]] = True
            valid_actions[self.substitutions] = True
        return valid_actions

    @staticmethod
    def sample(log_probs: np.array) -> int:
        """Samples an action from a log-probability distribution."""
        dist = np.exp(log_probs)
        rand = np.random.rand()
        for action, p in enumerate(dist):
            rand -= p
            if rand <= 0:
                break
        return action

    @staticmethod
    def remap_actions(action_scores: Dict[Any, float]) -> Dict[Any, float]:
        """Maps a generative oracle's edit to their conditional counterparts."""
        remapped_action_scores = dict()
        for action, score in action_scores.items():
            if isinstance(action, GenerativeEdit):
                remapped_action = action.conditional_counterpart()
            elif isinstance(action, Edit):
                remapped_action = action
            else:
                raise ValueError(f"Unknown action: {action, score}.\n"
                                 f"action_scores: {action_scores}")
            remapped_action_scores[remapped_action] = score
        return remapped_action_scores

    def expert_rollout(self, input_: str, target: str, alignment: int,
                       prediction: List[str]) -> List[int]:
        """Rolls out with optimal expert policy.

        Args:
            input_: Input string (x).
            target: Target prediction (t).
            alignment: Position of control in the input string.
            prediction: The current prediction so far (y).

        Returns:
            List of optimal actions as integer codes."""
        raw_action_scores = self.optimal_expert.score(
            input_, target, alignment, prediction)
        action_scores = self.remap_actions(raw_action_scores)

        optimal_value = min(action_scores.values())
        return [self.vocab.encode_unseen_action(action)
                for action, value in action_scores.items()
                if value == optimal_value]

    def mark_as_invalid(self, logits: torch.Tensor,
                        valid_actions: torch.Tensor) -> torch.Tensor:
        log_validity = torch.full(
            logits.size(),
            -np.inf,
            device=self.device,
        )  # All actions invalid by default.
        log_validity[valid_actions] = 0.
        return logits + log_validity

    def log_softmax(self, logits: torch.tensor,
                    valid_actions: torch.tensor) -> torch.tensor:
        logits_valid = self.mark_as_invalid(logits, valid_actions)
        return torch.nn.functional.log_softmax(logits_valid, dim=2)

    def log_sum_softmax_loss(self, logits: torch.Tensor,
                             optimal_actions: torch.Tensor,
                             valid_actions: torch.Tensor) -> torch.Tensor:
        """Compute log loss similar to Riezler et al 2000."""
        logits_valid = self.mark_as_invalid(logits, valid_actions)
        # padding can be inferred from optimal actions
        # --> if mask only consists of False values
        paddings = ~torch.any(optimal_actions, dim=2)
        logits_valid[paddings] = -np.inf
        logits_optimal = logits_valid.clone()
        logits_optimal[~(valid_actions*optimal_actions)] = -np.inf

        log_sum_selected_terms = torch.logsumexp(
            logits_optimal,
            dim=2,
        )

        normalization_term = torch.logsumexp(
            logits_valid,
            dim=2,
        )

        if paddings.sum() > 0:
            log_sum_selected_terms =\
                torch.where(~paddings, log_sum_selected_terms, torch.tensor(0.))
            normalization_term =\
                torch.where(~paddings, normalization_term, torch.tensor(0.))

        return log_sum_selected_terms - normalization_term

    def transduce(self, input_: List[List[str]], encoded_input: List[List[int]],
                  target: List[str] = None, optimal_actions: List[List[List[int]]] = None,
                  rollin: Optional[float] = None):
        """Runs the transducer for dynamic-oracle training and greedy decoding.

        Args:
            input_: Input string.
            encoded_input: List of integer character codes.
            target: Target string during training, `None` during prediction.
            optimal_actions: Optimal actions during training, `None` during prediction.
            rollin: The probability with which an action sampled from the model
                    is executed. Used during training."""
        is_training = bool(target)

        input_emb = torch.transpose(self.input_embedding(encoded_input, is_training), 0, 1)
        bidirectional_emb_, _ = self.enc(input_emb)  # L x B x E
        bidirectional_emb = bidirectional_emb_[1:, :]  # drop BEGIN_WORD
        batch_size = bidirectional_emb.size()[1]

        # adjust initial decoder states if batch_size has changed
        self.h0_c0 = batch_size
        decoder = self.h0_c0

        true_input_lengths = torch.tensor(
            # +1 because end word is not included in input
            [len(i) + 1 for i in input_], device=self.device)

        alignment = torch.full((batch_size,), 0, device=self.device)
        # maps action index to alignment update
        alignment_update = [0] * self.number_actions
        for i, action in enumerate(self.vocab.actions.i2w):
            if isinstance(action,
                          (ConditionalCopy, ConditionalDel, ConditionalSub)):
                alignment_update[i] = 1
        alignment_update = torch.tensor(alignment_update, device=self.device)

        valid_actions_lookup = torch.stack(
            [self.compute_valid_actions(s)
             # len(bidreictional_emb) = max input length
             for s in range(len(bidirectional_emb) + 1)],
            dim=0).unsqueeze(dim=0)

        action_history = torch.IntTensor([[[BEGIN_WORD]] * batch_size])
        log_p = torch.full((1, batch_size), 0.0, device=self.device)

        if is_training:
            # build mask for optimal actions (necessary for error term in loss func.)
            # actions are precomputed (before training)
            # optimal actions --> ACTIONS TO MAXIMIZE
            max_len_actions = len(max(optimal_actions, key=len))
            optimal_actions_lookup = torch.full(
                (1, max_len_actions, batch_size, self.number_actions), False,
                dtype=torch.bool, device=self.device)
            batch_pos, seq_pos, emb_pos = \
                zip(*[(b, s, a) for b in range(len(optimal_actions))
                      for s in range(min(len(optimal_actions[b]), max_len_actions))
                      for a in optimal_actions[b][s]])
            optimal_actions_lookup[:, seq_pos, batch_pos, emb_pos] = True

            loss = torch.full((1, batch_size), 0.0, requires_grad=True,
                              device=self.device)

            # the training goal is to minimize the (softmax) loss of the target
            # actions vector (see below). these actions are not dynamically computed
            # --> it only makes sense to decode until for all seqs in the batch
            # decoding steps >= length of target action sequence
            optimal_actions_lengths = torch.tensor([len(s) for s in optimal_actions],
                                                   device=self.device)

            def continue_decoding():
                return torch.where(
                    action_history.size(2) - 1 >= optimal_actions_lengths, 0, 1
                ).sum() > 0
        else:
            loss = None

            # during evaluation decoding is continued until
            # all sequences in the batch have "found" an end word
            def continue_decoding():
                return torch.any(action_history == END_WORD, dim=2).sum() < batch_size

        while continue_decoding() and action_history.size(2) <= MAX_ACTION_SEQ_LEN:

            valid_actions_mask = valid_actions_lookup[:, true_input_lengths - alignment]

            input_char_embedding = bidirectional_emb\
                [alignment, torch.arange(bidirectional_emb.size(1))].unsqueeze(dim=0)
            previous_action_embedding = self.act_lookup(action_history[:, :, -1])

            decoder_input = torch.cat(
                (input_char_embedding, previous_action_embedding),
                dim=2
            )
            decoder_output_, decoder = self.dec(decoder_input, decoder)
            decoder_output = decoder_output_

            logits = self.W(decoder_output)
            log_probs = self.log_softmax(logits, valid_actions_mask)

            if not is_training:
                # argmax decoding
                actions = torch.argmax(log_probs, dim=2)
            else:
                # -1 as BEGIN_WORD is excluded
                optimal_actions_mask = optimal_actions_lookup[:, action_history.size(2) - 1]

                # ACTION SPACE EXPLORATION: NEXT ACTION
                # rollin not implemented at the moment
                # todo: rollin
                if False:
                    # action is picked by sampling
                    # todo: sampling correct?
                    actions = torch.randint(0, self.number_actions, (1, batch_size),
                                            device=self.device)
                    # actions = self.sample(log_probs)
                else:
                    # action is picked from optimal actions
                    # reinforce model beliefs by picking highest probability
                    # action that is consistent with oracle
                    selected_logits = torch.clone(logits)
                    selected_logits[~optimal_actions_mask] = -np.inf
                    actions = torch.max(selected_logits, 2)[1]

                # add loss
                loss = loss +\
                    self.log_sum_softmax_loss(logits, optimal_actions_mask, valid_actions_mask)

            # update states
            log_p += log_probs[:, torch.arange(batch_size), actions.squeeze(dim=0)]
            action_history = torch.cat(
                (action_history, actions.unsqueeze(dim=2)),
                dim=2
            )
            alignment = alignment + alignment_update[actions.squeeze(dim=0)]

        # trim action history
        # --> first element is not considered (begin-of-sequence-token)
        # --> and only token up to the first end-of-sequence-token (including it)
        action_history = [seq[1:(seq.index(EndOfSequence()) + 1 if EndOfSequence() in seq else -1)]
                          for seq in action_history.squeeze(dim=0).tolist()]

        if is_training:
            # the loss is divided by the number of tokens in the sequence
            # that are minimized --> loss = avg. loss per token (per seq in batch)
            loss = -loss / optimal_actions_lengths
            # we do the same for log_p
            log_p = log_p / optimal_actions_lengths
        else:
            # in eval mode log_p is divided
            # by number of generated output tokens
            log_p = log_p / torch.tensor([len(h) for h in action_history], device=self.device)

        return Output(action_history, self.batch_decode(input_, action_history),
                      torch.mean(log_p).item(), loss)

    def batch_decode(self, input_: Union[List[str], List[List[str]]], encoded_output: List[List[int]]) -> List[str]:
        output = []
        for i, seq in enumerate(encoded_output):
            decoded_seq = []
            alignment = 0
            for a in seq:
                char_, alignment, _ = self.decode_char(input_[i], a, alignment)
                if char_ != "":
                    decoded_seq.append(char_)
            output.append("".join(decoded_seq))

        return output

    def decode_char(self, input_: Union[str, List[str]],
                    encoded_action: int, alignment: int) -> Tuple[str, int, bool]:
        action = self.vocab.decode_action(encoded_action)
        stop = False

        if isinstance(action, ConditionalCopy):
            char_ = input_[alignment]
            alignment += 1
        elif isinstance(action, ConditionalDel):
            char_ = ""
            alignment += 1
        elif isinstance(action, ConditionalIns):
            char_ = action.new
        elif isinstance(action, ConditionalSub):
            char_ = action.new
            alignment += 1
        elif isinstance(action, EndOfSequence):
            char_ = ""
            stop = True
        elif isinstance(action, BeginOfSequence):
            char_ = ""
        else:
            raise ValueError(f"Unknown action: {action}.")

        return char_, alignment, stop

    def beam_search_decode(self, input_: str, encoded_input: List[int],
                           beam_width: int):
        """Runs the transducer with beam search.

        Args:
            input_: Input string.
            encoded_input: List of integer character codes.
            beam_width: Width of the beam search.
        """
        input_emb = self.input_embedding(encoded_input, is_training=False)
        bidirectional_emb_, _ = self.enc(input_emb)
        bidirectional_emb = torch.transpose(bidirectional_emb_, 1, 0)
        input_length = len(bidirectional_emb)
        decoder = self.h0_c0

        beam: List[Hypothesis] = [
            Hypothesis(action_history=[BEGIN_WORD],
                       alignment=0,
                       decoder=decoder,
                       negative_log_p=0.,
                       output=[])]

        hypothesis_length = 0
        complete_hypotheses = []

        while beam and beam_width > 0 and hypothesis_length <= MAX_ACTION_SEQ_LEN:

            expansions: List[Expansion] = []

            for hypothesis in beam:

                length_encoder_suffix = input_length - hypothesis.alignment
                valid_actions = self.compute_valid_actions(length_encoder_suffix)
                # decoder
                decoder_input = torch.cat([
                    bidirectional_emb[hypothesis.alignment],
                    self.act_lookup(torch.tensor([hypothesis.action_history[-1]],
                                                 device=self.device))
                ], dim=1,
                ).unsqueeze(1)
                decoder_output_, decoder = self.dec(decoder_input, hypothesis.decoder)

                decoder_output = decoder_output_
                logits = self.W(decoder_output)
                log_probs = self.log_softmax(logits, valid_actions.unsqueeze(dim=0).unsqueeze(dim=1))

                for i, action in enumerate(valid_actions):
                    if not action:
                        continue
                    log_p = hypothesis.negative_log_p - \
                            log_probs[0, 0, i]  # min heap, so minus

                    heapq.heappush(expansions,
                                   Expansion(action, decoder,
                                             hypothesis, log_p))

            beam: List[Hypothesis] = []

            for _ in range(beam_width):

                expansion: Expansion = heapq.heappop(expansions)
                from_hypothesis = expansion.from_hypothesis
                action = expansion.action
                action_history = list(from_hypothesis.action_history)
                action_history.append(action)
                output = list(from_hypothesis.output)

                # execute the action to update the transducer state
                action = self.vocab.decode_action(action)

                if isinstance(action, EndOfSequence):
                    # 1. COMPLETE HYPOTHESIS, REDUCE BEAM
                    complete_hypothesis = Output(
                        action_history=action_history,
                        output="".join(output),
                        log_p=-expansion.negative_log_p)  # undo min heap minus

                    complete_hypotheses.append(complete_hypothesis)
                    beam_width -= 1
                else:
                    # 2. EXECUTE ACTION AND ADD FULL HYPOTHESIS TO NEW BEAM
                    alignment = from_hypothesis.alignment

                    if isinstance(action, ConditionalCopy):
                        char_ = input_[alignment]
                        alignment += 1
                        output.append(char_)
                    elif isinstance(action, ConditionalDel):
                        alignment += 1
                    elif isinstance(action, ConditionalIns):
                        output.append(action.new)
                    elif isinstance(action, ConditionalSub):
                        alignment += 1
                        output.append(action.new)
                    else:
                        raise ValueError(f"Unknown action: {action}.")

                    hypothesis = Hypothesis(
                        action_history=action_history,
                        alignment=alignment,
                        decoder=expansion.decoder,
                        negative_log_p=expansion.negative_log_p,
                        output=output)

                    beam.append(hypothesis)

            hypothesis_length += 1

        if not complete_hypotheses:
            # nothing found because the model is very bad
            for hypothesis in beam:

                complete_hypothesis = Output(
                    action_history=hypothesis.action_history,
                    output="".join(hypothesis.output),
                    log_p=-hypothesis.negative_log_p)  # undo min heap minus

                complete_hypotheses.append(complete_hypothesis)

        complete_hypotheses.sort(reverse=True)
        return complete_hypotheses

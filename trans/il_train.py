from typing import List, Tuple

import argparse
import dataclasses
import json
import logging
import math
import progressbar
import os
import random
import sys

import numpy as np

from trans import il_vocabulary
from trans import il_transducer
from trans import optimal_expert_substitutions
from trans import sed
from trans import il_utils

import dynet_config
dynet_config.set(random_seed=1)
import dynet as dy

random.seed(1)


@dataclasses.dataclass
class Sample:
    input: str
    target: str
    encoded_input: List[int]


def inverse_sigmoid_schedule(k: int):
    """Probability of sampling an action from the model as function of epoch."""
    return lambda epoch: (1 - k / (k + np.exp(epoch / k)))


@dataclasses.dataclass
class DecodingOutput:
    accuracy: float
    loss: float
    predictions: List[str]


def decode(transducer: il_transducer.Transducer, data: List[Sample],
           beam_width: int = 1) -> DecodingOutput:
    if beam_width == 1:
        decoding = lambda s: \
            transducer.transduce(s.input, s.encoded_input)
    else:
        decoding = lambda s: \
            transducer.beam_search_decode(s.input, s.encoded_input,
                                          beam_width)[0]
    predictions = []
    loss = 0
    correct = 0
    j = 0
    for j, sample in enumerate(data):
        if j % 20 == 0:
            dy.renew_cg()
        output = decoding(sample)
        prediction = output.output
        predictions.append(f"{sample.input}\t{prediction}")
        if prediction == sample.target:
            correct += 1
        loss += output.log_p / (len(output.action_history) - 1)
        if j > 0 and j % 500 == 0:
            logging.info(f"\t\t...{j} samples")
    logging.info(f"\t\t...{j} samples")

    return DecodingOutput(accuracy=correct / len(data),
                          loss=- loss / len(data),
                          predictions=predictions)


def decode(transducer: il_transducer.Transducer,
           data: List[Sample]) -> Tuple[float, float, List[str]]:
    predictions = []
    losses = []
    correct = 0
    for j, sample in enumerate(data):
        if j % 20 == 0:
            dy.renew_cg()
        output = transducer.transduce(
            input_=sample.input,
            encoded_input=sample.encoded_input,
            target=None,
            rollin=None,
            external_cg=True,
        )
        prediction = output.output
        predictions.append(f"{sample.input}\t{prediction}")
        if prediction == sample.target:
            correct += 1
        losses.extend([loss.npvalue() for loss in output.losses])
        if j > 0 and j % 500 == 0:
            logging.info(f"\t\t...{j} samples")
    accuracy = correct / len(data)
    loss = -float(np.mean(losses))
    return accuracy, loss, predictions


def beam_search_decode(transducer: il_transducer.Transducer, beam_width: int,
           data: List[Sample]) -> Tuple[float, List[str]]:
    predictions = []
    correct = 0
    for j, sample in enumerate(data):
        if j % 20 == 0:
            dy.renew_cg()
        outputs = transducer.beam_search_decode(
            input_=sample.input,
            encoded_input=sample.encoded_input,
            beam_width=beam_width,
            external_cg=True,
        )
        prediction = outputs[0].output
        predictions.append(f"{sample.input}\t{prediction}")
        if sample.target is not None and prediction == sample.target:
            correct += 1
        if j > 0 and j % 500 == 0:
            logging.info("\t\t...%d samples", j)
    accuracy = correct / len(data)
    return accuracy, predictions


def write_results(accuracy, predictions, output: str, dataset_name: str):
    logging.info("%s set accuracy: %.4f", dataset_name, accuracy)

    dev_eval = os.path.join(
        output, f"{dataset_name}_beam{args.beam_width}.eval")
    with open(dev_eval, mode="w") as w:
        w.write(f"{dataset_name} accuracy: {accuracy:.4f}")

    dev_prediction_tsv = os.path.join(
        output, f"{dataset_name}_beam{args.beam_width}.predictions")
    with open(dev_prediction_tsv, encoding="utf8", mode="w") as w:
        w.write("\n".join(predictions))


def main(args):

    dargs = args.__dict__
    for key, value in dargs.items():
        logging.info(f"%s: %s", str(key).ljust(15), value)

    os.mkdir(args.output)

    vocabulary = il_vocabulary.Vocabularies()

    training_data = []
    with open(args.train, encoding="utf8") as f:
        for line in f:
            input_, target = line.rstrip().split("\t", 1)
            encoded_input = vocabulary.encode_input(input_)
            vocabulary.encode_actions(target)
            sample = Sample(input_, target, encoded_input)
            training_data.append(sample)

    logging.info("%d actions: %s", len(vocabulary.actions),
                 vocabulary.actions)
    logging.info("%d chars: %s", len(vocabulary.characters),
                 vocabulary.characters)

    development_data = []
    with open(args.dev, encoding="utf8") as f:
        for line in f:
            input_, target = line.rstrip().split("\t", 1)
            encoded_input = vocabulary.encode_unseen_input(input_)
            sample = Sample(input_, target, encoded_input)
            development_data.append(sample)

    if args.test is not None:
        test_data = []
        with open(args.test, encoding="utf8") as f:
            for line in f:
                input_, target = line.rstrip().split("\t", 1)
                encoded_input = vocabulary.encode_unseen_input(input_)
                sample = Sample(input_, target, encoded_input)
                test_data.append(sample)

    sed_aligner = sed.StochasticEditDistance.fit_from_data(training_data,
        em_iterations=args.sed_em_iterations, discount=args.sed_discount)
    expert = optimal_expert_substitutions.OptimalSubstitutionExpert(sed_aligner)

    model = dy.Model()
    transducer = il_transducer.Transducer(model, vocabulary, expert, **dargs)

    widgets = [progressbar.Bar(">"), " ", progressbar.ETA()]
    train_progress_bar = progressbar.ProgressBar(
        widgets=widgets, maxval=args.epochs).start()

    train_log_path = os.path.join(args.output, "train.log")
    best_model_path = os.path.join(args.output, "best.model")

    with open(train_log_path, "w") as w:
        w.write("epoch\tavg_loss\ttrain_accuracy\tdev_accuracy\n")

    trainer = dy.AdadeltaTrainer(model)
    train_subset = training_data[:100]
    rollin_schedule = inverse_sigmoid_schedule(args.k)
    max_patience = args.patience
    batch_size = args.batch_size

    logging.info("Training for a maximum of %d with a maximum patience of %d.",
                 args.epochs, max_patience)
    logging.info("Number of train batches: %d.",
                 math.ceil(len(training_data) / batch_size))

    best_train_accuracy = 0
    best_dev_accuracy = 0
    best_epoch = 0
    patience = 0

    for epoch in range(args.epochs):

        logging.info("Training...")
        with il_utils.Timer():
            train_loss = 0.
            random.shuffle(training_data)
            batches = [training_data[i:i + batch_size]
                       for i in range(0, len(training_data), batch_size)]
            rollin = rollin_schedule(epoch)
            j = 0
            for j, batch in enumerate(batches):
                losses = []
                dy.renew_cg()
                for sample in batch:
                    output = transducer.transduce(
                        input_=sample.input,
                        encoded_input=sample.encoded_input,
                        target=sample.target,
                        rollin=rollin,
                        external_cg=True,
                    )
                    losses.extend(output.losses)
                batch_loss = -dy.average(losses)
                train_loss += batch_loss.scalar_value()
                batch_loss.backward()
                trainer.update()
                if j > 0 and j % 100 == 0:
                    logging.info("\t\t...%d batches", j)
            logging.info("\t\t...%d batches", j)

        avg_loss = train_loss / len(batches)
        logging.info("Average train loss: %.4f.", avg_loss)

        logging.info("Evaluating on training data subset...")
        with il_utils.Timer():
            train_accuracy, _avg_train_loss, _ = decode(transducer, train_subset)

        if train_accuracy > best_train_accuracy:
            best_train_accuracy = train_accuracy

        patience += 1

        logging.info("Evaluating on development data...")
        with il_utils.Timer():
            dev_accuracy, avg_dev_loss, _ = decode(transducer, development_data)

        if dev_accuracy > best_dev_accuracy:
            best_dev_accuracy = dev_accuracy
            best_epoch = epoch
            patience = 0
            logging.info("Found best dev accuracy %.4f.", best_dev_accuracy)
            model.save(best_model_path)
            logging.info("Saved new best model to %s.", best_model_path)

        logging.info(
            f"epoch {epoch} / {args.epochs - 1}: train loss: {avg_loss:.4f} "
            f"dev loss: {avg_dev_loss:.4f} train acc: {train_accuracy:.4f} "
            f"dev acc: {dev_accuracy:.4f} best train acc: {best_train_accuracy:.4f} "
            f"best dev acc: {best_dev_accuracy:.4f} best epoch: {best_epoch} "
            f"patience: {patience} / {max_patience - 1}"
        )

        log_line = f"{epoch}\t{avg_loss:.4f}\t{train_accuracy:.4f}\t{dev_accuracy:.4f}\n"
        with open(train_log_path, "a") as a:
            a.write(log_line)

        if patience == max_patience:
            logging.info("Out of patience after %d epochs.", epoch + 1)
            train_progress_bar.finish()
            break

        train_progress_bar.update(epoch)

    logging.info("Finished training.")

    if os.path.exists(best_model_path):
        model = dy.Model()
        transducer = il_transducer.Transducer(model, vocabulary, expert,
                                              **dargs)
        model.populate(best_model_path)
        logging.info("Evaluating best model on development data "
                     "using beam search (beam width %d)...", args.beam_width)
        with il_utils.Timer():
            accuracy, _, predictions = decode(transducer, development_data)
        logging.info("Greedy decoding accuracy: %.4f", accuracy)
        with open(os.path.join(args.output, "dev_greedy.predictions"), "w") as w:
            w.write("\n".join(predictions))
        with il_utils.Timer():
            accuracy, predictions = beam_search_decode(
                transducer, args.beam_width, development_data)
        write_results(accuracy, predictions, args.output, "dev")

        if args.test is None:
            sys.exit(0)

        logging.info("Evaluating best model on test data "
                     "using beam search (beam width %d)...", args.beam_width)
        with il_utils.Timer():
            accuracy, _, predictions = decode(transducer, development_data)
        logging.info("Greedy decoding accuracy: %.4f", accuracy)
        with open(os.path.join(args.output, "test_greedy.predictions"), "w") as w:
            w.write("\n".join(predictions))
        with il_utils.Timer():
            accuracy, predictions = beam_search_decode(
                transducer, args.beam_width, test_data)
        write_results(accuracy, predictions, args.output, "test")


if __name__ == "__main__":

    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Train a g2p neural transducer.")

    parser.add_argument("--dynet-mem", type=int, default=1000,
                        help="Allocate MEM MB to DyNET.")
    parser.add_argument("--dynet-autobatch", type=int,
                        help="Perform automatic minibatching.")
    parser.add_argument("--train", type=str, required=True,
                        help="Path to train set data.")
    parser.add_argument("--dev", type=str, required=True,
                        help="Path to development set data.")
    parser.add_argument("--test", type=str,
                        help="Path to development set data.")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory.")
    parser.add_argument("--char-dim", type=int, default=100,
                        help="Character peak_embedding dimension.")
    parser.add_argument("--action-dim", type=int, default=100,
                        help="Action peak_embedding dimension.")
    parser.add_argument("--enc-hidden-dim", type=int, default=200,
                        help="Encoder LSTM state dimension.")
    parser.add_argument("--dec-hidden-dim", type=int, default=200,
                        help="Decoder LSTM state dimension.")
    parser.add_argument("--enc-layers", type=int, default=1,
                        help="Number of encoder LSTM layers.")
    parser.add_argument("--dec-layers", type=int, default=1,
                        help="Number of decoder LSTM layers.")
    parser.add_argument("--beam-width", type=int, default=4,
                        help="Beam width for beam search decoding.")
    parser.add_argument("--k", type=int, default=1,
                        help="k for inverse sigmoid rollin schedule.")
    parser.add_argument("--patience", type=int, default=12,
                        help="Maximal patience for early stopping.")
    parser.add_argument("--epochs", type=int, default=60,
                        help="Maximal number of training epochs.")
    parser.add_argument("--batch-size", type=str, default=5,
                        help="Batch size.")
    parser.add_argument("--sed-discount", type=float, default=-0.999,
                        help="SED sparse EM discount.")
    parser.add_argument("--sed-em-iterations", type=int, default=10,
                        help="SED EM iterations.")

    args = parser.parse_args()
    main(args)

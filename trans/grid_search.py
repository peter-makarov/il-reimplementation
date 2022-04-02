"""CLI for performing grid search."""

import argparse
import json
import subprocess
import shutil
import os
import itertools
import time
import atexit
import re


BASH_EXECUTABLE = shutil.which("bash")


def cleanup():
    if 'process_list' in globals():
        for p in process_list:
            if p.poll() is None:
                p.kill()


atexit.register(cleanup)


def last_value_from_file(file_path: str, t=float):
    with open(file_path) as f:
        lines = f.readlines()
        return t(lines[-1].split()[-1])


def get_list(var):
    return var if isinstance(var, list) else [var]


def file_name_from_pattern(pattern: str, lang: str, split: str):
    file_name = pattern.replace("LANG", lang)
    file_name = file_name.replace("SPLIT", split)
    return file_name


def main(args: argparse.Namespace):
    os.makedirs(args.output, exist_ok=True)

    config_file = open(args.config)
    config_dict = json.load(config_file)

    process_list = []
    for name, grid_config in config_dict["grids"].items():
        os.makedirs(f"{args.output}/{name}")

        nm_pairs = [[(k, v) for v in get_list(grid_config[k])] for k in grid_config]
        combinations = itertools.product(*nm_pairs)

        # parse args
        args_list, comb_dict = [], {}
        for i, c in enumerate(combinations, 1):
            parsed_args, args_dict = [], {}
            for j in c:
                par_name, par_value = j
                if isinstance(par_value, bool) and par_value:
                    parsed_args.append(f"--{par_name}")
                elif par_name == 'sed-params':
                    continue
                else:
                    parsed_args.extend([f"--{par_name}", str(par_value)])
                args_dict[par_name] = par_value
            args_list.append(parsed_args)
            comb_dict[i] = args_dict

        with open(f"{args.output}/{name}/combinations.json", "w") as f:
            json.dump(comb_dict, f, indent=4)

        # train
        for i, args_ in enumerate(args_list, 1):
            for lang in config_dict["data"]["languages"]:
                for j in range(1, config_dict["runs_per_model"]+1):
                    output = f"{args.output}/{name}/{lang}/{i}/{i}.{j}"

                    if 'sed-params' in grid_config and lang in grid_config['sed-params']:
                        args_.extend(
                            [
                                "--sed-params", grid_config['sed-params'][lang]
                            ]
                        )

                    # create file names from pattern
                    train_file = file_name_from_pattern(config_dict['data']['pattern'], lang, 'train')
                    dev_file = file_name_from_pattern(config_dict['data']['pattern'], lang, 'dev')
                    test_file = file_name_from_pattern(config_dict['data']['pattern'], lang, 'test')

                    train = f"{config_dict['data']['path']}/{train_file}"
                    dev = f"{config_dict['data']['path']}/{dev_file}"
                    test = f"{config_dict['data']['path']}/{test_file}"

                    args_.extend(
                        [
                            "--output", output,
                            "--train", train,
                            "--dev", dev
                         ]
                    )

                    if os.path.exists(test):
                        args_.extend(
                            [
                                "--test", test
                            ]
                        )

                    p = subprocess.Popen(["python", "trans/train.py"]+args_)
                    process_list.append(p)

                    if len(process_list) < args.parallel_jobs:
                        continue

                    while len(process_list) >= args.parallel_jobs:
                        # check every few seconds
                        time.sleep(5)
                        process_list = [p for p in process_list if p.poll() is
                                        None]

    # all trainings in progress, stay in script so all processes can be aborted
    while len(process_list) > 0:
        process_list = [p for p in process_list if p.poll() is
                        None]
        # check every few seconds
        time.sleep(5)

    for name, grid_config in config_dict["grids"].items():
        for lang in config_dict["data"]["languages"]:

            results = []
            output_path = f"{args.output}/{name}/{lang}"
            for c_dir in os.listdir(output_path):
                dev_beam_avg, dev_greedy_avg = 0, 0
                test_beam_avg, test_greedy_avg = 0, 0

                # get beam size
                c_first_run = os.listdir(f"{output_path}/{c_dir}")[0]
                beam_match = re.search(r"beam[0-9]+", " ".join(os.listdir(f"{output_path}/{c_dir}/{c_first_run}")))

                # check if test files are evaluated
                test_match = "test_greedy.eval" in " ".join(os.listdir(f"{output_path}/{c_dir}/{c_first_run}"))

                c_dir_path = f"{output_path}/{c_dir}"
                n_runs = len(os.listdir(c_dir_path))
                for c_run in os.listdir(c_dir_path):
                    # dev greedy
                    dev_greedy_avg += last_value_from_file(f"{c_dir_path}/{c_run}/dev_greedy.eval")/n_runs

                    # dev beam
                    if beam_match:
                        dev_beam_avg += last_value_from_file(f"{c_dir_path}/{c_run}/dev_{beam_match[0]}.eval")/n_runs

                    if test_match:
                        # test greedy
                        test_greedy_avg += last_value_from_file(f"{c_dir_path}/{c_run}/test_greedy.eval")/n_runs
                        # test beam
                        if beam_match:
                            test_beam_avg += last_value_from_file(f"{c_dir_path}/{c_run}/test_{beam_match[0]}.eval")/n_runs

                result = {
                    'c_dir': c_dir,
                    'dev_greedy': round(dev_greedy_avg, 4),
                    'dev_beam': round(dev_beam_avg, 4) if beam_match else None,
                    'test_greedy': round(test_greedy_avg, 4) if test_match else None,
                    'test_beam': round(test_beam_avg, 4) if test_match and beam_match else None
                }
                results.append(result)

            with open(f"{args.output}/{name}/{lang}/results.txt", "w") as f:
                for r in sorted(results, key=lambda x: x['dev_greedy'], reverse=True):
                    f.write(r['c_dir']+"\n")
                    f.write(f"dev\ngreedy: {r['dev_greedy']}\n")
                    if r['dev_beam']:
                        f.write(f"beam: {r['dev_beam']}\n")
                    if r['test_greedy']:
                        f.write(f"test\ngreedy: {r['test_greedy']}\n")
                    if r['test_beam']:
                        f.write(f"beam: {r['test_beam']}\n\n")
                    else:
                        f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grid search.")

    parser.add_argument("--config", type=str, required=True,
                        help="Path to config file.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output directory.")
    parser.add_argument("--parallel-jobs", type=int,
                        default=30, help="Max number of parallel trainings.")

    args = parser.parse_args()
    main(args)

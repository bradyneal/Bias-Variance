import argparse
import re
import subprocess
import numpy as np
import os
import time

from fileio import MODEL_DIR

JOB_ID_REGEX = '.*Id: ([0-9]*)'
STATUS_REGEX = '.*Status: FINISHED_SUCCESS'


def add_to_log(log, job_id, seed, num_hidden, learning_rate):
    log['job_id'].append(job_id)
    log['seed'].append(seed)
    log['hidden_size'].append(num_hidden)
    log['learning_rate'].append(learning_rate)


def run_job_and_get_job_id(seed, num_hidden, learning_rate, other_things):
    args = ["--seed", str(seed), "--hidden_arr", str(num_hidden), "--learning_rate", str(learning_rate)] + other_things
    cmd = ["bash", "ngc_run.sh"] + args
    output = subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode('utf-8')
    job_id_str = re.match(JOB_ID_REGEX, output, re.DOTALL).group(1)
    return int(job_id_str)


def is_complete(job_id):
    output = subprocess.run(['ngc', 'batch', 'get', str(job_id)], stdout=subprocess.PIPE).stdout.decode('utf-8')
    match = re.match(STATUS_REGEX, output, re.DOTALL)
    return match is not None


def download_and_copy_job(job_id, first_job_id):
    subprocess.run(['ngc', 'result', 'download', str(job_id)], stdout=subprocess.PIPE)

    old_models_dir = os.path.join(str(job_id), 'models')

    p = re.compile('(.*job)0.pt')
    for file in os.listdir(old_models_dir):
        m = p.match(file)
        new_file = '{}{}.pt'.format(m.group(1), first_job_id)

        old_file_path = os.path.join(old_models_dir, file)
        os.rename(old_file_path, new_file)


def run_jobs(num_seeds, hidden_arr, num_learning_rates, other_things):
    log = {'job_id': [], 'seed': [], 'hidden_size': [], 'learning_rate': []}
    np.random.seed(0)
    learning_rates = [2 ** log_lr for log_lr in np.random.uniform(-25, 15, num_learning_rates)]

    print('Loop hidden:', hidden_arr)
    print('Loop learning rate:', learning_rates)
    print('Loop seed:', range(num_seeds))
    for num_hidden in hidden_arr:
        for learning_rate in learning_rates:
            for seed in range(num_seeds):
                job_id = run_job_and_get_job_id(seed, num_hidden, learning_rate, other_things)
                add_to_log(log, job_id, seed, num_hidden, learning_rate)
                print("Started job {}".format(job_id))
    first_job_id = log['job_id'][0]
    print('Log:', log)
    print('Loop hidden:', hidden_arr)
    print('Loop learning rate:', learning_rates)
    print('Loop seed:', range(num_seeds))

    prev_dir = os.getcwd()
    os.chdir(MODEL_DIR)

    while log['job_id']:
        for job_id in log['job_id'][:]:
            if is_complete(job_id):
                print("Finished job {}".format(job_id))
                download_and_copy_job(job_id, first_job_id)
                print("Downloaded and copied job {}".format(job_id))
                log['job_id'].remove(job_id)
        time.sleep(60)
        
    print('Log:', log)
    print('Loop hidden:', hidden_arr)
    print('Loop learning rate:', learning_rates)
    print('Loop seed:', range(num_seeds))

    os.chdir(prev_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Essential for each experiment
    parser.add_argument('--hidden_arr', nargs='+', type=int, default=[1, 2, 5, 25, 100, 500, 1000, 5000, 10000, 20000, 40000])
    parser.add_argument('--num_seeds', type=int, default=2)
    parser.add_argument('--num_learning_rates', type=int, default=100)

    # Takes other commands and passes it directly
    parser.add_argument('other_things', nargs=argparse.REMAINDER)

    args = parser.parse_args()

    run_jobs(args.num_seeds, args.hidden_arr, args.num_learning_rates, args.other_things[1:])
    print("COMPLETE")

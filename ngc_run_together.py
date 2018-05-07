import argparse
import re
import subprocess
import os
import time

from fileio import MODEL_DIR

JOB_ID_REGEX = '.*Id: ([0-9]*)'
STATUS_REGEX = '.*Status: FINISHED_SUCCESS'


def run_job_and_get_job_id(seed, num_hidden, other_things):
    args = ["--seed", str(seed), "--hidden_arr", str(num_hidden)] + other_things
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


def run_jobs(num_seeds, hidden_arr, other_things):
    job_ids = []
    for seed in range(num_seeds):
        for num_hidden in hidden_arr:
            job_id = run_job_and_get_job_id(seed, num_hidden, other_things)
            print("Started job {}".format(job_id))
            job_ids.append(job_id)
    first_job_id = job_ids[0]

    prev_dir = os.getcwd()
    os.chdir(MODEL_DIR)

    while job_ids:
        for job_id in job_ids[:]:
            if is_complete(job_id):
                print("Finished job {}".format(job_id))
                download_and_copy_job(job_id, first_job_id)
                print("Downloaded and copied job {}".format(job_id))
                job_ids.remove(job_id)
        time.sleep(60)

    os.chdir(prev_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Essential for each experiment
    parser.add_argument('--hidden_arr', nargs='+', type=int, default=[1, 2, 5, 25])
    parser.add_argument('--num_seeds', type=int, default=2)

    # Takes other commands and passes it directly
    parser.add_argument('other_things', nargs=argparse.REMAINDER)

    args = parser.parse_args()

    run_jobs(args.num_seeds, args.hidden_arr, args.other_things[1:])
    print("COMPLETE")

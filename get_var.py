from variance import Variance
from plot_bias_and_variance import load_probabilities_and_get_variances, plot_variances_with_diffs, \
    load_probabilities_and_get_biases, plot_biases_with_diffs
from rename_models import copy
import argparse

parser = argparse.ArgumentParser(description='hyperparam tuning for hidden size')
parser.add_argument('--step', type=int, help='which step to perform (0 = rename files, 1 = eval probs, '
                                             '2 = compute bias and vars, 3 = plot bias and vars)')

args = parser.parse_args()

HIDDEN_ARRS = [1, 2, 5, 25, 100, 500, 2500, 10000]  # [1, 2, 5, 25, 100, 10000, 2500, 500]
num_seeds = 50

print('performing step {}'.format(args.step))
if args.step == 0:
    # rename files
    SLURM_IDS = ['8060967', '8060973', '8060976', '8060984', '8061005', '8061013', '8061017', '8061046']
    for slurm_id in SLURM_IDS:
        copy(slurm_id, '0000000', 0, 0)

if args.step == 1:
    #  gets probs
    Variance(HIDDEN_ARRS, num_seeds, '0000000', 'evaluate_probabilities').calculate_plot_and_return_variances()

if args.step == 2:
    #  compute bias and variance
    load_probabilities_and_get_variances('0000000', HIDDEN_ARRS)
    load_probabilities_and_get_biases('0000000', HIDDEN_ARRS)

if args.step == 3:
    #  plot bias and variance
    plot_variances_with_diffs(slurm_id='0000000', hidden_arr=HIDDEN_ARRS, title='LBFGS Variances', label="Network Size")
    plot_biases_with_diffs(slurm_id='0000000', hidden_arr=HIDDEN_ARRS, title='LBFGS Variances', label="Network Size")

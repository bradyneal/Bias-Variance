from variance import Variance
from plot_bias_and_variance import load_probabilities_and_get_variances, plot_variances_with_diffs, \
    plot_variances_with_diffs_together
from rename_models import copy

HIDDEN_ARRS = [1, 2, 5, 25, 100, 500, 2500, 10000]  # [1, 2, 5, 25, 100, 10000, 2500, 500]
num_seeds = 50

# rename files
#SLURM_IDS = ['179703', '179704', '179705', '179706', '179707', '179715', '179716', '179717']
#for slurm_id in SLURM_IDS:
#    copy(slurm_id, '000000', 0, 0)


#  gets probs
#Variance(HIDDEN_ARRS, num_seeds, '000000', 'evaluate_probabilities').calculate_plot_and_return_variances()
load_probabilities_and_get_variances('000000', HIDDEN_ARRS)

plot_variances_with_diffs(slurm_id='000000', hidden_arr=HIDDEN_ARRS, title='LBFGS Variances', label="None")

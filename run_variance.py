from variance import Variance
from plot_bias_and_variance import load_probabilities_and_get_variances, plot_variances_with_diffs, load_probabilities_and_get_biases, plot_biases_with_diffs

num_layers = [1,2,3,4,5,6,7,8,9,10]
num_seeds = 50
slurm_id = 190771

inter = 0
data_type = 'evaluate_probabilities'

print(slurm_id)
Variance(100, num_layers, num_seeds, slurm_id, data_type, inter=inter).calculate_plot_and_return_variances()

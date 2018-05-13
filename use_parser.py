from parsing import parse_validations_table
import os

slurm_files = []

for file in slurm_files:
    formatted_output, output, hidden_size = parse_validations_table(os.path.join(os.getcwd(), 'slurm-175169.out'))
    print(formatted_output)
    print(hidden_size)
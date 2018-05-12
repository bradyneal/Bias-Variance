import argparse
from models import ShallowNet
from DataModelComp import DataModelComp

parser = argparse.ArgumentParser()

# Essential for each experiment
parser.add_argument('--hidden_arr', nargs='+', type=int, default=[1, 2, 5, 25])
parser.add_argument('--seed', type=int)
parser.add_argument('--num_seeds', type=int, default=2)

# Needed for hyperparameter tuning
parser.add_argument('--learning_rate', nargs='+', type=float, default=[0.1])
parser.add_argument('--momentum', type=float, default=0.9)

# Parameters for different experiments
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--size_of_one_pass', type=int)
parser.add_argument('--variance_over', choices=["all", "initialization", "sampling"], default="all")
parser.add_argument('--num_train_after_split', type=int)
parser.add_argument('--no_bootstrap', action="store_true")
parser.add_argument('--print_errors', choices=["all", "train_and_val"], default="all")
parser.add_argument('--max_epochs', type=int, default=50)
parser.add_argument('--save_best_model', action="store_true")

# Parameter to split it
parser.add_argument('--start_seed', type=int, default=0)

# Default is good almost always
parser.add_argument('--save_model', choices=["only_end", "every_epoch"], default="only_end")

# Default values are good always
parser.add_argument('--no_cuda', action="store_true")

# Essentially useless for now
parser.add_argument('--decay', action="store_true", default=False)
parser.add_argument('--gamma', type=float, default=0.1)


args = parser.parse_args()

if args.print_errors == "all":
    print_all_errors = True
    print_only_train_and_val_errors = False

if args.print_errors == "train_and_val":
    print_all_errors = False
    print_only_train_and_val_errors = True

if args.seed is not None:
    seeds = [args.seed]
else:
    seeds = range(args.start_seed, args.num_seeds)

for seed in seeds:
    for i, num_hidden in enumerate(args.hidden_arr):
        if len(args.learning_rate) == len(args.hidden_arr):
            lr = args.learning_rate[i]
        else:
            lr = args.learning_rate[0]
        print(DataModelComp(ShallowNet(num_hidden), epochs=args.max_epochs,
            run_i=seed, bootstrap=(not args.no_bootstrap), batch_size=args.batch_size,
            size_of_one_pass=args.size_of_one_pass,
            train_val_split_seed=0 if args.variance_over is "initialization" else seed,
            seed=0 if args.variance_over is "sampling" else seed,
            lr=lr, momentum=args.momentum,
            print_all_errors=print_all_errors, print_only_train_and_val_errors=print_only_train_and_val_errors,
            num_train_after_split=args.num_train_after_split, save_model=args.save_model, save_best_model=args.save_best_model,
            decay=args.decay, gamma=args.gamma, no_cuda=args.no_cuda).train())

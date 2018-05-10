from models import ShallowNet
from DataModelComp import DataModelComp

small_hidden_layer_sizes = [5, 25, 100, 1E3, 5E3, 10E3, 20E3, 40E3]  # 80E3 - this is temporarily out
intermediate_hidden_layer_sizes = [160E3, 320E3, 640E3, 1.28E6]
large_hidden_layer_sizes = [2.5E6, 5E6, 10E6]

print("Experiment to run code for small hidden layer sizes with 50 seeds\n")
for seed in range(50):
    for num_hidden in small_hidden_layer_sizes:
        print("Running for hidden size: %d" % (num_hidden))
        print(DataModelComp(ShallowNet(num_hidden), epochs=50, log_interval=None,
              run_i=seed, train_val_split_seed=seed, save_all_at_end=True, seed=seed,
              bootstrap=True, save_model_every_epoch=True, batch_size=100).train())

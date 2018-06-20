from models import ShallowNet, DeepNet
from DataModelComp import DataModelComp

# small_hidden_layer_sizes = [5, 25, 100, 1E3, 5E3, 10E3, 20E3, 40E3]  # 80E3 - this is temporarily out
# intermediate_hidden_layer_sizes = [160E3, 320E3, 640E3, 1.28E6]
# large_hidden_layer_sizes = [2.5E6, 5E6, 10E6]

num_layers = [15]
num_hiddens = [100]
learning_rates = [0.1] # [0.1,0.01,0.001,0.0001]

print("Experiment to run code for small hidden layer sizes with 50 seeds\n")
for seed in range(1):
    for lr in learning_rates:
        for num_hidden in num_hiddens:
            for num_layer in num_layers:
                print("Running for hidden size: %d and number of layers: %d" % (num_hidden,num_layer))
                print(DataModelComp(DeepNet(num_hidden,num_layer), epochs=250,log_interval=None,
              run_i=seed, train_val_split_seed=seed, seed=seed,
              bootstrap=True, batch_size=100, print_all_errors=True, plot_curves=True, lr=lr, num_train_after_split=None).train()[0].item())

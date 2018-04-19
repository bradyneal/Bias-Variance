from DataModelComp import DataModelComp
from models import ShallowNetCIFAR10, ThreeLayerNetCIFAR10, AlexNetCIFAR10, InceptionCIFAR10
from matplotlib import pyplot as plt
import pickle

deepish_net = ThreeLayerNetCIFAR10(num_hidden=512)
shallow_net = ShallowNetCIFAR10(num_hidden=512)
alex_net = AlexNetCIFAR10()
incp_net = InceptionCIFAR10(use_batch_norm=True)

#  Learning Curves
train_losses_list = []
for k in [0, 1]:
    data_model_comp = DataModelComp(deepish_net, batch_size=128, test_batch_size=128, epochs=60,
                                    lr=0.01, decay=True, step_size=1, gamma=0.95, momentum=0.9,
                                    no_cuda=False, seed=False, log_interval=100,
                                    run_i=0, save_interval=None, data='CIFAR10', corruption=k)
    _, _, train_losses = data_model_comp.train(eval_path=False, early_stopping=False)
    train_losses_list.append(train_losses)

with open('matt_folder/fig_a_series.pkl', 'wb') as f:
    pickle.dump(train_losses_list, f)

plt.figure()
plt.title('Learning Curves')
plt.plot(train_losses_list[0], color='blue', label='true labels')
plt.plot(train_losses_list[1], color='red', label='random labels')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.savefig('result_fig_1')

print('done?')

#  Label Corruption
label_corruption_list = []
for network in [deepish_net, alex_net, incp_net]:
    network_series = []
    for k in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
        data_model_comp = DataModelComp(network, batch_size=128, test_batch_size=128, epochs=1,
                                        lr=0.01, decay=True, step_size=1, gamma=0.95, momentum=0.9,
                                        no_cuda=False, seed=False, log_interval=100,
                                        run_i=0, save_interval=None, data='CIFAR10', corruption=k)
        _, _, train_losses = data_model_comp.train(eval_path=False, early_stopping=True)
        test_error = data_model_comp.evaluate_test(cur_iter=1)
        network_series.append(test_error)

    label_corruption_list.append(network_series)


# Check params
def get_weights(net):
    total_weights = 0
    for k in net.state_dict().keys():
        print(k)
        print(net.state_dict()[k].numel())
        total_weights += net.state_dict()[k].numel()
    print('total weights: {}'.format(total_weights))
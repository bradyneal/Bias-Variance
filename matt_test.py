from new_DataModelComp import DataModelComp
from models import ShallowNetCIFAR10, ThreeLayerNetCIFAR10, AlexNetCIFAR10, InceptionCIFAR10

deepish_net = ThreeLayerNetCIFAR10(num_hidden=512)
shallow_net = ShallowNetCIFAR10(num_hidden=512)
alex_net = AlexNetCIFAR10()
bf_net = InceptionCIFAR10()

data_model_comp = DataModelComp(bf_net, batch_size=16, test_batch_size=16, epochs=10,
                                lr=0.01, decay=False, step_size=10, gamma=0.1, momentum=0.5,
                                no_cuda=False, seed=False, log_interval=100,
                                run_i=0, save_interval=None, data='CIFAR10')

train_seq, test_seq = data_model_comp.train(eval_path=True)



print('done?')
# Check params
def get_weights(net):
    total_weights = 0
    for k in net.state_dict().keys():
        print(k)
        print(net.state_dict()[k].numel())
        total_weights += net.state_dict()[k].numel()
    print('total weights: {}'.format(total_weights))

get_weights(shallow_net)
get_weights(deepish_net)
get_weights(alex_net)
get_weights(bf_net)
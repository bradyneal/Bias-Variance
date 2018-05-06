i = 3
hidden_sizes = [5, 25, 100, 1E3, 5E3, 10E3, 20E3, 40E3, 80E3]
num_hidden = hidden_sizes[i]
seed = 0

def main(job_id, params):
    print('Anything printed here will end up in the output directory for job #%d' % job_id)
    lr = 2 ** params['loglr']
    print('learning rate:', lr)
    print('hidden size:', num_hidden)
    val_acc, _ = DataModelComp(ShallowNet(num_hidden), epochs=200, log_interval=None,
                               run_i=seed, train_val_split_seed=seed, save_all_at_end=True, seed=seed,
                               bootstrap=True, save_model_every_epoch=False, batch_size=10, num_train_after_split=100,
                               print_only_train_and_val_errors=True, print_all_errors=False, lr=lr, momentum=0.9).train()
    return 1 - val_acc

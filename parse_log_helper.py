def check_present(key, tuple):
    if key not in tuple:
        raise Exception('{} should have been in {}'.format(key, tuple))


def parse_seed(line):
    relevant_tuple = line.split(',')[-1]
    check_present('seed', relevant_tuple)
    return int(relevant_tuple.split(':')[1])


def parse_total_epochs(line):
    relevant_tuple = line.split(',')[3]
    check_present('epochs', relevant_tuple)
    return int(relevant_tuple.split(':')[1])


def parse_hidden_size(line):
    check_present('hidden size', line)
    return int(line.split(' ')[-1])


def parse_quantity_value(line, quantity_required):
    if quantity_required == 'loss':
        relevant_tuple = line.split(',')[1]
        check_present('Average loss', relevant_tuple)
        return float(relevant_tuple.split(':')[2])
    elif quantity_required == 'error':
        relevant_tuple = line.split(',')[2]
        check_present('Accuracy', relevant_tuple)
        accuracy = eval(relevant_tuple.split(' ')[1])
        return 1-accuracy
    else:
        raise Exception('quantity_required should not be {}'.format(quantity_required))


def parse_common_fields(line, types):
    current_epoch = int(line.split(' ')[1])
    for type in types:
        if type.name.lower() in line.lower():
            line_type = type
    return current_epoch, line_type


def parse_from_error_line(line, quantity_required, types):
    quantity_value = parse_quantity_value(line, quantity_required)
    current_epoch, type = parse_common_fields(line, types)
    return quantity_value, current_epoch, type

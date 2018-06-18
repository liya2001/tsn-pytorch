import torch


def load_pretrain(net, pth_path):
    checkpoint = torch.load(pth_path)

    net_dict = net.state_dict()

    pretrained_dict = {}
    count = 0
    for k, v in checkpoint.items():
        count = count + 1
        print(count, k)
        if 415 > count > 18:
            pretrained_dict.setdefault(k[7:], checkpoint[k])
        if count < 19:
            pretrained_dict.setdefault(k, checkpoint[k])

    net_dict.update(pretrained_dict)
    net.load_state_dict(net_dict)


def remove_freezed_params(policies):
    '''
    Remove freezed params from policies
    :param policies:
    :return:
    '''
    grad_policies = []

    for group in policies:
        grad_params = []
        for param in group['params']:
            if param.requires_grad is True:
                grad_params.append(param)

        if len(grad_params) > 0:
            group['params'] = grad_params
            grad_policies.append(group)

    return grad_policies

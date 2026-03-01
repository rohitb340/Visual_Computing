import torch
import copy
from ..utils.fedevi import fedevi_scoring_func

class FedAvgComm:
    def __init__(self, server_model=None):
        pass

    def __call__(self, clients, weights, server, round):
        return average(clients, weights, server)


class FedAvgMComm:
    def __init__(self, server_model: torch.nn.Module):
        self.beta = 0.9
        self.v = copy.deepcopy(server_model)
        for key in self.v.state_dict().keys():
            self.v.state_dict()[key].data.copy_(torch.zeros_like(self.v.state_dict()[key]).float())

    def __call__(self, clients, weights, server, round):
        server_copy = copy.deepcopy(server)
        server = average(clients, weights, server)

        for key in server.state_dict().keys():
            self.v.state_dict()[key].data.copy_(self.beta * self.v.state_dict()[key] +
                                                server_copy.state_dict()[key] - server.state_dict()[key])
            server.state_dict()[key].data.copy_(server_copy.state_dict()[key] - self.v.state_dict()[key])

        return server
    

class FedCLAMComm:
    def __init__(self, server_model: torch.nn.Module, config):
        self.client_num = len(config.INNER_SITES)
        dim = (self.client_num, config.TRAIN_ROUNDS)
        self.agg_lr = config.AGG_LR
        self.zero_init = config.ZERO_INIT
        self.dataset = config.DATASET
        self.of_penalties = torch.zeros(dim)
        self.val_loss_ratios = torch.zeros(dim)
        self.local_train_losses = torch.zeros(dim)
        self.local_init_val_losses = torch.zeros(dim)
        self.local_trained_val_losses = torch.zeros(dim)
        self.client_weight_speeds = {client_idx: {} for client_idx in range(self.client_num)}
        for client_idx in range(self.client_num):
            for key, v in server_model.state_dict().items():
                self.client_weight_speeds[client_idx][key] = torch.zeros_like(v).float()

    def __call__(self, clients, weights, server, round):
        for idx, c in enumerate(clients):
            self.local_init_val_losses[idx, round] = c['init_val_loss']
            self.local_trained_val_losses[idx, round] = c['trained_val_loss']
            self.local_train_losses[idx, round] = c['train_loss']
            self.val_loss_ratios[idx, round] = c['val_loss_ratio']
            self.of_penalties[idx, round] = c['of_penalty']

        server_copy = copy.deepcopy(server)
        fedavg_result = average([c['model'] for c in clients], [1/self.client_num]*self.client_num, server)
        pseudo_gradient = {key: server_copy.state_dict()[key] - fedavg_result.state_dict()[key] for key in server.state_dict().keys()}
        for key in server.state_dict().keys():
            if not (key.endswith("weight") or key.endswith("bias")):
                # debug, put into separate file these keys
                with open("non_weight_bias_keys.txt", "a") as f:
                    f.write(key + "\n")
                server.state_dict()[key].data.copy_(fedavg_result.state_dict()[key])  # copy the result of FedAvg
            else:
                for client_idx in range(self.client_num):
                    if round == 0:
                        if self.zero_init:
                            self.client_weight_speeds[client_idx][key].copy_(torch.zeros_like(pseudo_gradient[key]))
                        else:
                            # initialize the speed of the weights as per PyTorch implementation of momentum SGD
                            self.client_weight_speeds[client_idx][key].copy_(pseudo_gradient[key])
                    else:
                        mom = torch.clamp(self.val_loss_ratios[client_idx, round], min=0, max=1)
                        self.client_weight_speeds[client_idx][key].copy_(
                            mom * self.client_weight_speeds[client_idx][key] + (1 - self.of_penalties[client_idx, round]) * pseudo_gradient[key]
                        )
                avg_speed = torch.mean(torch.stack([self.client_weight_speeds[c_idx][key] for c_idx in range(self.client_num)]), dim=0)
                server.state_dict()[key].data.copy_(server_copy.state_dict()[key] - self.agg_lr * avg_speed)
        return server
    

class FedEviComm:
    def __init__(self, server_model: torch.nn.Module, config):
        self.client_num = len(config.INNER_SITES)
        dim = (self.client_num, config.TRAIN_ROUNDS)
        self.u_dis = torch.zeros(dim)
        self.u_data = torch.zeros(dim)


    def __call__(self, clients, weights, server, round):
        server = average([c['model'] for c in clients], weights, server)

        for client_idx in range(self.client_num):
            self.u_dis[client_idx, round], self.u_data[client_idx, round] = fedevi_scoring_func(
                server,
                clients[client_idx]['model'],
                clients[client_idx]['valid_loader'],
            )
        weights = torch.tensor(weights)
        weights += 1.0 * self.u_dis[:, round] / self.u_data[:, round]
        weights = weights / torch.sum(weights)
        server = average([c['model'] for c in clients], weights, server)

        return server


class FedDynComm:
    def __init__(self, server_model: torch.nn.Module):
        self.alpha = 0.1
        self.v = copy.deepcopy(server_model)
        for key in self.v.state_dict().keys():
            self.v.state_dict()[key].data.copy_(torch.zeros_like(self.v.state_dict()[key]).float())

    def __call__(self, clients, weights, server, round):
        server_copy = copy.deepcopy(server)
        server = average(clients, weights, server)

        for key in server.state_dict().keys():
            tmp = torch.zeros_like(server.state_dict()[key]).float()
            for client_idx in range(len(weights)):
                tmp += weights[client_idx] * clients[client_idx].state_dict()[key]

            v = self.v.state_dict()[key] - self.alpha * (tmp - server_copy.state_dict()[key])
            self.v.state_dict()[key].data.copy_(v)

            server.state_dict()[key].data.copy_(server.state_dict()[key] - 1.0 / self.alpha * v)

        return server


class FedFAComm:
    def __init__(self, server_model: torch.nn.Module):
        pass

    def __call__(self, clients, weights, server, round):
        for key in server.state_dict().keys():
            tmp = torch.zeros_like(server.state_dict()[key]).float()
            for client_idx in range(len(weights)):
                tmp += weights[client_idx] * clients[client_idx].state_dict()[key]
            server.state_dict()[key].data.copy_(tmp)

            if 'running_var_mean_bmic' in key or 'running_var_std_bmic' in key:
                tmp = []
                for client_idx in range(len(weights)):
                    tmp.append(clients[client_idx].state_dict()[key.replace('running_var_', 'running_')])

                tmp = torch.stack(tmp)
                var = torch.var(tmp)
                server.state_dict()[key].data.copy_(var)

                # wandb.log({'server.{}'.format(key): torch.norm(var).cpu().numpy()}, commit=False)

        return server


class FedBNComm:
    def __init__(self, server_model):
        pass

    def __call__(self, clients, weights, server, round):
        for key in server.state_dict().keys():
            if 'bn' not in key:
                tmp = torch.zeros_like(server.state_dict()[key])
                for client_idx in range(len(weights)):
                    tmp += weights[client_idx] * clients[client_idx].state_dict()[key]
                server.state_dict()[key].data.copy_(tmp)

        return server


def average(clients, weights, server):
    for key in server.state_dict().keys():
        tmp = torch.zeros_like(server.state_dict()[key]).float()
        for client_idx in range(len(weights)):
            tmp += weights[client_idx] * clients[client_idx].state_dict()[key]
        server.state_dict()[key].data.copy_(tmp)

    return server


class Comm:
    def __init__(self, server_model, config):
        if config.COMM_TYPE == 'FedAvg':
            self.comm_fn = FedAvgComm()
        elif config.COMM_TYPE == 'FedAvgM':
            self.comm_fn = FedAvgMComm(server_model)
        elif config.COMM_TYPE == 'FedCLAM':
            self.comm_fn = FedCLAMComm(server_model, config)
        elif config.COMM_TYPE == 'FedDyn':
            self.comm_fn = FedDynComm(server_model)
        elif config.COMM_TYPE == 'FedFA':
            self.comm_fn = FedFAComm(server_model)
        elif config.COMM_TYPE == 'FedBN':
            self.comm_fn = FedBNComm(server_model)
        elif config.COMM_TYPE == 'FedEvi':
            self.comm_fn = FedEviComm(server_model, config)

    def __call__(self, clients, weights, server, round):
        return self.comm_fn(clients, weights, server, round)

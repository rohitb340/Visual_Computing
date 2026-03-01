import os
import shutil

import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from pyfed.manager.comm import Comm
from pyfed.manager.helper.build_model import (
    build_model,
    build_client
)
from pyfed.utils.log import print_log


class Manager(object):
    def __init__(self, config):
        self.config = config
        self.best_acc = 0
        self.best_epoch = 0

        self._setup()

    def _setup(self):
        self.server_model = build_model(self.config)

        pytorch_total_params = sum(p.numel() for p in self.server_model.parameters())
        total_params_size = abs(pytorch_total_params * 4. / (1024 ** 2.))
        print(f'Network: {self.config.NETWORK}, Params size (MB): {total_params_size}')

        self._build_clients()
        self.comm = Comm(self.server_model, self.config)
        self.writer = SummaryWriter(log_dir=self.config.DIR_LOG)


    def _build_clients(self):
        client_class = build_client(self.config)
        print_log(f'Client type: {self.config.CLIENT}')

        if self.config.TRAIN_MODE == 'centralized':
            self.central = client_class(self.config, self.config.INNER_SITES, self.server_model)
        else:
            self.inner_clients = [client_class(self.config, site, self.server_model)
                                  for site in self.config.INNER_SITES]

            train_nums = [len(client.train_loader.dataset)  for client in self.inner_clients]
            self.client_weights = [num * 1.0 / sum(train_nums) for num in train_nums]
            print(self.client_weights)
            if len(self.config.OUTER_SITES) > 0:
                self.outer_clients = [client_class(self.config, site, self.server_model)
                                      for site in self.config.OUTER_SITES]

    def train(self):
        metrics = {}
        best_server_val_acc, best_server_val_round = 0, 0
        best_person_val_acc, best_person_val_round = 0, 0
        best_server_test_acc, best_server_test_round = 0, 0

        for iter_round in range(self.config.TRAIN_ROUNDS):
            for iter_local in range(self.config.TRAIN_EPOCH_PER_ROUND):
                print_log(f"============ Train epoch: {iter_local + iter_round * self.config.TRAIN_EPOCH_PER_ROUND}/{self.config.TRAIN_ROUNDS} ============")

                for ci, client in enumerate(self.inner_clients):
                    train_loss, train_acc = client.train(server_model=self.server_model)

                    print_log(f'site-{client.name:<10s}| train loss: {train_loss:.4f} | train acc: {train_acc:.4f}')
                    metrics['train_loss_' + client.name] = train_loss
                    self.writer.add_scalar(f'Train_Loss/{client.name}', train_loss, iter_round)
                    metrics['train_acc_' + client.name] = train_acc
                    self.writer.add_scalar(f'Train_Acc/{client.name}', train_acc, iter_round)

            # client to server
            if self.config.COMM_TYPE == 'FedCLAM':
                # clients = [client.client_to_server() for client in self.inner_clients]
                clients = []
                for client in self.inner_clients:
                    client_info = client.client_to_server()
                    clients.append(client_info)
                    self.writer.add_scalar(f'Init_Val_Loss/{client.name}', client_info['init_val_loss'], iter_round)
                    metrics['init_val_loss_' + client.name] = client_info['init_val_loss']
                    self.writer.add_scalar(f'Trained_Val_Loss/{client.name}', client_info['trained_val_loss'], iter_round)
                    metrics['trained_val_loss_' + client.name] = client_info['trained_val_loss']
                    self.writer.add_scalar(f'Val_Loss_Ratio/{client.name}', client_info['val_loss_ratio'], iter_round)
                    metrics['val_loss_ratio_' + client.name] = client_info['val_loss_ratio']
                    self.writer.add_scalar(f'OF_Penalty/{client.name}', client_info['of_penalty'], iter_round)
                    metrics['of_penalty_' + client.name] = client_info['of_penalty']
                client_weights = self.client_weights
                self.server_model = self.comm(clients, client_weights, self.server_model, iter_round)
            elif self.config.COMM_TYPE == 'FedEvi':
                clients = [client.client_to_server() for client in self.inner_clients]
                client_weights = self.client_weights
                self.server_model = self.comm(clients, client_weights, self.server_model, iter_round)
            else:
                client_models = [client.client_to_server()['model'] for client in self.inner_clients]
                client_weights = self.client_weights
                self.server_model = self.comm(client_models, client_weights, self.server_model, iter_round)

            # run global validation and testing
            print_log('============== Server Model Validation ==============')
            val_accs, test_accs = [], []
            for ci, client in enumerate(self.inner_clients):
                val_loss, val_acc = client.val(self.server_model)
                test_loss, test_acc = client.test(self.server_model)
                val_accs.append(val_acc)
                test_accs.append(test_acc)

                metrics['server_val_loss_on_' + client.name] = val_loss
                metrics['server_val_acc_on_' + client.name] = val_acc
                metrics['server_test_acc_on_' + client.name] = test_acc

                print_log('site-{:<10s}| val loss: {:.4f} | val acc: {:.4f} | test acc: {:.4f}'.format(
                    client.name, val_loss, val_acc, test_acc))
                
                self.writer.add_scalar(f'Server_Val_Loss/{client.name}', val_loss, iter_round)
                self.writer.add_scalar(f'Server_Val_Acc/{client.name}', val_acc, iter_round)
                self.writer.add_scalar(f'Server_Test_Acc/{client.name}', test_acc, iter_round)

            metrics['server_val_acc_avg'] = np.mean(val_accs)
            metrics['server_test_acc_avg'] = np.mean(test_accs)
            self.writer.add_scalar('Server_Val_Acc/Avg', np.mean(val_accs), iter_round)
            self.writer.add_scalar('Server_Test_Acc/Avg', np.mean(test_accs), iter_round)
            print_log('avg val acc: {:.4f} | avg test acc: {:.4f}'.format(np.mean(val_accs), np.mean(test_accs)))

            # for FedCLAM, we already do local validation in client.train
            if self.config.COMM_TYPE != 'FedCLAM':
                print_log('============== Local Validation ==============')
                val_accs, test_accs = [], []
                for ci, client in enumerate(self.inner_clients):
                    val_loss, val_acc = client.val()
                    test_loss, test_acc = client.test()
                    val_accs.append(val_acc)
                    test_accs.append(test_acc)

                    metrics['l_val_loss_' + client.name] = val_loss
                    metrics['l_val_acc_' + client.name] = val_acc
                    metrics['l_test_acc_' + client.name] = test_acc

                    print_log('site-{:<10s}| val loss: {:.4f} | val acc: {:.4f} | test acc: {:.4f}'.format(
                        client.name, val_loss, val_acc, test_acc))

                metrics['l_val_acc_avg'] = np.mean(val_accs)
                metrics['l_test_acc_avg'] = np.mean(test_accs)
                print_log('avg val acc: {:.4f} | avg test acc: {:.4f}'.format(np.mean(val_accs), np.mean(test_accs)))
                if metrics['l_val_acc_avg'] > best_person_val_acc:
                    best_person_val_acc = metrics['l_val_acc_avg']
                    best_person_val_round = iter_round
            if metrics['server_val_acc_avg'] > best_server_val_acc:
                best_server_val_acc = metrics['server_val_acc_avg']
                best_server_val_round = iter_round

            if metrics['server_test_acc_avg'] > best_server_test_acc:
                best_server_test_acc = metrics['server_test_acc_avg']
                best_server_test_round = iter_round

            print_log('============== Summary ==============')
            print_log(f'best server val round: {best_server_val_round} | best server val acc: {best_server_val_acc:.4f}')
            print_log(f'best server test round: {best_server_test_round} | best server test acc: {best_server_test_acc:.4f}')
            # print_log('best person val round: {} | best person val acc: {:.4f}'.format(best_person_val_round, best_person_val_acc))

            # server to client
            for ci, client in enumerate(self.inner_clients):
                client.server_to_client(self.server_model)
            
            np.save(os.path.join(self.config.DIR_LOG, f'metrics_{iter_round}.npy'), metrics)
            self.save(iter_round, np.mean(val_accs))

    def train_inner_outer(self):
        metrics = {}
        for iter_round in range(0, self.config.TRAIN_ROUNDS):
            # train each client for one round
            for iter_local in range(self.config.TRAIN_EPOCH_PER_ROUND):
                print_log("============ Train epoch: {}/{} ============".format(
                    iter_local + iter_round * self.config.TRAIN_EPOCH_PER_ROUND, self.config.TRAIN_ROUNDS))

                for ci, client in enumerate(self.inner_clients):
                    if self.config.CLIENT == 'fedprox' or self.config.CLIENT == 'fedproxcls':
                        train_loss, train_acc = client.train(server_model=self.server_model)
                    else:
                        train_loss, train_acc = client.train()

                    print_log('site-{:<10s}| train loss: {:.4f} | train acc: {:.4f}'.format(
                        client.name, train_loss, train_acc))

                    metrics['train_loss_' + client.name] = train_loss
                    metrics['train_acc_' + client.name] = train_acc

            # client to server
            client_models = [client.client_to_server()['model'] for client in self.inner_clients]
            client_weights = self.client_weights

            self.server_model = self.comm(client_models, client_weights, self.server_model, self.config.COMM_TYPE)

            # run global validation and testing
            print_log('============== {} =============='.format('Inner Testing'))
            val_accs, test_accs = [], []
            for ci, client in enumerate(self.inner_clients):
                val_loss, val_acc = client.val(self.server_model)
                test_loss, test_acc = client.test(self.server_model)
                val_accs.append(val_acc)
                test_accs.append(test_acc)

                metrics['g_inner_val_loss_' + client.name] = val_loss
                metrics['g_inner_val_acc_' + client.name] = val_acc
                metrics['g_inner_test_acc_' + client.name] = test_acc

                print_log('site-{:<10s}| val loss: {:.4f} | val acc: {:.4f} | test acc: {:.4f}'.format(
                    client.name, val_loss, val_acc, test_acc))

            metrics['g_inner_val_acc_avg'] = np.mean(val_accs)
            metrics['g_inner_test_acc_avg'] = np.mean(test_accs)
            print_log(
                '[inner] avg val acc: {:.4f} | avg test acc: {:.4f}'.format(np.mean(val_accs), np.mean(test_accs)))

            print_log('============== {} =============='.format('Outer Testing'))
            val_accs, test_accs = [], []
            for ci, client in enumerate(self.outer_clients):
                val_loss, val_acc = client.val(self.server_model)
                test_loss, test_acc = client.test(self.server_model)
                val_accs.append(val_acc)
                test_accs.append(test_acc)

                metrics['g_outer_val_loss_' + client.name] = val_loss
                metrics['g_outer_val_acc_' + client.name] = val_acc
                metrics['g_outer_test_acc_' + client.name] = test_acc

                print_log('site-{:<10s}| val loss: {:.4f} | val acc: {:.4f} | test acc: {:.4f}'.format(
                    client.name, val_loss, val_acc, test_acc))

            metrics['g_outer_val_acc_avg'] = np.mean(val_accs)
            metrics['g_outer_test_acc_avg'] = np.mean(test_accs)
            print_log(
                '[outer] avg val acc: {:.4f} | avg test acc: {:.4f}'.format(np.mean(val_accs), np.mean(test_accs)))

            # server to client
            for ci, client in enumerate(self.inner_clients):
                client.server_to_client(self.server_model)

            self.save(iter_round, np.mean(val_accs))

            # wandb.log(metrics)

    def train_centralized(self):
        metrics = {}
        for iter_round in range(0, self.config.TRAIN_ROUNDS):
            # train each client for one round
            for iter_local in range(self.config.TRAIN_EPOCH_PER_ROUND):
                print_log("============ Train epoch: {}/{} ============".format(
                    iter_local + iter_round * self.config.TRAIN_EPOCH_PER_ROUND, self.config.TRAIN_ROUNDS))

                train_loss, train_acc = self.central.train()

                print_log('central| train loss: {:.4f} | train acc: {:.4f}'.format(train_loss, train_acc))

                metrics['train_loss_central'] = train_loss
                metrics['train_acc_central'] = train_acc

            # run global validation and testing
            print_log('============== {} =============='.format('Global Validation'))
            val_loss_list, val_acc_list = self.central.val()
            test_loss_list, test_acc_list = self.central.test()

            for site_idx, site in enumerate(self.config.INNER_SITES):
                metrics['g_val_loss_' + site] = val_loss_list[site_idx]
                metrics['g_val_acc_' + site] = val_acc_list[site_idx]
                metrics['g_test_acc_' + site] = test_acc_list[site_idx]

                print_log('site-{:<10s}| val loss: {:.4f} | val acc: {:.4f} | test acc: {:.4f}'.format(
                    site, val_loss_list[site_idx], val_acc_list[site_idx], test_acc_list[site_idx]))

            metrics['l_val_acc_avg'] = np.mean(val_acc_list)
            metrics['l_test_acc_avg'] = np.mean(test_acc_list)
            print_log(
                'avg val acc: {:.4f} | avg test acc: {:.4f}'.format(np.mean(val_acc_list), np.mean(test_acc_list)))

            self.save(iter_round, np.mean(val_acc_list))

            # wandb.log(metrics)

    def train_individual(self):
        metrics = {}
        for iter_round in range(0, self.config.TRAIN_ROUNDS):
            # train each client for one round
            for iter_local in range(self.config.TRAIN_EPOCH_PER_ROUND):
                print_log("============ Train epoch: {}/{} ============".format(
                    iter_local + iter_round * self.config.TRAIN_EPOCH_PER_ROUND, self.config.TRAIN_ROUNDS))

                for ci, client in enumerate(self.inner_clients):
                    train_loss, train_acc = client.train()

                    print_log('site-{:<10s}| train loss: {:.4f} | train acc: {:.4f}'.format(
                        client.name, train_loss, train_acc))

                    metrics['train_loss_' + client.name] = train_loss
                    metrics['train_acc_' + client.name] = train_acc

            # run global validation and testing
            print_log('============== {} =============='.format('Global Validation'))
            val_accs, test_accs = [], []
            for ci, client in enumerate(self.inner_clients):
                val_loss, val_acc = client.val()
                test_loss, test_acc = client.test()
                val_accs.append(val_acc)
                test_accs.append(test_acc)

                metrics['l_val_loss_' + client.name] = val_loss
                metrics['l_val_acc_' + client.name] = val_acc
                metrics['l_test_acc_' + client.name] = test_acc

                print_log('site-{:<10s}| val loss: {:.4f} | val acc: {:.4f} | test acc: {:.4f}'.format(
                    client.name, val_loss, val_acc, test_acc))

            metrics['l_val_acc_avg'] = np.mean(val_accs)
            metrics['l_test_acc_avg'] = np.mean(test_accs)
            print_log('avg val acc: {:.4f} | avg test acc: {:.4f}'.format(np.mean(val_accs), np.mean(test_accs)))

            self.save(iter_round, np.mean(val_accs))

            # wandb.log(metrics)

    def save(self, iter_round, val_acc):
        better = False
        if val_acc > self.best_acc:
            self.best_acc = val_acc
            self.best_epoch = iter_round
            better = True

        # save server status
        save_dicts = {'server': self.server_model.state_dict(),
                      'best_epoch': self.best_epoch,
                      'best_acc': self.best_acc,
                      'round': iter_round}

        if self.config.TRAIN_MODE == 'centralized':
            save_dicts['optim'] = self.central.client_to_server()['optimizer'].state_dict()
        else:
            # save local status
            for ci, client in enumerate(self.inner_clients):
                save_dicts[f'optim_{ci}'] = client.client_to_server()['optimizer'].state_dict()

        torch.save(save_dicts, os.path.join(self.config.DIR_CKPT, 'model_latest.pth'))
        if better:
            shutil.copyfile(os.path.join(self.config.DIR_CKPT, 'model_latest.pth'),
                            os.path.join(self.config.DIR_CKPT, 'model_best.pth'))


    def finish(self):
        self.writer.close()
        # in self.config.DIR_LOG we have one npy file for each round. We now merge them into one file and delete the others
        metrics = {}
        for iter_round in range(self.config.TRAIN_ROUNDS):
            metrics[iter_round] = np.load(os.path.join(self.config.DIR_LOG, f'metrics_{iter_round}.npy'), allow_pickle=True).item()
            os.remove(os.path.join(self.config.DIR_LOG, f'metrics_{iter_round}.npy'))
        np.save(os.path.join(self.config.DIR_LOG, 'metrics.npy'), metrics)

        print_log('Finished training')
        if self.config.COMM_TYPE == 'FedCLAM':
            import matplotlib.pyplot as plt
            vlr = self.comm.comm_fn.val_loss_ratios.cpu().numpy()
            of = self.comm.comm_fn.of_penalties.cpu().numpy()
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            for client_idx in range(vlr.shape[1]):
                ax[0].plot(np.arange(len(vlr)), vlr[:, client_idx], label=f"Client {client_idx}")
                ax[1].plot(np.arange(len(of)), of[:, client_idx], label=f"Client {client_idx}")
            ax[0].set_title("VLR")
            ax[1].set_title("OF")
            plt.legend()
            plt.savefig(os.path.join(self.config.DIR_LOG, "vlr_of.png"))
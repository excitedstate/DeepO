import joblib
import torch

import numpy as np
import pickle
import torch.utils.data
import torch.nn as nn
import torch.optim as optim

from blitz.modules import BayesianLSTM
from blitz.utils import variational_estimator

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# @variational_estimator
# class CostEstimationNet(nn.Module):
#     def __init__(self):
#         super(CostEstimationNet, self).__init__()
#         self.relu = nn.ReLU()
#         self.lstm_1 = BayesianLSTM(79, 32, prior_sigma_1=1, prior_pi=1, posterior_rho_init=-3.0)
#         self.linear_1 = nn.Linear(32, 16)
#         self.linear_2 = nn.Linear(16, 1)
#         self.drop_out = nn.Dropout(0.2)
#
#     def forward(self, x):
#         x_, _ = self.lstm_1(x)
#
#         # gathering only the latent end-of-sequence for the linear layer
#         x_ = x_[:, -1, :]
#         x_ = self.relu(x_)
#         x_ = self.linear_1(x_)
#         x_ = self.relu(x_)
#         x_ = self.drop_out(x_)
#         x_ = self.linear_2(x_)
#         return x_
@variational_estimator
class CostEstimationNet(nn.Module):
    def __init__(self):
        super(CostEstimationNet, self).__init__()
        self.lstm_1 = BayesianLSTM(79, 10, prior_sigma_1=1, prior_pi=1, posterior_rho_init=-3.0)
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        x_, _ = self.lstm_1(x)

        # gathering only the latent end-of-sequence for the linear layer
        x_ = x_[:, -1, :]
        x_ = self.linear(x_)
        return x_


class CostEstimationTrainer:
    # # Hyper Parameter
    LEARNING_RATE = 0.001
    EPOCHS = 5000
    BATCH_SIZE = 64

    def __init__(self, _fp_plan_sequences_output, _fp_cost_labels, _fp_std_scalar_labels):
        """

        Args:
            _fp_plan_sequences_output:
            _fp_cost_labels:
        """
        self.sequences, self.cost_labels = self.load_data(
            _fp_plan_sequences_output,
            _fp_cost_labels)

        self.x_train, self.x_test, self.y_train, self.y_test = self.preprocess(_fp_std_scalar_labels)
        self.fp_std_scalar_labels = _fp_std_scalar_labels
        self.sc = None
        self.net = None

    def flow(self, _fp_model_save_prefix):
        self.preprocess(self.fp_std_scalar_labels)
        self.train(_fp_model_save_prefix)
        self.evaluate()

    def train(self, _fp_model_save_prefix):
        _dataloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(self.x_train, self.y_train),
            batch_size=self.BATCH_SIZE,
            shuffle=True)
        _net = CostEstimationNet().float()
        self.net = _net
        _criterion = nn.MSELoss()
        _optimizer = optim.Adam(_net.parameters(), lr=self.LEARNING_RATE)
        _iteration = 0
        for _epoch_no in range(self.EPOCHS):
            for _batch_no, (_batch_x, _batch_y) in enumerate(_dataloader):
                _optimizer.zero_grad()
                _loss = _net.sample_elbo(inputs=_batch_x,
                                         labels=_batch_y,
                                         criterion=_criterion,
                                         sample_nbr=3,
                                         complexity_cost_weight=1 / self.x_train.shape[0])
                _loss.backward()
                _optimizer.step()
                _iteration += 1
                if _iteration % 100 == 0:
                    # # 在测试集上预测并输出
                    _pred_test = _net(self.x_test)[:, 0].unsqueeze(1)
                    _loss_test = _criterion(_pred_test, self.y_test)
                    print(f"Epoch: {_epoch_no} Iteration: {_iteration} "
                          f"Train-loss {_loss:.4f}  Val-loss: {_loss_test:.4f}")
            torch.save(_net, f"{_fp_model_save_prefix}-{_epoch_no}")

    def preprocess(self, _fp_std_scalar_labels):
        # # Transform features by scaling each feature to a given range.
        # # first parameter default to (0, 1)
        # # cost labels scale到 0 1 区间
        sc = MinMaxScaler()
        self.cost_labels = sc.fit_transform(self.cost_labels)
        joblib.dump(sc, _fp_std_scalar_labels, compress=True)
        # #
        max_length = max(map(lambda sequence: np.shape(sequence)[0], self.sequences), default=0)

        # for sequence in self.sequences:
        #     if np.shape(sequence)[0] > max_length:
        #         max_length = np.shape(sequence)[0]
        # # 扩充sequences, 让输入维度一致
        padded_sequences = []
        for seq in self.sequences:
            if len(seq) < max_length:
                # # 列表前部添加这么多, 可替换
                # tmp = [*[([0] * 79) for _ in range(max_length - len(seq))], *seq]
                tmp = [[0] * 79] * (max_length - len(seq))
                tmp.extend(seq)
                padded_sequences.append(tmp)
            else:
                # # len(seq) == max_length
                padded_sequences.append(seq)

        # # 转换成tensor
        padded_sequences = torch.tensor(np.array(padded_sequences), dtype=torch.float32)
        cost_label = torch.tensor(self.cost_labels, dtype=torch.float32)

        x_train, x_test, y_train, y_test = train_test_split(padded_sequences,
                                                            cost_label,
                                                            test_size=0.25,
                                                            random_state=42,
                                                            shuffle=False)
        self.sc = sc
        return x_train, x_test, y_train, y_test

    def evaluate(self):
        idx = 2
        assert self.sc is not None, "self.sc is None"
        assert self.net is not None, "self.net is None"
        pred, pred_vals = self.pred_cost(self.x_test[idx].unsqueeze(0),
                                         _sc=self.sc, _net=self.net)

        unscaled_y, in_range, y_under_upper, y_above_lower = self.evaluate_pred_vals(pred_vals, self.y_test[idx], 2,
                                                                                     _sc=self.sc)

        upper, lower = self.get_intervals(pred_vals)

        print("label: ", unscaled_y)
        print("prediction: ", pred)
        print("prediction upper bound: ", upper)
        print("prediction lower bound: ", lower)
        print("label in prediction range: ", in_range)
        # %%
        cnt = 0
        for idx in range(len(self.x_test)):
            pred, pred_vals = self.pred_cost(self.x_test[idx].unsqueeze(0),
                                             _sc=self.sc, _net=self.net)
            unscaled_y, in_range, y_under_upper, y_above_lower = self.evaluate_pred_vals(pred_vals, self.y_test[idx], 2,
                                                                                         _sc=self.sc)
            if in_range:
                cnt += 1
        print(cnt / len(self.x_test))

    @staticmethod
    def load_data(_fp_plan_sequences_output, _fp_cost_labels):
        # # 这是plan_to_seq的输出
        with open(_fp_plan_sequences_output, "rb") as f:
            sequences = pickle.load(f)
        cost_labels = np.load(_fp_cost_labels).reshape(-1, 1)
        print("Data loaded.")
        return sequences, cost_labels

    @staticmethod
    def pred_cost(x, sample_nbr=100, _sc: MinMaxScaler = None, _net: nn.Module = None):
        pred_vals = [_net(x).cpu().item() for _ in range(sample_nbr)]
        pred_mean = np.mean(pred_vals)
        pred_mean = _sc.inverse_transform(pred_mean.reshape(1, 1))[0][0]
        pred_vals = _sc.inverse_transform(np.array(pred_vals).reshape(1, -1))
        return pred_mean, pred_vals

    @staticmethod
    def evaluate_pred_vals(pred_vals, scaled_y, std_multiplier=2, _sc: MinMaxScaler = None):
        # print(scaled_y)
        y = _sc.inverse_transform(scaled_y.reshape(1, 1))[0][0]
        mean = np.mean(pred_vals)
        std = np.std(pred_vals)
        ci_upper = mean + (std_multiplier * std)
        ci_lower = mean - (std_multiplier * std)
        ic_acc = (ci_lower <= y) * (ci_upper >= y)
        return y, ic_acc, (ci_upper >= y), (ci_lower <= y)

    @staticmethod
    def get_intervals(pred_vals, std_multiplier=2):
        mean = np.mean(pred_vals)
        std = np.std(pred_vals)

        upper_bound = mean + (std * std_multiplier)
        lower_bound = mean - (std * std_multiplier)

        return upper_bound, lower_bound

import torch
import typing
import pickle
import os.path
import logging
import numpy as np
from blitz.modules import BayesianLSTM
from blitz.utils import variational_estimator
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from src.basic import GeneralLogger
from src.config import DATA_PATH_MODEL, DATA_PATH_PIC, DATA_PATH_PKL, DATA_PATH
from src.query_plan import QueryPlanNode, QueryPlan

from src.embedding import Embedding, EmbeddingNet4Nodes

CostLearnerLogger = GeneralLogger(name="embedding", stdout_flag=True, stdout_level=logging.INFO,
                                  file_mode="a")


class CostLearnerDataLoader:
    def __init__(self, data_x: typing.Union[str, np.ndarray], data_y: typing.Union[str, np.ndarray], *,
                 test_size=0.1, batch_size=10):
        self.data_x = np.load(data_x) if isinstance(data_x, str) else data_x
        self.data_y = np.load(data_y) if isinstance(data_y, str) else data_y
        assert len(self.data_x), 'data x is not loaded'
        assert len(self.data_y), 'data y is not loaded'
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            torch.tensor(self.data_x, dtype=torch.float32),
            torch.tensor(self.data_y, dtype=torch.float32).unsqueeze(1),
            test_size=test_size,
            shuffle=False  # # 不许shuffle, 不许random
        )
        self.data_set = TensorDataset(self.x_train, self.y_train)
        self.dataloader_train = DataLoader(self.data_set, batch_size=batch_size, shuffle=True)

    def __iter__(self):
        return iter(self.dataloader_train)


@variational_estimator
class CostLearnerNet(torch.nn.Module):
    def __init__(self, input_shape):
        """
            每条数据包含多个节点, 每个节点的特征维度为 input_shape
        Args:
            input_shape:
        """
        super(CostLearnerNet, self).__init__()
        self.lstm = BayesianLSTM(input_shape, 32, prior_sigma_1=1, prior_pi=1, posterior_rho_init=-3.0)
        self.relu_1 = torch.nn.ReLU()
        self.linear_1 = torch.nn.Linear(32, 16)
        self.relu_2 = torch.nn.ReLU()
        self.drop_out = torch.nn.Dropout(0.2)
        self.output_layer = torch.nn.Linear(16, 1)

    def forward(self, out):
        out, _ = self.lstm(out)

        # gathering only the latent end-of-sequence for the linear layer
        out = out[:, -1, :]
        out = self.relu_1(out)
        out = self.linear_1(out)
        out = self.relu_2(out)
        out = self.drop_out(out)
        out = self.output_layer(out)
        return out


class NetTrainer:
    def __init__(self, data_x: typing.Union[str, np.ndarray], data_y: typing.Union[str, np.ndarray], *,
                 test_size=0.1, batch_size=10, input_shape: float = None, epochs=10, lr=0.01):
        self.data_loader = CostLearnerDataLoader(data_x, data_y, test_size=test_size, batch_size=batch_size)
        self.net = CostLearnerNet(input_shape if input_shape is not None else self.data_loader.data_x.shape[2]).float()
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.epochs = epochs

    def train(self):
        count = 0
        x_axis_data, y_axis_data = [], []
        for epoch in range(self.epochs):
            for i, (datapoints, labels) in enumerate(self.data_loader):
                self.optimizer.zero_grad()

                loss = self.net.sample_elbo(inputs=datapoints,
                                            labels=labels,
                                            criterion=self.criterion,
                                            sample_nbr=3,
                                            complexity_cost_weight=1 / self.data_loader.x_train.shape[0])
                loss.backward()
                self.optimizer.step()

                count += 1
                x_axis_data.append(count)
                y_axis_data.append(loss.detach().numpy())
                if count % 20 == 0:
                    output = self.net(self.data_loader.x_test)
                    output = output[:, 0].unsqueeze(1)
                    loss_test = self.criterion(output, self.data_loader.y_test)
                    CostLearnerLogger.info(f"Epoch: {epoch} Iteration: {count} "
                                           f"Train-loss {loss.detach().numpy():.4f}  Val-loss: {loss_test:.4f}")
                    self.save_model(os.path.join(DATA_PATH_MODEL, f"cost_model_{epoch}.model"))
        return x_axis_data, y_axis_data

    def save_model(self, file_path: str):
        torch.save(self.net, file_path)

    def flow(self, plot=True):
        x_axis_data, y_axis_data = self.train()
        if plot:
            import matplotlib.pyplot as plt

            plt.plot(x_axis_data, y_axis_data)
            plt.title("loss history")
            plt.savefig(os.path.join(DATA_PATH_PIC, "loss_history.png"))


class CostLearner:
    """
        完成训练之后, 现在应当能对给定的查询计划进行代价预测了!
        对于任意的查询计划, 该类的一个实例可以给出查询代价的预测值
    """

    def __init__(self, vocab_encoding_dict: typing.Union[str, dict[str, np.ndarray]],
                 node_embedding_net_load_from: str, cost_learner_net_load_from: str, eval_mode=True):
        self.vocab_encode_mapping = pickle.load(open(vocab_encoding_dict, "rb"))
        self.embedding_net = EmbeddingNet4Nodes(load_from=node_embedding_net_load_from)
        # self.embedding_net.model.summary()
        self.cost_learner_net = torch.load(cost_learner_net_load_from)
        if eval_mode:
            self.cost_learner_net.eval()

    def __call__(self, _query_plan: typing.Union[np.ndarray, QueryPlan, torch.Tensor] = None) -> float:
        """
            对于任意的查询计划, 该类的一个实例可以给出查询代价的预测值
        """
        embedded_vec = self.embed(_query_plan) if isinstance(_query_plan, QueryPlan) else _query_plan
        _out = self.cost_learner_net(torch.tensor(embedded_vec, dtype=torch.float32).unsqueeze(0))
        return _out.detach().numpy()[0][0]

    def embed(self, qp: typing.Union[typing.List[typing.List[str]], QueryPlan], need_cost_label: bool = False):
        node_list = self.get_node_list_of_query_plan(qp) if isinstance(qp, QueryPlan) else qp
        # # 2. 做查询计划嵌入, 节点中的每一个词
        plan_seq = Embedding.vectorize_sentences(list(map(lambda x: x[:-1], node_list)),
                                                 self.vocab_encode_mapping)
        # # 扩充
        plan_seq = pad_sequences(plan_seq, padding="post", dtype=np.float32,
                                 maxlen=self.embedding_net.model.input_shape[1])
        embedded_vector = self.embedding_net(plan_seq)
        return (embedded_vector, 0) if not need_cost_label else (embedded_vector, node_list[-1][-1])

    @staticmethod
    def get_node_list_of_query_plan(qp: QueryPlan):
        def __callback(cur_node: QueryPlanNode, _list: list):
            _list.append(cur_node.to_vector())

        # # 1. 获取查询计划内所有被执行节点的描述
        node_list = list()
        qp.post_order(_callback=__callback, _list=node_list)
        # # 在pycharm外运行 设置draw = False
        qp.make_digraph(draw=False)
        return node_list

    @staticmethod
    def predict_cost_and_get_confidence_intervals(query_plan: QueryPlan, n_sample=10, ci_multiplier=6,
                                                  c: 'CostLearner' = None):
        """
            mean, std, diff, lower bound, upper bound, in_or_not
        """
        c = c if c is not None else CostLearner.default_factory()
        embedded_vec, cost = c.embed(query_plan, True)
        cost_norm = (5346925.973 - float(cost)) / 5346925.973
        pred_test = np.array(list(map(lambda x: c(embedded_vec), range(n_sample))))
        pred_test_mean = pred_test.mean()
        pred_std = pred_test.std(ddof=1)
        return dict(
            mean=pred_test_mean,
            std=pred_std,
            lower_bound=pred_test_mean - (pred_std * ci_multiplier),
            upper_bound=pred_test_mean + (pred_std * ci_multiplier),
            diff=pred_test_mean - cost_norm,
            ans=(pred_test_mean - (pred_std * ci_multiplier)) <= cost_norm <= (
                    pred_test_mean + (pred_std * ci_multiplier))
        )

    @staticmethod
    def predict_cost_and_get_confidence_intervals_eval_mode(query_plan: QueryPlan, n_sample=10,
                                                            ci_multiplier=6,
                                                            c: 'CostLearner' = None):
        """
            mean, std, lower bound, upper bound, func
        """
        c = c if c is not None else CostLearner.default_factory()
        embedded_vec, _ = c.embed(query_plan, False)
        pred_test = np.array(list(map(lambda x: c(embedded_vec), range(n_sample))))
        pred_test_mean = pred_test.mean()
        pred_std = pred_test.std(ddof=1)
        return dict(
            qp=query_plan,
            mean=pred_test_mean,
            std=pred_std,
            lower_bound=pred_test_mean - (pred_std * ci_multiplier),
            upper_bound=pred_test_mean + (pred_std * ci_multiplier),
            inverse_function=lambda cost: 5346925.973 - float(cost) * 5346925.973
        )

    @staticmethod
    def test_single():
        query_plan_path = r"C:\Users\QQ863\Documents\Projects\PycharmProjects\DeepO\data\plan\5c"
        query_plan = Embedding.load_from(_query_plan_path=query_plan_path)
        c = CostLearner.default_factory()
        c.predict_cost_and_get_confidence_intervals(query_plan)

    @staticmethod
    def default_factory():
        c = CostLearner(os.path.join(DATA_PATH_PKL, "vocab_encode.pkl"),
                        os.path.join(DATA_PATH_MODEL, "embedding_all_nodes.h5"),
                        os.path.join(DATA_PATH_MODEL, "cost_model_9.model"))
        return c

    @staticmethod
    def get_width_of_confidence_intervals():
        """
            结果
            |p|满足条件的比例|
            |---|---|
            |6 |0.1504424778761062|
            |7 |0.19469026548672566|
            |8 |0.18584070796460178|
            |9 |0.25663716814159293|
            |10| 0.26548672566371684|
            |11| 0.2831858407079646|
            |12| 0.3805309734513274|
            |13| 0.4424778761061947|
            |14| 0.4336283185840708|
            |15| 0.6017699115044248|
            |16| 0.672566371681416|
            |17| 0.6283185840707964|
            |18| 0.8230088495575221|
            |19| 0.7522123893805309|
            |20| 0.8407079646017699|
            |21| 0.8938053097345132|
            |22| 0.8407079646017699|
            |23| 0.8938053097345132|
            |24| 0.9469026548672567|
            |25| 0.9203539823008849|
            |26| 0.9292035398230089|
            |27| 0.9557522123893806|
            |28| 0.9469026548672567|
            |29| 0.9734513274336283|
        """
        from src.config import DATA_PATH_PLANS_FOR_TRAIN
        # 加载查询计划
        qp_list = []
        for _dir in [os.path.join(DATA_PATH, "synthetic_plan"), DATA_PATH_PLANS_FOR_TRAIN]:
            for query_plan_path in os.listdir(_dir):
                full_name = os.path.join(_dir, query_plan_path)
                with open(full_name, "r", encoding="utf-8") as f:
                    qp = Embedding.load_from(_query_plan_raw=f.read())
                    qp_list.append(qp)

        c = CostLearner.default_factory()
        # 计算ci_multiplier: 29

        res = list(map(lambda qp: c.predict_cost_and_get_confidence_intervals(qp, ci_multiplier=0), qp_list))
        np.save('res', res)
        # # 计算
        max_p = 40
        # res = np.load('res.npy', allow_pickle=True)
        res_ratio = {p: len(list(filter(lambda qp_res: abs(abs(qp_res['diff']) <= qp_res['std'] * p), res))) / len(res)
                     for p in
                     range(max_p)}
        for p, v in res_ratio.items():
            if v > 0.9:
                return res, res_ratio, p
        # for p in range(6, 30):
        #     print()
        # for p in range(6, 30):
        #     ans = list(map(lambda qp: c.predict_cost_and_get_confidence_intervals(qp, ci_multiplier=p)['ans'], qp_list))
        #
        #     # print(p, len(list(filter(bool, ans))) / len(ans))
        #     if len(list(filter(bool, ans))) / len(ans) > 0.95:
        #         return p
        return res, res_ratio, max_p

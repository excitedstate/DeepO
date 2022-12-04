"""
    查询计划的嵌入
    - 叶子节点嵌入
"""
import functools
import logging
import os.path
import pickle
import time
import typing

import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import models, layers
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from src.basic import GeneralLogger
from src.config import DATA_PATH_PLANS_FOR_TRAIN, DATA_PATH_TXT, DATA_PATH_MODEL, DATA_PATH_NPY, DATA_PATH_PKL, \
    DATA_PATH
from src.query_plan import QueryPlan, QueryPlanNode, CostInfo

EmbeddingLogger = GeneralLogger(name="embedding", stdout_flag=True, stdout_level=logging.INFO,
                                file_mode="a")


class EmbeddingNet4Nodes:
    """
        所有的节点都在这里做统一的信息编码
    """

    def __init__(self, input_shape: typing.Tuple[int, int] = None, load_from: str = None):
        assert input_shape is not None or load_from is not None, "none of parameters is not None"
        if load_from is not None:
            self.model = models.load_model(load_from)
        else:
            self.model = models.Sequential()
            self.model.add(layers.SimpleRNN(128, return_sequences=True, activation='relu', input_shape=input_shape))
            self.model.add(layers.SimpleRNN(128, return_sequences=True, activation='relu'))
            self.model.add(layers.Flatten())
            self.model.add(layers.Dense(128, activation='relu'))
            self.model.add(layers.Dense(64, activation='relu'))
            self.model.add(layers.Dense(1, activation='relu'))
            self.model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])
        # self.model.summary()
        self.intermediate_layer_model = models.Model(inputs=self.model.input, outputs=self.model.layers[4].output)

    def fit(self, x: np.array, y: np.array):
        self.model.fit(x, y, validation_split=0.2, epochs=10, batch_size=128, shuffle=True)

    def __call__(self, test_x: np.array):
        return self.intermediate_layer_model.predict(test_x)

    def save_model(self, prefix: str, save_dir: str):
        self.model.save(os.path.join(save_dir, f"{prefix}-{time.time_ns()}.h5"))


class Embedding:
    def __init__(self, _query_plan_dir_for_train=DATA_PATH_PLANS_FOR_TRAIN):
        self.query_plan_dir_for_train = _query_plan_dir_for_train

    @staticmethod
    def load_from(_query_plan_tree: QueryPlan = None,
                  _query_plan_raw: str = None,
                  _query_plan_path: str = None):
        query_plan_tree = _query_plan_tree
        query_plan_raw = _query_plan_raw
        query_plan_path = _query_plan_path
        if query_plan_tree is None:
            if query_plan_raw is None:
                if query_plan_path is None:
                    assert False, "no valid parameters"
                else:
                    with open(query_plan_path, "r", encoding="utf-8") as _query_plan_file:
                        query_plan_raw = _query_plan_file.read()
            assert query_plan_raw is not None, "query plan raw is None"
            query_plan_tree = QueryPlan(query_plan_raw)
        assert query_plan_tree is not None, "query plan tree is None"
        return query_plan_tree

    @staticmethod
    def get_data_for_node_embedding(_query_plan_dir_for_train: str, *path) -> tuple[
        np.ndarray, np.ndarray, list[list[list[str]]], dict[str, np.array], list[list[str]],
        list[float], set[str]
    ]:
        """
        Args:
            _query_plan_dir_for_train: default to `DATA_PATH_PLANS_FOR_TRAIN`
            path: 其他的路径
        Returns:
            data_x: sentences的编码表
            data_y: 对cost做log变化后的值
            plan_list: 查询计划树
            vocab_encode_mapping: 词的编码表
            sentences: 节点句子表
            cost_labels: cost原始值
            vocabs: 所有词汇的表
        """

        def __callback(cur_node: QueryPlanNode, _list: list):
            _list.append(cur_node.to_vector())

        plan_list = list()
        nodes_list = list()
        for _dir in [_query_plan_dir_for_train, *path]:
            for file_name in os.listdir(_dir):
                node_list = list()
                with open(os.path.join(_dir, file_name)) as query_plan_file:
                    qp = QueryPlan(query_plan_file.read())
                    qp.post_order(_callback=__callback, _list=node_list)
                    qp.make_digraph(draw=False)
                    plan_list.append(node_list)
                    nodes_list.extend(node_list)

        # # 拿到处理后的数据
        sentences = list(map(lambda l: list(l[:-1]), nodes_list))
        cost_labels = list(map(lambda l: l[-1], nodes_list))
        # # 处理Sentences, 获得词表,
        vocabs = set(functools.reduce(lambda x, y: x + y, sentences, list()))
        max_actual_cost, min_actual_cost = max(CostInfo.ACTUAL_COST_RECORDS), min(CostInfo.ACTUAL_COST_RECORDS)
        actual_cost_range = max_actual_cost - min_actual_cost
        EmbeddingLogger.info(len(vocabs))
        EmbeddingLogger.info(f"max cost: {max_actual_cost}", )
        EmbeddingLogger.info(f"min cost: {min_actual_cost}", )
        # # 获得编码表, 可通过词汇获取到编码
        label_encoder = LabelEncoder().fit_transform(list(vocabs))
        encoded = to_categorical(label_encoder)
        vocab_encode_mapping = {v: np.reshape(np.array(e), (1, len(vocabs))) for v, e in zip(vocabs, encoded)}

        data_x = Embedding.vectorize_sentences(sentences, vocab_encode_mapping)
        # data_x = np.array(data_x)
        data_x = pad_sequences(data_x, padding="post", dtype=np.float32, maxlen=max(map(len, sentences)))
        # # 处理cost_labels数据
        data_y = np.array(list(map(lambda cost: (max_actual_cost - float(cost)) / actual_cost_range, cost_labels)))
        # data_y = np.array(list(map(lambda cost: np.log(float(cost)), cost_labels)))
        return data_x, data_y, plan_list, vocab_encode_mapping, sentences, cost_labels, vocabs

    @staticmethod
    def vectorize_sentences(sentences: list[list[str]], vocab_encode_mapping: dict):
        return list(map(
            lambda sentence: Embedding.vectorize_sentence(sentence, vocab_encode_mapping),
            sentences))

    @staticmethod
    def vectorize_sentence(sentence: list[str], vocab_encode_mapping: dict):
        # # 这里可能出错误, 全1表示缺失值 无法编码
        return list(map(
            lambda word: vocab_encode_mapping.get(word, np.ones(len(vocab_encode_mapping))).reshape(
                len(vocab_encode_mapping), ),
            sentence)
        )

    def plan_embedding(self, plan_list: list[list[list[str]]],
                       vocab_encode_mapping: dict[str, np.array],
                       net: EmbeddingNet4Nodes = None, net_load_from=None):
        assert plan_list, "plan list is null"
        assert net is not None or net_load_from is not None, "net is null"
        if net is None:
            net = EmbeddingNet4Nodes(load_from=net_load_from)
            assert net is not None, f"net can not be load from {net_load_from}"
        root_cost_labels = []
        plan_sequences = []
        for *other, root_node_abs in plan_list:
            root_cost_labels.append(float(root_node_abs[-1]))
            plan_seq = self.vectorize_sentences(list(map(lambda x: x[:-1], [*other, root_node_abs])),
                                                vocab_encode_mapping)
            # # 扩充
            plan_seq = pad_sequences(plan_seq, padding="post", dtype=np.float32, maxlen=net.model.input_shape[1])
            plan_sequences.append(net(plan_seq))
        # # 最后在执行一次填充
        data_x = pad_sequences(plan_sequences, padding="post", dtype=np.float32, maxlen=max(map(len, plan_sequences)))
        data_y = np.array(root_cost_labels)
        # #
        data_y = (np.max(data_y) - data_y) / (np.max(data_y) - np.min(data_y))
        # data_y = np.array(list(map(lambda cost: np.log(float(cost)), data_y)))
        return data_x, data_y

    def train_embedding_net(self, data_x, data_y):
        net = EmbeddingNet4Nodes(input_shape=data_x.shape[1:])
        net.model.summary()
        net.fit(data_x, data_y)
        net.save_model("embedding_all_nodes", DATA_PATH_MODEL)
        EmbeddingLogger.info("embedding over")

        # # reload
        # plan_list = pickle.load(open(os.path.join(DATA_PATH_PKL, "plan_list.pkl"), "rb"))
        # vocab_encode_mapping = pickle.load(open(os.path.join(DATA_PATH_PKL, "vocab_encode.pkl"), "rb"))
        # print(*plan_list[0], sep='\n')
        # net = EmbeddingNet4Nodes(load_from="embedding_all_nodes.h5")
        # net.model.summary()
        return net

    def flow(self):
        data_x, data_y, plan_list, vocab_encode_mapping, *_ = self.get_data_for_node_embedding(
            self.query_plan_dir_for_train, os.path.join(DATA_PATH, "synthetic_plan"))
        EmbeddingLogger.info(f"data X shape: ({data_x.shape}")
        EmbeddingLogger.info(f"data y shape: ({data_y.shape}")
        pickle.dump(plan_list, open(os.path.join(DATA_PATH_PKL, "plan_list.pkl"), "wb"))
        pickle.dump(vocab_encode_mapping, open(os.path.join(DATA_PATH_PKL, "vocab_encode.pkl"), "wb"))

        net = self.train_embedding_net(data_x, data_y)
        plan_sequences, root_cost_labels = self.plan_embedding(plan_list, vocab_encode_mapping, net)
        EmbeddingLogger.info(f"cost learner, data X shape: ({plan_sequences.shape}")
        EmbeddingLogger.info(f"cost learner, data y shape: ({root_cost_labels.shape}")
        # # ((113, 35, 64) 和 ((113,)
        np.save(os.path.join(DATA_PATH_NPY, "cost_learner_x"), plan_sequences)
        np.save(os.path.join(DATA_PATH_NPY, "cost_learner_y"), root_cost_labels)
        return plan_sequences, root_cost_labels

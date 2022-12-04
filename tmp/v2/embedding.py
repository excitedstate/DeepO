"""
    本脚本用于完成查询计划的EMBEDDING

    测试完成
    输出文件

        data/output.npy

        data/cost_label.npy
        data/job-cardinality-sequence.pkl
"""
import csv
import functools
import logging
import os
import pickle

import unicodedata

import numpy as np

from tensorflow.keras import preprocessing
from tensorflow.keras import models

from tensorflow.keras import optimizers, layers, utils

from sklearn.preprocessing import LabelEncoder

from src.basic import GeneralLogger
from src.config import DATA_PATH

EmbeddingLogger = GeneralLogger(name="embedding", stdout_flag=True, stdout_level=logging.INFO,
                                file_mode="a")


class ScanEmbedding:
    """
        用于完成叶子节点(Scan)的嵌入, 关键函数是get_data_and_label
    """

    def __init__(self, _fpath_column_min_max_vals,
                 _dir_query_plan):
        """
            初始化
        Args:
            _fpath_column_min_max_vals: 包含某些表主键上的V(R, A)信息的文件,
                来自项目: learned cardinalities
            _dir_query_plan: 从JOB项目的各个SQL执行语句生成的查询计划
        """
        self._fpath_cardinality_estimation = _fpath_column_min_max_vals
        self._fpath_query_plan = _dir_query_plan

        self.column_min_max_vals = self.read_column_min_max_vals(_fpath_column_min_max_vals)

    @staticmethod
    def build_model(_max_len, _vocab_size):
        model = models.Sequential()
        # # 两层循环神经网络, max_len是一个sentence的最大长度, vocab_size 是句子中词的数量, one-hot编码
        # #  根本就用不了这么多层, 输出即可
        model.add(layers.SimpleRNN(128, return_sequences=True, activation='relu',
                                   input_shape=(_max_len, _vocab_size + 1)))
        # # Fully-connected RNN where the output is to be fed back to input.
        model.add(layers.SimpleRNN(128, return_sequences=True, activation='relu'))
        # # 展平
        model.add(layers.Flatten())
        # # 三层全连接层
        model.add(layers.Dense(128, activation='relu'))  # ,kernel_regularizer=regularizers.l2(0.01)))
        model.add(layers.Dense(64, activation='relu'))  # ,kernel_regularizer=regularizers.l2(0.01)))
        model.add(layers.Dense(1, activation='relu'))

        model.compile(optimizer=optimizers.Adagrad(lr=0.001), loss='mse', metrics=['mse', 'mae'])
        model.summary()
        return model

    def train(self, _save_path):
        """
            1. 获取基础数据
        Args:
            _save_path:

        Returns:

        """
        # # 1. 获取所有查询计划中的句子, 每个Seq Scan所涉及的行数 以及 词汇的编码表
        sentences, rows, vocab_dict = self.get_vocabularies(self._fpath_query_plan)
        # # 2. 获取长度最长的句子和词的数量
        max_len = max(map(len, sentences), default=0)
        vocab_size = len(vocab_dict)

        with open(os.path.join(DATA_PATH, "vocab_dict.pkl"), "wb") as _pkl_f:
            pickle.dump(vocab_dict, _pkl_f)
        # # 3. 获取神经网络模型: 两层 RNN, 三层全连接
        model = self.build_model(max_len, vocab_size)
        # # 4. 获取编码后的数据和标签(log rows), data维度是不一样的 len(query_path)(112) * len(sentences[i](4...20)) * len(vocab_dict)(98)
        data, labels, labels_norm = self.prepare_data_and_label(sentences, rows, vocab_dict)
        # # pad_sequences 序列转化为经过填充以后的一个长度相同的新序列
        # # 现在大小 len(query_path)(112) * max_len(20) * len(vocab_dict)(98)
        padded_sentences = preprocessing.sequence.pad_sequences(data, maxlen=max_len, padding='post',
                                                                dtype='float32')
        # # 做数据拟合, 输入为 112 * 20 * 98, 输出为 rows (112)
        model.fit(padded_sentences, labels_norm, validation_split=0.2, epochs=100, batch_size=128,
                  shuffle=True)
        # '../model/embedding_model.h5'
        # # 保存这个模型, 输入为特定的编码后的Seq Scan, 输出为该Seq Scan查询所涉及的行数
        model.save(_save_path)

    def embed(self, _save_path_output, _save_path_vectors, _save_path_labels, _test_query_plan, _model_save_path):
        """
            "../data/vectors.csv"
            "../data/labels.csv"
            "../data/{}.npy".format(test_path.split("/")[-2])
        Args:
            _model_save_path:
            _test_query_plan:
            _save_path_output:
            _save_path_vectors:
            _save_path_labels:

        Returns:

        """
        # For validation & feature extraction, 加载的是同一个plan_path
        # # 就是重做了以便, 数据集和测试集都是一个
        _test_query_plan = self._fpath_query_plan if _test_query_plan == '' else _test_query_plan
        test_sentences, test_rows, _test_vocab_dict = self.get_vocabularies(_test_query_plan)
        test_data, test_label, _test_norm = self.prepare_data_and_label(test_sentences, test_rows, _test_vocab_dict)
        test_max_len = max(map(len, test_sentences), default=0)
        test_padded_sentences = preprocessing.sequence.pad_sequences(test_data, maxlen=test_max_len,
                                                                     padding='post', dtype='float32')

        # # 事实上, 这两个参数应当是相同的
        model = self.build_model(test_max_len, len(_test_vocab_dict))
        model.load_weights(_model_save_path)
        # # 就是借用一下, 用的不是最后的输出, 然后倒数第二层的输出, 全连接层(128->64), 因此输出维度为(64 * 112)
        intermediate_layer_model = models.Model(inputs=model.input,
                                                outputs=model.layers[4].output)
        # # 输出预测结果
        intermediate_output = intermediate_layer_model.predict(test_padded_sentences)
        # # 保存嵌入结果, 这就是之后要用的结果
        np.save(_save_path_output, intermediate_output)

        # # For vector demo
        # scan_labels = [['table', 'detail']]
        # for sentence in test_sentences:
        #     tmp = []
        #     # # 这也不知道是个什么原理
        #     table = sentence[0]
        #     tmp.append(table)
        #     if len(sentence) > 2:
        #         tmp.append(' '.join(str(each) for each in sentence[2:]))
        #     else:
        #         tmp.append(table)
        #     scan_labels.append(tmp)
        #
        # np.savetxt(_save_path_vectors, intermediate_output, delimiter="\t")
        # np.savetxt(_save_path_labels, scan_labels, fmt='%s', delimiter="\t")

    def flow(self, _model_save_path, _save_path_output, _save_path_vectors, _save_path_labels):
        self.train(_model_save_path)
        self.embed(_save_path_output, _save_path_vectors, _save_path_labels, '', _model_save_path)

    def get_data_and_label(self, _plans_dir: str):
        """
            这一段应该就是 本脚本的核心内容了, 用于获取训练数据
            todo 其实就是获取 Seq Scan 语句中的核心信息, 如 相关的表、别名以及过滤器
            1. plans先做了一次排序, 不知道什么意思
            2. 对每个查询计划, 执行以下步骤
                2.1 读取查询计划内容, 遍历给出的每一个步骤, 如果不包含 Seq Scan, 则继续执行
                    ... 就是找到叶子节点, 因为 Seq Scan只会出现在叶子节点上,
                    但是否可能是别的节点, 还不知道， Seq Scan形如
                    Seq Scan on movie_info_idx mi_idx  (cost=0.00..25185.44 rows=460242 width=4) (actual time=0.023..286.648 rows=459925 loops=1)
                2.2 获取 计划 消耗的时间 和 真实消耗的时间, 就是从括号中的字符串中匹配数字并返回
                2.2 获取table名称, 就是上方说的 movie_info_idx,
                2.3 sentences中除了表名 还带有数据, 如 cost, rows width等

        Args:
            _plans_dir: 是 plan_path 或者 test_plan

        Returns:
            sentences 表 / 表的别名, 谓词
            rows 返回涉及到的真实的行数
            pg 返回查询涉及到的行数, 估计的
        """
        sentences = []
        rows = []
        # # sorted是保证顺序的
        for file in sorted(os.listdir(_plans_dir)):
            with open(os.path.join(_plans_dir, file), "r", encoding="utf-8") as f:
                # # 1. 一行一行处理, 最后两行不处理
                plan = f.readlines()
                for i in range(len(plan) - 2):
                    # # 2. 处理每一行查询计划, 发现Seq Scan
                    if "Seq Scan" in plan[i] and 'never executed' not in plan[i]:
                        # # 2.1 获取 计划 消耗的时间 和 真实消耗的时间, 只是代码中只用了 _a_rows
                        _, _, _, _, _, _, _a_rows = self.extract_time(plan[i])
                        # # 2.2 获取table, 都是3
                        _temp_parts = plan[i].strip().split("  ")[1].split(" ")
                        _sentence = " ".join(_temp_parts[:-1]) + " "
                        if "Parallel" in plan[i]:
                            table = _temp_parts[5]
                        else:
                            table = _temp_parts[4]
                        # # 把 Seq Scan的下一行也带上, 应该是要 Filter的
                        if "Filter" in plan[i + 1]:
                            _sentence += plan[i + 1].strip()
                        else:
                            # # 不带Filter, 用sentence自己补位
                            _sentence += table
                            _sentence = _sentence + ' ' + _sentence
                        # # 将sentence中的分隔符去掉 还有 bpchar和 numeric等等修饰, 最后还是把 Seq Scan on 去掉了
                        _sentence = _sentence \
                            .replace("(", "").replace(")", "").replace("'", "") \
                            .replace(",", " ").replace("\\", "") \
                            .replace("::bpchar", "").replace("::numeric", "").replace("::text[]", "") \
                            .replace("::text", "").replace(":", "") \
                            .replace("Parallel", "").replace("Seq Scan on ", "").strip()
                        # # 对 _sentences 中的每个词都进行处理
                        # # 如果是数字, 进行正则化, 如果不是, 那就直接添加
                        # # 可以改写成 map
                        # # todo 这里还可以改进, 最好是自动机实现解析
                        sentence = []
                        ll = _sentence.split(" ")
                        # # 数字, 正则化
                        for cnt in range(len(ll)):
                            if self.is_not_number(ll[cnt]):
                                sentence.append(ll[cnt])
                            else:
                                # # 这里是有BUG的
                                sentence.append(
                                    self.normalize_data(ll[cnt], table + '.' + str(ll[cnt - 2]),
                                                        self.column_min_max_vals))

                        sentences.append(sentence)
                        rows.append(_a_rows)
        return sentences, rows

    def prepare_data_and_label(self, sentences, rows, _vocab_dict):
        """
            labels 就是 rows
            labels_norm 就是 rows 做了一下正则化，就是全体做了一个log
            data 对sentences中的每个词都用_vocab_dict进行编码(多加了一列), 形成一行数据
            [Xi;yi] = [v_i1_encode, v_i2_encode, v_i3_encode..., v_in_encode; log(rows[i])]
        Args:
            sentences:
            rows:
            _vocab_dict:

        Returns: data, labels, labels_norm...

        """
        vocab_size = len(_vocab_dict)
        data = []
        labels = []
        for sentence, row in zip(sentences, rows):
            _s = []
            for word in sentence:
                if self.is_not_number(word):
                    # # 不是数字, 可以编码
                    # # ... column_stack 做列拼接，这一行相当于在之前加了一个0
                    _tmp = np.column_stack((np.array([0]), _vocab_dict[word]))
                    # # ... 变成一维数组
                    _tmp = np.reshape(_tmp, (vocab_size + 1))
                    # # ... 这什么tpzfp行为,
                    # assert (len(_tmp) == vocab_size + 1)
                    _s.append(_tmp)
                else:
                    # # 是数字, 编码?
                    # # 用相同的数字填充
                    _tmp = np.full(vocab_size + 1, word)

                    # assert len(_tmp) == vocab_size + 1
                    _s.append(_tmp)
            data.append(np.array(_s))
            labels.append(row)
        labels_norm, _, _ = self.normalize_labels(labels)
        return data, labels, labels_norm

    def get_vocabularies(self, dir_query_plan):
        """
            1. 从所有的查询计划中获取 原始特征 (Seq Scan 涉及的表, 别名 以及过滤条件)
            2. 获取句子中的所有非数字词汇.
            3. 做编码, 类似于以下过程, 然后每个词汇都有自己的 one-hot 编码, 然后再保存成字典的形式, 就可以返回了
                b = le.fit_transform(["1", "2", "3", "4", "5", "6", "5", "10", "22"])
                1 0 [1. 0. 0. 0. 0. 0. 0. 0.]
                2 2 [0. 0. 1. 0. 0. 0. 0. 0.]
                3 4 [0. 0. 0. 0. 1. 0. 0. 0.]
                4 5 [0. 0. 0. 0. 0. 1. 0. 0.]
                5 6 [0. 0. 0. 0. 0. 0. 1. 0.]
                6 7 [0. 0. 0. 0. 0. 0. 0. 1.]
                5 6 [0. 0. 0. 0. 0. 0. 1. 0.]
                10 1 [0. 1. 0. 0. 0. 0. 0. 0.]
                22 3 [0. 0. 0. 1. 0. 0. 0. 0.]
            Out[32]:
                {'1': array([[1., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32),
                 '2': array([[0., 0., 1., 0., 0., 0., 0., 0.]], dtype=float32),
                 '3': array([[0., 0., 0., 0., 1., 0., 0., 0.]], dtype=float32),
                 '4': array([[0., 0., 0., 0., 0., 1., 0., 0.]], dtype=float32),
                 '5': array([[0., 0., 0., 0., 0., 0., 1., 0.]], dtype=float32),
                 '6': array([[0., 0., 0., 0., 0., 0., 0., 1.]], dtype=float32),
                 '10': array([[0., 1., 0., 0., 0., 0., 0., 0.]], dtype=float32),
                 '22': array([[0., 0., 0., 1., 0., 0., 0., 0.]], dtype=float32)}
        Args:
            dir_query_plan:

        Returns:

        """
        # # 获取 数据 以及这个Seq Scan的涉及的行数
        sentences, rows = self.get_data_and_label(dir_query_plan)
        # # 获取句子中无重复的词汇, 但是要去掉数字
        vocabulary = list(set(
            filter(
                self.is_not_number,
                set(functools.reduce(lambda x, y: x + y, sentences, list()))
            )
        ))

        # # 对获取的词汇做编码, 就是给一个编号
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(np.array(vocabulary))
        # # 做 one-hot 编码 形成一个矩阵
        encoded = utils.to_categorical(integer_encoded)
        # # 保存成字典的形式
        vocab_dict = dict()
        for v, e in zip(vocabulary, encoded):
            # # 一维数组升维为二维的
            vocab_dict[v] = np.reshape(np.array(e), (1, len(vocabulary)))
        return sentences, rows, vocab_dict

    @staticmethod
    def is_not_number(s):
        """
            输入应该是字符串类型, 判断是否为数字
            这个方法写的不是很优雅, 可以先用. split, 然后判断分成的个数
            在判断每个元素是否都是数字即可
        Args:
            s:

        Returns:

        """
        try:
            float(s)
            return False
        except ValueError:
            pass
        try:
            unicodedata.numeric(s)
            return False
        except (TypeError, ValueError):
            pass
        return True

    @staticmethod
    def extract_time(line):
        """
            获取本次表空间算法所用的时间, 包括估计的时间和实际使用的时间
            能用 不改了

            当查询计划中出现 Sequence Scan 的时候 使用这个方法, 这个必须是在叶子节点出现的
            这个用re应该会更好
            "Seq Scan.*?cost=(.*?)..(.*?) rows=(.*?) width=(.*?)) (actual time=(.*?)..(.*?) rows=(.*?) loops=(.*?)"
            Seq Scan on title t  (cost=0.00..61281.03 rows=2528303 width=4) (actual time=0.018..681.646 rows=2528312 loops=1)
            匹配出所有数字[\d]+, 然后对 res[0]和res[3]分割
        Args:
            line: Seq Scan那一行

        Returns:
            start_cost Estimated start-up cost. This is the time expended before the output phase can begin,
                    ...e.g., time to do the sorting in a sort node.
            end_cost Estimated total cost. This is stated on the assumption that the plan node is run to completion,
                    ...i.e., all available rows are retrieved. In practice a node's parent node might stop short of reading
                    ...all available rows (see the LIMIT example below).
            rows    Estimated number of rows output by this plan node. Again, the node is assumed to be run to completion.
            width   Estimated average width of rows output by this plan node (in bytes).
            a_start_cost 实际的 启动时间
            a_end_cost  实际的 运行时间
            a_rows 实际影响的行数
            loops(没有返回, 对于Seq Loop 一般就是一趟)
        """
        data = line.replace("->", "").lstrip().split("  ")[-1].split(" ")
        start_cost = data[0].split("..")[0].replace("(cost=", "")
        end_cost = data[0].split("..")[1]
        rows = data[1].replace("rows=", "")
        width = data[2].replace("width=", "").replace(")", "")
        a_start_cost = data[4].split("..")[0].replace("time=", "")
        a_end_cost = data[4].split("..")[1]
        a_rows = data[5].replace("rows=", "")
        return float(start_cost), float(end_cost), float(rows), float(width), float(a_start_cost), float(
            a_end_cost), float(
            a_rows)

    @staticmethod
    def normalize_data(val, column_name, column_min_max_vals):
        """
            按照论文中的说法, 这里好像是对每一列做最大最小正则化
            调用前提：is_not_number()方法返回了False, 也就是说, 这是一个浮点数
        Args:
            val: 值
            column_name: 列名称
            column_min_max_vals: 列的最大最小值, 数据来自 data/column_min_max_vals.csv

        Returns:

        """
        if column_name not in column_min_max_vals:
            EmbeddingLogger.warning(f"{column_name} not in dict")
        # # 默认值为 [0, 10]
        min_val, max_val = column_min_max_vals.get(column_name, (0, 10))
        val = float(val)
        if val > max_val:
            val = max_val
        elif val < min_val:
            val = min_val
        val = float(val)
        val_norm = (val - min_val) / (max_val - min_val)
        return val_norm

    @staticmethod
    def normalize_labels(labels):
        """
            log transformation without normalize
        Args:
            labels:

        Returns:

        """
        labels = np.array([
            # # 直接一个map多好, 直接用 np.logs多好
            np.log(float(_label)) for _label in labels
        ]).astype(np.float32)
        return labels, 0, 1

    @staticmethod
    def read_column_min_max_vals(_fpath_column_min_max_vals):
        """
            读取 CSV 文件
        Args:
            _fpath_column_min_max_vals:

        Returns:

        """
        _column_min_max_vals = dict()
        with open(_fpath_column_min_max_vals, 'r') as _csv_f:
            _reader = csv.reader(_csv_f, delimiter=',')
            next(_reader)
            for _idx, _row in enumerate(_reader):
                _column_min_max_vals[_row[0]] = [float(_row[1]), float(_row[2])]
        return _column_min_max_vals


class Node(object):
    """
        简单的多叉树的实现
    """

    def __init__(self, data, parent=None):
        self.data = data
        self.children = []
        self.parent = parent

    def add_child(self, obj: 'Node'):
        self.children.append(obj)
        # # 需要设置子节点的parent为自己
        obj.parent = self

    def set_parent(self, obj):
        self.parent = obj

    def __str__(self, tabs=0):
        tab_spaces = str.join("", [" " for _ in range(tabs)])
        return tab_spaces + "+-- Node: " + str.join("|", self.data) + "\n" + str.join("\n",
                                                                                      [child.__str__(tabs + 2) for child
                                                                                       in self.children])


class PlanSequential:
    # # 没有涉及到的不处理了
    operators = ['Merge Join', 'Hash', 'Index Only Scan using title_pkey on title t', 'Sort', 'Seq Scan',
                 'Index Scan using title_pkey on title t', 'Materialize', 'Nested Loop', 'Hash Join',
                 # # new
                 'Gather'
                 ]
    columns = ['ci.movie_id', 't.id', 'mi_idx.movie_id', 'mi.movie_id', 'mc.movie_id', 'mk.movie_id']

    def __init__(self, _embedding_res_path, _folder_name):
        """

        Args:
            _embedding_res_path: 嵌入完成后的输出, 就是 某某 npy 很乱
        """
        self.scan_features = np.load(_embedding_res_path)
        self.dir_query_plan = _folder_name

    def flow(self, fp_job_cardinality_sequence, fp_cost_labels):
        """
            parse_dep_tree_text
        Args:
            fp_job_cardinality_sequence: 保存编码之后的树,
            fp_cost_labels: 保存cost标签

        Returns:

        """
        # # 1. 生成查询计划树
        trees, max_children = self.parse_dep_tree_text(self.dir_query_plan)

        # # 后序遍历, 生成所有树的data的列表
        all_trees = list(map(self.plan2seq, trees))

        # # 把所有查询计划的查询树保存一下
        with open(fp_job_cardinality_sequence, "wb") as f:
            pickle.dump(all_trees, f)

        # #  生成 cost_label, a_end_cost 这参数保存一下, 父节点的Cost
        cost_label = list(map(lambda tree: tree.data[-2], trees))

        np.save(fp_cost_labels, np.array(cost_label))

    def extract_operator(self, line: str):
        """
            输入为查询计划的某一行, 为啥只处理Seq Scan,
            后面那个 Parallel Seq Scan 是 本人加的
            输出模式, 以下几种开头 之后带任意多个字符 表示在哪个索引 哪个表:
                ["Bitmap Heap Scan",
                "Bitmap Index Scan",
                "Gather",
                "Hash",
                "Hash Join",
                "Index Only Scan",
                "Index Scan",
                "Memoize",
                "Nested Loop",
                "Partial Aggregate",
                "Seq Scan"]
        Args:
            line: 查询计划的某一行

        Returns:

        """
        # # 去掉箭头和前后的空白字符, 取第一个词, 这个是有很多的, 比如 all_operators.json中词语
        operator = line.replace("->", "").lstrip().split("  ")[0]
        # #
        if operator.startswith("Seq Scan") or operator.startswith("Parallel Seq Scan"):
            operator = "Seq Scan"
        if operator == "Parallel Hash":
            operator = "Hash"
        return operator, operator in self.operators

    def extract_attributes(self, operator, line, feature_vec, i=None):
        """
            这里应该要解析操作的参数了
            todo 对operator的修改都要在这里有所体现
        Args:
            operator:
            line:
            feature_vec:
            i:

        Returns:

        """
        operators_count = len(self.operators)  # 9
        if operator in ["Hash", "Materialize", "Nested Loop", "Gather"]:
            pass
        elif operator == "Merge Join":
            # # 看看那个列在, 就把这个列标成1, 下同
            if "Cond" in line:
                for column in self.columns:
                    if column in line:
                        feature_vec[self.columns.index(column) + operators_count] = 1.0
        elif operator == "Index Only Scan using title_pkey on title t":
            if "Cond" in line:
                # # 特定的将t.id标成1
                feature_vec[self.columns.index("t.id") + operators_count] = 1.0
                for column in self.columns:
                    if column in line:
                        feature_vec[self.columns.index(column) + operators_count] = 1.0
        elif operator == "Sort":
            for column in self.columns:
                if column in line:
                    feature_vec[self.columns.index(column) + operators_count] = 1.0
        elif operator == 'Index Scan using title_pkey on title t':
            # # 特定的将t.id标成1
            if "Cond" in line:
                feature_vec[self.columns.index("t.id") + operators_count] = 1.0
                for column in self.columns:
                    if column in line:
                        feature_vec[self.columns.index(column) + operators_count] = 1.0
        elif operator == 'Hash Join':
            if "Cond" in line:
                for column in self.columns:
                    if column in line:
                        feature_vec[self.columns.index(column) + operators_count] = 1.0
        elif operator == 'Seq Scan' or operator == "Parallel Seq Scan":
            # # 叶子节点嵌入
            feature_vec[15:79] = self.scan_features[i]
        else:
            pass

    def parse_dep_tree_text(self, dir_query_path):
        """

        Args:
            dir_query_path: 查询计划路径

        Returns:

        """
        scan_cnt = 0  # # Scan的数量
        max_children = 0  # # 节点最大子节点树
        plan_trees = []  # # 每个查询计划有且只有一个节点,其实就是根节点
        # # feature_len怎么给出的, 但不重要, 最后反正是 给出 feature_vec 就可以啦
        # # 9是操作个数(self.operators), 6 是列数量(self.columns), 这15个特征是 one-hot编码
        # # 涉及到哪个操作或者哪个列 就把 哪个操作标为1
        # # 64是scan embedding 的数据
        # # 7是cost估计和cost实际值(cost=4285.08..343940.07 rows=1 width=32) (actual time=51.653..267.320 rows=14...)
        feature_len = len(self.operators) + len(self.columns) + 64 + 7
        for each_plan in sorted(os.listdir(dir_query_path)):
            # # 读取查询计划
            print(each_plan)
            feature_vec = [0.0] * feature_len
            with open(os.path.join(dir_query_path, each_plan), 'r') as f:
                lines = f.readlines()
                # # 获取第一行的operator
                operator, in_operators = self.extract_operator(lines[0])

                if not in_operators:
                    # # todo 这里并不能断言 operator 一定在
                    # # 不在 self.operators, 就解析第一行的, 第一行就一定在吗?
                    # # 所以这里简单处理一下, 如果不在, 那就continue
                    operator, in_operators = self.extract_operator(lines[1])
                    assert in_operators, f"{operator} 不在operators中"
                    # # 获取该操作的运行估计,
                    # (cost=4285.08..343940.07 rows=1 width=32) (actual time=51.653..267.320 rows=14 loops=3)
                    start_cost, end_cost, rows, width, a_start_cost, a_end_cost, a_rows = self.extract_time(
                        lines[1])
                    # # j应该是index, 就是下一行从哪里开始
                    j = 2
                else:
                    start_cost, end_cost, rows, width, a_start_cost, a_end_cost, a_rows = self.extract_time(
                        lines[0])
                    j = 1
                # # 最后七个特征直接写入
                feature_vec[- 7:] = [start_cost, end_cost, rows, width, a_start_cost, a_end_cost, a_rows]
                # # todo 前面是one-hot编码
                if in_operators:
                    feature_vec[self.operators.index(operator)] = 1.0
                # # 看看是否是叶子节点
                if operator == "Seq Scan" or operator == "Parallel Seq Scan":
                    # # 是叶子节点... 这时, 树中只有一个节点
                    # #
                    self.extract_attributes(operator, lines[j], feature_vec, scan_cnt)
                    scan_cnt += 1

                    # # ... 将节点添加到plans_trees中, 做下一个查询计划
                    root_tokens = feature_vec
                    current_node = Node(root_tokens)
                    plan_trees.append(current_node)
                    # # # ...
                    # continue
                else:
                    # # 不是叶子节点, 下面还有, 这些是这个operator的附属信息
                    # # 这两行等价, 遇到 -> 才是新的操作,
                    # while "->" not in lines[j]:
                    while "actual" not in lines[j] and "Plan" not in lines[j]:
                        self.extract_attributes(operator, lines[j], feature_vec)
                        j += 1
                    # # 将该节点添加到tree中
                    root_tokens = feature_vec
                    current_node = Node(root_tokens)
                    plan_trees.append(current_node)

                    spaces = 0  # # 用以标识当前节点深度
                    node_stack = []  # # 结点栈
                    i = j
                    # # not lines[i].startswith("Planning time") 这是查询计划的终点
                    # # i是根节点之后的第一个节点
                    while not lines[i].startswith("Planning Time"):
                        # # 对于每个节点, 事实上的处理和根节点一致
                        line = lines[i]
                        i += 1
                        # # 迭代出口
                        if line.startswith("Planning time") or line.startswith("Execution time"):
                            break
                        elif line.strip() == "":
                            break
                        elif "->" not in line:
                            # # 完全没用
                            continue
                        else:
                            if line.index("->") < spaces:
                                # # 当前行的深度小于上一行, 退栈, 直到相同
                                while line.index("->") < spaces:
                                    current_node, spaces = node_stack.pop()
                            elif line.index("->") > spaces:
                                # # 当前行是上一行的子节点
                                # # current_node实际上是不能直接用的
                                # # 字符串... line_copy???
                                line_copy = line
                                # # 用的和之前是一样的标识符 feature_vec, 怪不得之前要做一下引用
                                # # 这个过程和之前就一样了
                                feature_vec = [0.0] * feature_len
                                start_cost, end_cost, rows, width, a_start_cost, a_end_cost, a_rows = self.extract_time(
                                    line_copy)
                                feature_vec[-7:] = [start_cost,
                                                    end_cost, rows, width, a_start_cost,
                                                    a_end_cost,
                                                    a_rows]

                                operator, in_operators = self.extract_operator(line_copy)
                                if in_operators:
                                    feature_vec[self.operators.index(operator)] = 1.0
                                if operator == "Seq Scan" or operator == "Parallel Seq Scan":
                                    self.extract_attributes(operator, line_copy, feature_vec, scan_cnt)
                                    scan_cnt += 1
                                else:
                                    j = 0
                                    while "actual" not in lines[i + j] and "Plan" not in lines[i + j]:
                                        self.extract_attributes(
                                            operator, lines[i + j], feature_vec)
                                        j += 1
                                tokens = feature_vec
                                new_node = Node(tokens, parent=current_node)
                                current_node.add_child(new_node)
                                # # ... ======= 以上过程, 分析操作和操作参数, 编码成树节点 插入
                                # # ... 这是对栈的修改
                                if len(current_node.children) > max_children:
                                    max_children = len(current_node.children)
                                # # 对栈的修改
                                node_stack.append((current_node, spaces))
                                # # ....
                                current_node = new_node
                                spaces = line.index("->")
                            else:
                                # # 同级节点
                                line_copy = line
                                feature_vec = [0.0] * feature_len
                                start_cost, end_cost, rows, width, a_start_cost, a_end_cost, a_rows = self.extract_time(
                                    line_copy)
                                feature_vec[- 7:] = [start_cost,
                                                     end_cost, rows, width, a_start_cost,
                                                     a_end_cost,
                                                     a_rows]
                                operator, in_operators = self.extract_operator(line_copy)
                                if in_operators:
                                    feature_vec[self.operators.index(operator)] = 1.0
                                if operator == "Seq Scan" or operator == "Parallel Seq Scan":
                                    self.extract_attributes(
                                        operator, line_copy, feature_vec, scan_cnt)
                                    scan_cnt += 1
                                else:
                                    j = 0
                                    while "actual" not in lines[i + j] and "Plan" not in lines[i + j]:
                                        self.extract_attributes(
                                            operator, lines[i + j], feature_vec)
                                        j += 1
                                tokens = feature_vec
                                new_node = Node(tokens, parent=node_stack[-1][0])
                                node_stack[-1][0].add_child(new_node)
                                # # .... ======= 以上过程, 分析操作和操作参数, 编码成树节点 插入
                                # # 对栈的修改
                                if len(node_stack[-1][0].children) > max_children:
                                    max_children = len(node_stack[-1][0].children)
                                current_node = new_node
                                spaces = line.index("->")
        return plan_trees, max_children

    @staticmethod
    def extract_time(line):
        """
            和上面的完全一致
        Args:
            line:

        Returns:

        """
        try:
            data = line.replace("->", "").lstrip().split("  ")[-1].split(" ")
            start_cost = data[0].split("..")[0].replace("(cost=", "")
            end_cost = data[0].split("..")[1]
            rows = data[1].replace("rows=", "")
            width = data[2].replace("width=", "").replace(")", "")
            if "never executed" not in line:
                a_start_cost = data[4].split("..")[0].replace("time=", "")
                a_end_cost = data[4].split("..")[1]
                a_rows = data[5].replace("rows=", "")
            else:
                a_start_cost = start_cost
                a_end_cost = end_cost
                a_rows = rows
            return float(start_cost), float(end_cost), float(rows), float(width), float(a_start_cost), float(
                a_end_cost), float(
                a_rows)
        except Exception as e:
            EmbeddingLogger.exception(f"{e}, data: {line.strip()}")

    @staticmethod
    def p2t(node: Node):
        tree = {}
        tmp = node.data
        operators_count = 9
        columns_count = 6
        scan_features = 64
        assert len(tmp) == operators_count + columns_count + 7 + scan_features
        tree['features'] = tmp[:operators_count + columns_count + scan_features]
        if node.data[-1] != 0:
            tree['labels'] = np.log(node.data[-1])
        else:
            tree['labels'] = np.log(1)
        tree['pg'] = np.log(node.data[-5])
        tree['children'] = []
        for children in node.children:
            tree['children'].append(PlanSequential.p2t(children))
        return tree

    @staticmethod
    def plan2seq(node: Node):
        """
            其实就是执行了一个后序遍历
        Args:
            node:

        Returns:

        """
        sequence = []
        # # data就是 9 + 6 + 64 + 7 维度数组
        tmp = node.data
        operators_count = 9
        columns_count = 6
        scan_features = 64
        cost_info_count = 7
        # # 判断是否为叶子节点
        if len(node.children) != 0:
            # # 不是叶子节点
            for i in range(len(node.children)):
                # #
                sequence.extend(PlanSequential.plan2seq(node.children[i]))
        # # 不要最后的cost_info
        # sequence.append(tmp[:-cost_info_count])
        sequence.append(tmp[:operators_count + columns_count + scan_features])
        return sequence

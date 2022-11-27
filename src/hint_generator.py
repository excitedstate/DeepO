import copy
import logging
import pickle

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv
import os
from sklearn import preprocessing
import joblib
import numpy as np
import torch
from tqdm import tqdm

from src.basic import PostgresDB, GeneralLogger
from src.config import DATA_PATH_GOT_PLANS, DATA_PATH_GENE_SQL_WITH_HINTS, DATA_PATH_GENE_SQL, DB_LAB_VM_CONFIG
from src.cost_learner import CostEstimationTrainer
from src.embedding import Node, PlanSequential

HintGeneratorLogger = GeneralLogger(name="hint_generator", stdout_flag=True, stdout_level=logging.INFO,
                                    file_mode="w")


class HintGenerator:
    def __init__(self, _sql_path):
        self.tables, self.joins, self.predicates = self.load_data(_sql_path)
        self.db = PostgresDB(_config=DB_LAB_VM_CONFIG)

    def flow(self):
        # # 创建文件夹
        os.makedirs(os.path.join(DATA_PATH_GENE_SQL), mode=0o777, exist_ok=True)
        os.makedirs(os.path.join(DATA_PATH_GENE_SQL_WITH_HINTS), mode=0o777, exist_ok=True)
        os.makedirs(os.path.join(DATA_PATH_GOT_PLANS), mode=0o777, exist_ok=True)

        for query_idx in tqdm(range(0, 20)):
            # # 创建本查询专属的文件夹
            os.makedirs(os.path.join(DATA_PATH_GOT_PLANS, f"{query_idx}"), mode=0o777, exist_ok=True)

            # # 获取带hint的查询
            queries_with_hint, sql = self.generate_hint_queries(query_idx, method="explain")

            # # 写入SQL
            with open(os.path.join(DATA_PATH_GENE_SQL, f"{query_idx}"), "w") as f:
                f.write(sql)

            with open(os.path.join(DATA_PATH_GENE_SQL_WITH_HINTS, f"{query_idx}"), "w") as f:
                f.write("\n".join(queries_with_hint))

            for idx, query_with_hint in enumerate(tqdm(queries_with_hint)):
                # # query_with_hint 是带有 explain 的
                plan = self.db.execute(query_with_hint)

                # # 写入日志
                with open(os.path.join(DATA_PATH_GOT_PLANS, f"{query_idx}", f"{idx}"), "w") as f:
                    # # 拼成一个元组即可
                    if not isinstance(plan, tuple):
                        HintGeneratorLogger.exception(f"plan: {plan} is not tuple")
                    else:
                        f.write('\n'.join(tuple(zip(*plan))[0]))

    def generate_join_method_hints_from_orders(self, join_order_hints, join_orders_list):
        join_methods = ["NestLoop({})", "MergeJoin({})", "HashJoin({})"]
        join_hints = []
        for order_hint, order in zip(join_order_hints, join_orders_list):
            parsed_order = self.parse_order(order)
            join_order = []
            for idx in range(2, len(parsed_order) + 1):
                join_order.append(" ".join(parsed_order[0:idx]))
            join_candidate = []
            for idx, level in enumerate(join_order):
                join_candidate.append([each.format(level) for each in join_methods])
            candidates = [" ".join(x) for x in self.cartesian(join_candidate, 'object')]
            join_hints.extend([each + " " + order_hint for each in candidates])
        if not join_hints:
            join_hints = [""]
        return join_hints

    def generate_scan_hints(self, tables):
        scan_methods = ["SeqScan({})", "IndexScan({})"]
        hint_candidate = []
        for table in tables:
            table_candidate = []
            t = table.split(" ")[1]
            for method in scan_methods:
                table_candidate.append(method.format(t))
            hint_candidate.append(table_candidate)
        candidates = [" ".join(x) for x in self.cartesian(hint_candidate, 'object')]
        return candidates

    def generate_join_order_hints(self, tables):
        if len(tables) == 1:
            return [""], []
        join_tables = [x.split(" ")[1] for x in tables]
        num_tables = len(tables)
        str_order_length = 3 * num_tables - 2
        join_orders = []
        starter = copy.deepcopy(join_tables)
        stack = [[each] for each in starter]
        while len(stack) != 0:
            cur = stack.pop(0)
            if len(cur) < str_order_length:
                extended_orders = self.add_one_rel(cur, join_tables)
                stack.extend(extended_orders)
            else:
                join_orders.append(cur)
        str_join_orders = [" ".join(each) for each in join_orders]
        # print(str_join_orders)
        str_join_orders = set(str_join_orders)
        join_orders_string = ["Leading ({})".format(each) for each in str_join_orders]
        # print(join_orders)
        return join_orders_string, join_orders

    def generate_hint_queries(self, query_idx, method):
        scan_hints = self.generate_scan_hints(self.tables[query_idx])
        join_order_hints, join_orders = self.generate_join_order_hints(self.tables[query_idx])
        join_hints = self.generate_join_method_hints_from_orders(join_order_hints, join_orders)
        candidates = [scan_hints, join_hints]
        hints_set = [" ".join(x) for x in self.cartesian(candidates, 'object')]
        sql = self.construct_sql(self.tables[query_idx], self.joins[query_idx], self.predicates[query_idx], method)
        queries = []
        for each in hints_set:
            query = "/*+ {} */ ".format(each) + sql + ";"
            queries.append(query)
        return queries, sql + ";"

    @staticmethod
    def cartesian(arrays, dtype=None, out=None):
        """笛卡尔积"""
        arrays = [np.asarray(x) for x in arrays]
        if dtype is None:
            dtype = arrays[0].dtype
        n = np.prod([x.size for x in arrays])
        if out is None:
            out = np.zeros([n, len(arrays)], dtype=dtype)

        m = int(n / arrays[0].size)
        out[:, 0] = np.repeat(arrays[0], m)
        if arrays[1:]:
            HintGenerator.cartesian(arrays[1:], out=out[0:m, 1:])
            for j in range(1, arrays[0].size):
                out[j * m:(j + 1) * m, 1:] = out[0:m, 1:]
        return out

    @staticmethod
    def construct_sql(table, join, predicates, method="explain"):
        tables = ", ".join(table)
        HintGeneratorLogger.info(join)
        HintGeneratorLogger.info(predicates)
        if join != [""] and predicates != [""]:
            joins = " and ".join(join)
            sql = method + " select count(*) from {} where {} and {}"
        elif join != [""] and predicates == [""]:
            joins = " and ".join(join)
            sql = method + " select count(*) from {} where {} {}"
        elif join == [""] and predicates != [""]:
            joins = ""
            sql = method + " select count(*) from {} where {} {}"
        else:
            joins = ""
            sql = method + " select count(*) from {} {} {}"
        temp_list = list()
        for n in range(len(predicates) // 3):
            temp_list.append(' '.join(predicates[n * 3:n * 3 + 3]))
        predicates = " and ".join(temp_list)
        return sql.format(tables, joins, predicates)

    @staticmethod
    def parse_order(order):
        left = 0
        right = len(order) - 1
        parsed_order = []
        while left < right:
            if order[left] == "(" and order[right] == ")":
                left += 1
                right -= 1
            elif order[left] == "(":
                parsed_order.insert(0, order[right])
                right -= 1
            elif order[right] == ")":
                parsed_order.insert(0, order[left])
                left += 1
            else:
                parsed_order.insert(0, order[right])
                parsed_order.insert(0, order[left])
                left += 1
                right -= 1
        return parsed_order

    @staticmethod
    def add_one_rel(cur, join_tables):
        extended_order = []
        for table in join_tables:
            if table not in cur:
                tmp = ["("]
                tmp.extend(cur)
                tmp.append(table)
                tmp.append(")")
                extended_order.append(tmp)

                tmp = ["(", table]
                tmp.extend(cur)
                tmp.append(")")

                extended_order.append(tmp)
            else:
                continue
        return extended_order

    @staticmethod
    def load_data(file_name):
        joins = []
        predicates = []
        tables = []

        # Load queries
        with open(file_name, 'r') as f:
            data_raw = list(list(rec) for rec in csv.reader(f, delimiter='#'))
            for row in data_raw:
                tables.append(row[0].split(','))
                joins.append(row[1].split(','))
                predicates.append(row[2].split(','))
                # label.append(row[3])
        HintGeneratorLogger.info("Loaded queries")

        return tables, joins, predicates


class CostEstimator:
    operators = PlanSequential.operators
    columns = PlanSequential.columns

    # vocabulary = ['movie_info_idx', 'Filter', 'info_type_id', '>', 'title', 'kind_id', '=', 'production_year',
    #               'movie_keyword', 'keyword_id', 'cast_info', 'person_id', 'AND', 'role_id', 'mk', 't', '<',
    #               'movie_info', 'mi', 'movie_companies', 'mc', 'ci', 'company_id', 'company_type_id', 'mi_idx'
    #               ]
    vocabulary = []

    def __init__(self, _fp_std_scaler, cost_model_path):
        self.sc = joblib.load(_fp_std_scaler)
        self.net = torch.load(cost_model_path)

    def flow(self, plan_dir, leaf_model_path, fp_column_statistics, fp_vocab_dict=None):
        max_length = 8

        line = []
        candidate_plans_num = len(os.listdir(plan_dir))
        for idx in range(candidate_plans_num):
            HintGeneratorLogger.warning(f"{idx} / {candidate_plans_num}, 1")
            cur_plan_path = os.path.join(plan_dir, str(idx))
            # # 叶子节点嵌入
            leaf_embedding = self.leaf_embedded(cur_plan_path, model_path=leaf_model_path,
                                                fp_column_statistics=fp_column_statistics,
                                                fp_vocab_dict=fp_vocab_dict)
            if leaf_embedding is None:
                continue
            # # 树嵌入
            test_tree = self.tree_embedding(leaf_embedding, cur_plan_path)
            HintGeneratorLogger.warning(f"{idx} / {candidate_plans_num}, 2")
            if len(test_tree) < max_length:
                tmp = [[0] * 79] * (max_length - len(test_tree))
                tmp.extend(test_tree)
                padded_sequences = tmp
            else:
                padded_sequences = test_tree
            padded_sequences = np.array(padded_sequences)
            padded_sequences = torch.tensor(padded_sequences, dtype=torch.float32)
            HintGeneratorLogger.warning(f"{idx} / {candidate_plans_num}, 3")
            pred, pred_vals = self.pred_cost(padded_sequences.unsqueeze(0), 100, self.sc, self.net)

            upper, lower = CostEstimationTrainer.get_intervals(pred_vals, 1)
            HintGeneratorLogger.warning(f"{idx} / {candidate_plans_num}, 4")
            # print("label: ", unscaled_y)
            HintGeneratorLogger.info(f"prediction: {pred}")
            HintGeneratorLogger.info(f"prediction upper bound: {upper}")
            HintGeneratorLogger.info(f"prediction lower bound: {lower}")
            HintGeneratorLogger.info("*" * 30)
            # print("label in prediction range: ",in_range)
            line.append("{},{},{}".format(pred, upper, lower))
        with open("./example/pred_result.txt", 'w') as f:
            f.writelines("\n".join(line))
            # break

    def parse_tree(self, operators, columns, leaf_embedding, plan_path):
        scan_cnt = 0
        max_children = 0
        plan_trees = []
        feature_len = 9 + 6 + 4 + 64
        with open(plan_path, 'r') as f:
            lines = f.readlines()
        feature_vec = [0.0] * feature_len
        operator, in_operators = self.extract_operator(lines[0], operators)
        # print("operator: ",operator)
        if not in_operators:
            operator, in_operators = self.extract_operator(lines[1], operators)
            start_cost, end_cost, rows, width = self.extract_plan(lines[1])
            j = 2
        else:
            start_cost, end_cost, rows, width = self.extract_plan(lines[0])
            j = 1
        feature_vec[feature_len - 7:feature_len] = [start_cost, end_cost, rows, width]
        if in_operators:
            feature_vec[operators.index(operator)] = 1.0
        if operator == "Seq Scan":
            self.extract_attributes(operator, operators, columns, lines[j], feature_vec, leaf_embedding, scan_cnt)
            scan_cnt += 1
            # root_tokens = feature_vec
            # current_node = Node(root_tokens)
            # plan_trees.append(current_node)
        else:
            while j < len(lines) and "->" not in lines[j]:
                self.extract_attributes(operator, operators, columns, lines[j], feature_vec, leaf_embedding)
                j += 1
        root_tokens = feature_vec  # 所有吗
        current_node = Node(root_tokens)
        plan_trees.append(current_node)
        spaces = 0
        node_stack = []
        i = j
        while not i >= len(lines):
            line = lines[i]
            i += 1
            if line.startswith("Planning Time") or line.startswith("Execution Time"):
                break
            elif line.strip() == "":
                break
            elif "->" not in line:
                continue
            else:
                if line.index("->") < spaces:
                    while line.index("->") < spaces:
                        current_node, spaces = node_stack.pop()
                if line.index("->") > spaces:
                    line_copy = line
                    feature_vec = [0.0] * feature_len
                    start_cost, end_cost, rows, width = self.extract_plan(
                        line_copy)
                    feature_vec[feature_len - 7:feature_len] = [start_cost, end_cost, rows, width]
                    operator, in_operators = self.extract_operator(line_copy, operators)
                    if in_operators:
                        feature_vec[operators.index(operator)] = 1.0
                    if operator == "Seq Scan":
                        self.extract_attributes(
                            operator, operators, columns, line_copy, feature_vec, leaf_embedding, scan_cnt)
                        scan_cnt += 1
                    else:
                        j = 0
                        while (i + j) < len(lines) and "->" not in lines[i + j]:
                            self.extract_attributes(
                                operator, operators, columns, lines[i + j], feature_vec, leaf_embedding)
                            j += 1
                    tokens = feature_vec
                    # print("token",tokens)
                    new_node = Node(tokens, parent=current_node)
                    current_node.add_child(new_node)
                    if len(current_node.children) > max_children:
                        max_children = len(current_node.children)
                    node_stack.append((current_node, spaces))
                    current_node = new_node
                    spaces = line.index("->")
                elif line.index("->") == spaces:
                    line_copy = line
                    feature_vec = [0.0] * feature_len
                    start_cost, end_cost, rows, width = self.extract_plan(
                        line_copy)
                    feature_vec[feature_len - 7:feature_len] = [start_cost, end_cost, rows, width]
                    operator, in_operators = self.extract_operator(line_copy, operators)
                    if in_operators:
                        feature_vec[operators.index(operator)] = 1.0
                    if operator == "Seq Scan":
                        self.extract_attributes(
                            operator, operators, columns, line_copy, feature_vec, leaf_embedding, scan_cnt)
                        scan_cnt += 1
                    else:
                        j = 0
                        while (i + j) < len(lines) and "->" not in lines[i + j]:
                            self.extract_attributes(
                                operator, operators, columns, lines[i + j], feature_vec, leaf_embedding)
                            j += 1
                    tokens = feature_vec
                    new_node = Node(tokens, parent=node_stack[-1][0])
                    node_stack[-1][0].add_child(new_node)
                    if len(node_stack[-1][0].children) > max_children:
                        max_children = len(node_stack[-1][0].children)
                    current_node = new_node
                    spaces = line.index("->")
        # print("scan count: ",scan_cnt)
        return plan_trees, max_children  # a list of the roots nodes

    def tree_embedding(self, leaf_embedding, plan):

        root_node, max_children = self.parse_tree(self.operators, self.columns, leaf_embedding, plan)
        # print(root_node)
        embedded_tree = self.plan2seq(root_node[0])
        return embedded_tree

    def plan2seq(self, node: Node):
        sequence = []
        tmp = node.data

        operators_count = 9
        columns_count = 6
        scan_features = 64
        if len(node.children) != 0:
            for i in range(len(node.children)):
                sequence.extend(self.plan2seq(node.children[i]))
        sequence.append(tmp[:operators_count + columns_count + scan_features])
        return sequence

    def leaf_embedded(self, plan, model_path, fp_column_statistics, fp_vocab_dict=None):
        """embedding leaf node in plan into vector

        Args:
            fp_vocab_dict:
            fp_column_statistics:
            plan ([path]): path of a plan file
            model_path (str, optional): [description]. Defaults to "/home/sunluming/deepO/Mine_total/final/embedding_model.h5".
        """
        # base statistics and vocabulary dict for leaf embedding
        column_min_max_vals = self.get_column_statistics(fp_column_statistics)
        vocab_dict = self.get_vocabulary_encoding(fp_vocab_dict)
        # extract features from plan
        test_sentences, test_rows, test_pg = self.get_data_and_label(column_min_max_vals, plan)
        test_data, test_label = self.prepare_data_and_label(test_sentences, test_rows, vocab_dict, len(vocab_dict))
        padded_sentences = self.padding_sentence(test_data, 20)
        # load model
        model = load_model(model_path)
        # model.summary()
        intermediate_layer_model = Model(inputs=model.input,
                                         outputs=model.layers[4].output)
        if len(padded_sentences) > 0:
            intermediate_output = intermediate_layer_model.predict(padded_sentences)
        else:
            intermediate_output = None
        return intermediate_output

    def get_data_and_label(self, column_min_max_vals, plan_path):
        sentences = []
        rows = []
        pg = []
        with open(plan_path, 'r') as f:
            plan = f.readlines()
        for i in range(len(plan)):
            if "Seq Scan" in plan[i]:
                _start_cost, _end_cost, _rows, _width = self.extract_plan(plan[i])
                if len(plan[i].strip().split("  ")) == 2:
                    _sentence = " ".join(plan[i].strip().split("  ")[0].split(" ")[:-1]) + " "
                    table = plan[i].strip().split("  ")[0].split(" ")[4]
                else:
                    _sentence = " ".join(plan[i].strip().split("  ")[1].split(" ")[:-1]) + " "
                    table = plan[i].strip().split("  ")[1].split(" ")[4]
                if (i + 1 < len(plan) and "actual" not in plan[i + 1] and "Plan" not in plan[i + 1] and "->" not in
                        plan[
                            i + 1]):
                    _sentence += plan[i + 1].strip()
                else:
                    _sentence += table
                    _sentence = _sentence + ' ' + _sentence
                _sentence = _sentence.replace(": ", " ").replace("(", "").replace(")", "").replace("'", "").replace(
                    "::bpchar", "") \
                    .replace("[]", "").replace(",", " ").replace("\\", "").replace("::numeric", "").replace("  ", " ") \
                    .replace("Seq Scan on ", "").strip()
                sentence = []
                ll = _sentence.split(" ")
                for cnt in range(len(ll)):
                    if self.is_not_number(ll[cnt]):
                        sentence.append(ll[cnt])
                    else:
                        try:
                            sentence.append(
                                self.normalize_data(ll[cnt], table + '.' + str(ll[cnt - 2]), column_min_max_vals))
                        except Exception as e:
                            HintGeneratorLogger.exception(e)
                sentences.append(tuple(sentence))
                rows.append(0)
                pg.append(_rows)
        # print(sentences)
        return sentences, rows, pg

    def prepare_data_and_label(self, sentences, rows, vocab_dict, vocab_size):
        data = []
        label = []
        for sentence, row in zip(sentences, rows):
            _s = []
            sentence = list(filter(lambda item: item in vocab_dict, sentence))
            for word in sentence:
                if self.is_not_number(word):
                    _tmp = np.column_stack((np.array([0]), vocab_dict[word]))
                    _tmp = np.reshape(_tmp, (vocab_size + 1))
                    assert (len(_tmp) == vocab_size + 1)
                    _s.append(_tmp)
                else:
                    _tmp = np.full((vocab_size + 1), word)
                    assert (len(_tmp) == vocab_size + 1)
                    _s.append(_tmp)
            data.append(np.array(_s))
            label.append(row)
        return data, label

    def get_vocabulary_encoding(self, _fp_vocab_dict=None):
        """
            这里必须是和之前一样的vocab
        Returns:

        """
        vocab_dict = {}
        if _fp_vocab_dict is not None:
            with open(_fp_vocab_dict, "rb") as _pkl_f:
                vocab_dict = pickle.load(_pkl_f)
        else:
            vocab_size = len(self.vocabulary)
            _vocabulary = np.array(self.vocabulary)
            label_encoder = preprocessing.LabelEncoder()
            integer_encoded = label_encoder.fit_transform(_vocabulary)
            encoded = to_categorical(integer_encoded)
            for v, e in zip(self.vocabulary, encoded):
                vocab_dict[v] = np.reshape(np.array(e), (1, vocab_size))
        return vocab_dict

    @staticmethod
    def extract_plan(line):
        data = line.replace("->", "").lstrip().split("  ")[-1].split(" ")
        start_cost = data[0].split("..")[0].replace("(cost=", "")
        end_cost = data[0].split("..")[1]
        rows = data[1].replace("rows=", "")
        width = data[2].replace("width=", "").replace(")", "")
        # a_start_cost = data[4].split("..")[0].replace("time=","")
        # a_end_cost = data[4].split("..")[1]
        # a_rows = data[5].replace("rows=","")
        return float(start_cost), float(end_cost), float(rows), float(
            width)  # ,float(a_start_cost),float(a_end_cost),float(a_rows)

    @staticmethod
    def extract_operator(line, operators):
        operator = line.replace("->", "").lstrip().split("  ")[0]
        if operator.startswith("Seq Scan") or operator.startswith("Parallel Seq Scan"):
            operator = "Seq Scan"
        if operator == "Parallel Hash":
            operator = "Hash"
        return operator, operator in operators

    @staticmethod
    def extract_attributes(operator, operators, columns, line, feature_vec, leaf_embedding, i=None):
        operators_count = len(operators)  # 9
        if operator in ["Hash", "Materialize", "Nested Loop", "Gather"]:
            pass
        elif operator == "Merge Join":
            if "Cond" in line:
                for column in columns:
                    if column in line:
                        feature_vec[columns.index(column) + operators_count] = 1.0
        elif operator == "Index Only Scan using title_pkey on title t":
            if "Cond" in line:
                feature_vec[columns.index("t.id") + operators_count] = 1.0
                for column in columns:
                    if column in line:
                        feature_vec[columns.index(column) + operators_count] = 1.0
        elif operator == "Sort":
            for column in columns:
                if column in line:
                    feature_vec[columns.index(column) + operators_count] = 1.0
        elif operator == 'Index Scan using title_pkey on title t':
            if "Cond" in line:
                feature_vec[columns.index("t.id") + operators_count] = 1.0
                for column in columns:
                    if column in line:
                        feature_vec[columns.index(column) + operators_count] = 1.0
        elif operator == 'Hash Join':
            if "Cond" in line:
                for column in columns:
                    if column in line:
                        feature_vec[columns.index(column) + operators_count] = 1.0
        elif operator == 'Seq Scan' or operator == "Parallel Seq Scan":
            feature_vec[15:79] = leaf_embedding[i]
        else:
            pass
        # print(feature_vec)

    @staticmethod
    def normalize_data(val, column_name, column_min_max_vals):
        if column_name not in column_min_max_vals:
            HintGeneratorLogger.debug(f"{column_name} not in dict")
        # # 默认值为 [0, 10]
        min_val, max_val = column_min_max_vals.get(column_name, (0, 10))
        # min_val = column_min_max_vals[column_name][0]
        # max_val = column_min_max_vals[column_name][1]
        val = float(val)
        if val > max_val:
            val = max_val
        elif val < min_val:
            val = min_val
        val = float(val)
        val_norm = (val - min_val) / (max_val - min_val)
        return val_norm

    @staticmethod
    def is_not_number(s):
        try:
            float(s)
            return False
        except ValueError:
            pass
        try:
            import unicodedata
            unicodedata.numeric(s)
            return False
        except (TypeError, ValueError):
            pass
        return True

    @staticmethod
    def padding_sentence(test_data, max_len=9):
        padded_sentences = pad_sequences(test_data, maxlen=max_len, padding='post', dtype='float32')
        return padded_sentences

    @staticmethod
    def get_column_statistics(path):
        with open(path, 'r') as f:
            data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
            column_min_max_vals = {}
            for i, row in enumerate(data_raw[1:]):
                column_min_max_vals[row[0]] = [float(row[1]), float(row[2])]
        return column_min_max_vals

    @staticmethod
    def pred_cost(x, sample_nbr=100, _sc=None, _net=None):
        pred_vals = [_net(x).cpu().item() for _ in range(sample_nbr)]
        pred = np.mean(pred_vals)
        pred = _sc.inverse_transform(pred.reshape(1, 1))[0][0]
        pred_vals = _sc.inverse_transform(np.array(pred_vals).reshape(-1, 1))
        return pred, pred_vals

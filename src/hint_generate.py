import copy
import csv
import itertools
import os
import random
import time
import typing

from tqdm import tqdm

from src.basic import PostgresDB
from src.config import DATA_PATH_OUTPUT, DATA_PATH_LC_SQL_TRAIN_CSV, TEST_DB_CONFIG, DB_LAB_VM_CONFIG
from src.query_plan import QueryPlan


class SQLWithHintsGenerator:
    def __init__(self, train_csv_path: str):
        self.tables, self.join, self.predicate = list(), list(), list()
        self.csv_file_generator = self.load_csv_data(train_csv_path)

    @staticmethod
    def load_csv_data(csv_path: str):
        # Load queries
        with open(csv_path, 'r') as f:
            for tables, join, predicate, _ in csv.reader(f, delimiter='#'):
                yield tables.split(','), join.split(','), predicate.split(',')

    @staticmethod
    def cartesian(element_container: typing.Iterable):
        """
            做笛卡尔基
        """
        return list(itertools.product(*element_container))

    def generate_scan_hints(self):
        """
            生成Scan Hint: Seq Scan 和 Index Scan
            tables形如: ["person p", "table t"]
        """
        scan_methods = ["SeqScan({})", "IndexScan({})"]
        hint_candidate = []
        for table in self.tables:
            # # 取别名
            table_alias = table.split(" ")[1]
            table_candidate = list(map(lambda method: method.format(table_alias), scan_methods))
            hint_candidate.append(table_candidate)
        # # hint_candidate:
        #   [
        #       ['SeqScan(p)', 'IndexScan(p)'],  // p的两种方法
        #       ['SeqScan(t)', 'IndexScan(t)']   // t的两种方法
        #   ]
        candidates = list(map(" ".join, list(itertools.product(*hint_candidate))))
        return candidates

    @staticmethod
    def add_one_rel(cur, join_tables):
        """
            添加一个关系
        """
        extended_order = []
        for table in join_tables:
            if table not in cur:
                extended_order.extend([
                    ["(", *cur, table, ")"],
                    ["(", table, *cur, ")"]
                ])
        return extended_order

    def generate_join_order_hints(self):
        """
            不改了
        """
        # # 取表的别名
        table_alias = [x.split(" ")[1] for x in self.tables]
        # #
        str_order_length = 3 * len(self.tables) - 2
        join_orders = []
        starter = copy.deepcopy(table_alias)
        stack = [[each] for each in starter]
        while len(stack) != 0:
            cur = stack.pop(0)
            if len(cur) < str_order_length:
                extended_orders = self.add_one_rel(cur, table_alias)
                stack.extend(extended_orders)
            else:
                join_orders.append(cur)
        str_join_orders = [" ".join(each) for each in join_orders]
        str_join_orders = set(str_join_orders)  # # 去重
        # # 放表的顺序
        join_orders_string = list(map("Leading ({})".format, str_join_orders))
        return join_orders_string, join_orders

    def construct_sql(self, method):
        tables = ", ".join(self.tables)
        if self.join != [""] and self.predicate != [""]:
            joins = " and ".join(self.join)
            sql = method + " select count(*) from {} where {} and {}"
        elif self.join != [""] and self.predicate == [""]:
            joins = " and ".join(self.join)
            sql = method + " select count(*) from {} where {} {}"
        elif self.join == [""] and self.predicate != [""]:
            joins = ""
            sql = method + " select count(*) from {} where {} {}"
        else:
            joins = ""
            sql = method + " select count(*) from {} {} {}"
        predicates = " and ".join([' '.join(self.predicate[n: n + 3]) for n in range(0, len(self.predicate), 3)])
        return sql.format(tables, joins, predicates) + ";"

    @staticmethod
    def parse_order(order):
        """
            就是获取一个排序
        """
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

    def generate_join_method_hints_from_orders(self):
        """
            join_order_hints: ['Leading (( t mi_idx ))', 'Leading (( mi_idx t ))']
            join_orders_list: [
                ['(', 't', 'mi_idx', ')'],
                ['(', 'mi_idx', 't', ')'],
                ['(', 'mi_idx', 't', ')'],
                ['(', 't', 'mi_idx', ')']
             ]

        """
        join_order_hints, join_orders_list = self.generate_join_order_hints()
        join_methods = ["NestLoop({})", "MergeJoin({})", "HashJoin({})"]

        join_hints = []

        for order_hint, order in zip(join_order_hints, join_orders_list):
            parsed_order = self.parse_order(order)
            # # JOIN ORDER
            join_order = []
            for idx in range(2, len(parsed_order) + 1):
                join_order.append(" ".join(parsed_order[0:idx]))
            # #
            join_candidate = []
            for level in join_order:
                join_candidate.append([each.format(level) for each in join_methods])
            candidates = list(map(lambda x: " ".join(x), list(itertools.product(*join_candidate))))
            join_hints.extend(list(map(lambda each: f"{each} {order_hint}", candidates)))
        if not join_hints:
            join_hints = [""]
        return join_hints

    def generate_hint_queries(self, command):
        # # 获取scan_hints提示
        scan_hints = self.generate_scan_hints()
        # # 生成 Join Order的Hint

        join_hints = self.generate_join_method_hints_from_orders()

        # # 生成sql
        sql = self.construct_sql(command)
        # # 通过笛卡尔积排列组合生成所有的hints
        queries = list(
            map(lambda each: f"/*+ {each} */ {sql}", map(" ".join, list(itertools.product(*[scan_hints, join_hints])))))

        return queries, sql

    def __iter__(self):
        return self

    def __next__(self):
        try:
            self.tables, self.join, self.predicate = next(self.csv_file_generator)
            queries_with_hint, sql = self.generate_hint_queries(command="explain")
            return queries_with_hint, sql
        except StopIteration:
            raise StopIteration

    @staticmethod
    def test():
        for idx, (query_with_hint, sql) in enumerate(tqdm(SQLWithHintsGenerator(DATA_PATH_LC_SQL_TRAIN_CSV))):
            if idx > 100:
                break
            output_dir = os.path.join(DATA_PATH_OUTPUT, f"output-{idx}")
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, f"{idx}a.sql"), "w", encoding="utf-8") as f:
                f.write(sql)
            with open(os.path.join(output_dir, f"{idx}b.sql"), "w", encoding="utf-8") as f:
                f.write('\n'.join(query_with_hint))
            # print(query_with_hint, sql)
            # print(len(query_with_hint))
            pass


class CostEstimation:
    THRESHOLD = 10  # # 可选用的查询方案大于10, 就随机抽选10条SQL语句

    def __init__(self, sql: str, sql_with_hints: list[str], query_plan_save_dir: str):
        self.sql: str = sql
        self.sql_with_hints: list[str] = copy.deepcopy(sql_with_hints)
        self.query_plan_save_dir = query_plan_save_dir
        self.query_plans = self._get_query_plan_tree()

    def _get_query_plan_tree(self):
        """

        """
        os.makedirs(self.query_plan_save_dir, exist_ok=True)
        QueryPlan.EVAL_MODE = True
        # # 随机抽选 `THRESHOLD`条SQL语句
        if len(self.sql_with_hints) > self.THRESHOLD:
            random.shuffle(self.sql_with_hints)
            self.sql_with_hints = self.sql_with_hints[:10]
        # # 获取查询计划
        db = PostgresDB(_config=DB_LAB_VM_CONFIG, _try_connect=True)
        res = list()
        for idx, sql_with_hint in enumerate(tqdm(self.sql_with_hints)):
            query_plan = db.execute(sql_with_hint, )
            query_plan_raw = "\n".join(map(lambda _items: _items[0], query_plan))
            with open(os.path.join(self.query_plan_save_dir, f"plan-{idx}"), "w",
                      encoding="utf-8") as f:
                f.write(sql_with_hint)
                f.write("\n")
                f.write(query_plan_raw)
            res.append(QueryPlan(query_plan_raw))
        return res

    def predict_costs(self):
        from src.cost_learner import CostLearner

        c = CostLearner.default_factory()

        costs = list(map(lambda qp: c.predict_cost_and_get_confidence_intervals_eval_mode(qp, ci_multiplier=30, c=c),
                         self.query_plans))
        return costs

    @staticmethod
    def load_from(idx: int):
        test_dir = os.path.join(DATA_PATH_OUTPUT, f"output-{idx}")
        with open(os.path.join(test_dir, f"{idx}a.sql"), "r", encoding="utf-8") as f:
            sql = f.read()
        with open(os.path.join(test_dir, f"{idx}b.sql"), "r", encoding="utf-8") as f:
            sql_with_hints = f.readlines()
        return CostEstimation(sql, sql_with_hints, os.path.join(test_dir, "plans"))

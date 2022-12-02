import enum
import logging
import re
import typing

import networkx as nx
import rich
from matplotlib import pyplot as plt

from src.basic import GeneralLogger, SequenceGenerator

ParseQueryPlanLogger = GeneralLogger(name="parse_query_plan", stdout_flag=True, stdout_level=logging.DEBUG,
                                     file_mode="a")


@enum.unique
class OperatorType(enum.Enum):
    Aggregate = "Aggregate"
    BitMapHeapScan = "Bitmap Heap Scan"
    BitMapIndexScan = "Bitmap Index Scan"
    Gather = "Gather"
    Hash = "Hash"
    HashJoin = "Hash Join"
    IndexOnlyScan = "Index Only Scan"
    IndexScan = "Index Scan"
    Memorize = "Memoize"
    NestedLoop = "Nested Loop"
    PartialAggregate = "Partial Aggregate"
    SeqScan = "Seq Scan"
    Materialize = "Materialize"
    FinalizeAggregate = "Finalize Aggregate"
    NotImplemented = "Not Implemented"

    @staticmethod
    def from_str(_str: str):
        try:
            return OperatorType(_str)
        except Exception as e:
            ParseQueryPlanLogger.info(f"{e}, {_str}")
            return OperatorType.NotImplemented


class CostInfo:
    def __init__(self, _content: str, _with_actual=True):
        """
            查询计划某步骤的成本信息
        """
        self.content: str = _content.strip()
        self.estimate_start_cost: float = 0
        self.estimate_end_cost: float = 0
        self.estimate_rows_involved: int = 0
        self.actual_start_cost: float = 0
        self.actual_end_cost: float = 0
        self.actual_rows_involved: int = 0
        self.actual_loops: int = 0
        self.parse_content()

    def parse_content(self):
        """
            self.content.count("(") == 2 有两组括号
            '(never executed)' not in self.content 某些操作最后带有该标记
        Returns:

        """

        self.estimate_start_cost, self.estimate_end_cost, self.estimate_rows_involved, \
            self.actual_start_cost, self.actual_end_cost, self.actual_rows_involved, \
            self.actual_loops = map(float, re.match(
            "\(cost=(.*?)\.\.(.*?) rows=(.*?) width=.*?\) \(actual time=(.*?)\.\.(.*?) rows=(.*?) loops=(.*?)\)",
            self.content).groups() if self.content.count("(") == 2 and '(never executed)' not in self.content else (
            *re.match("\(cost=(.*?)\.\.(.*?) rows=(.*?) width=.*?\)", self.content).groups(), 0, 0, 0, 0))

    def __str__(self):
        return f"estimate cost:{self.estimate_end_cost}, actual cost: {self.actual_end_cost}"


@enum.unique
class CondExpType(enum.IntEnum):
    """
            0: IS NULL
            1: IS NOT NULL
            2: ANY/ALL复合表达式
            3: 简单判断表达式
    """
    IS_NULL = 0
    IS_NOT_NULL = 1
    ANY_ALL_COMP_EXP = 2
    SIMPLE_COMPARE = 3


class Operator:
    __NO_SENSE_VARI__BUT_ZERO = 0

    def __init__(self, _content):
        """
            关注时间信息而不关注其他内存消耗信息,
        """
        self.content: str = _content
        self.op_type: OperatorType = OperatorType.NotImplemented
        # # for scan
        self.op_on: typing.Optional[list[str, str]] = list()
        self.index_using: str = ''
        # # ... filters and logic
        # # 不同的逻辑表达式的查询估计是不同的,
        # # todo 最好的方式还是做成将filter表达成树的形式
        # # 但这过于复杂了. 现在这种形式表达不了节点之间的关系
        self.filters: typing.List[typing.Tuple[typing.Union[CondExpType, str], ...]] = list()
        self.filters_sep: typing.List[str] = list()
        self.index_or_join_cond: typing.List[typing.Tuple[typing.Union[CondExpType, str], ...]] = list()
        self.index_or_join_cond_sep: typing.List[str] = list()
        self.parallel = False
        # # cost信息
        self.cost_info: typing.Optional[CostInfo] = None
        self.parse_content()

    def parse_content(self):
        """
            针对不同的指令做不同的分析
        Returns:

        """
        _lines: tuple[str] = tuple(map(lambda _s: str(_s).strip(), self.content.split("\n")))
        _operator_line = _lines[0].strip("->").strip()
        if _operator_line.startswith("Parallel "):
            _operator_line = _operator_line.lstrip("Parallel ")
            self.parallel = True
        _additional_info_lines = _lines[1:]
        # # 分析 operator_line
        ParseQueryPlanLogger.debug(_operator_line)
        _operator_complicated, _cost_info_raw = _operator_line.split("  ")
        # # 获取 该操作的代价信息
        self.cost_info = CostInfo(_cost_info_raw)
        # # 拆分操作信息 根据 on 和 using
        _temp_split_res = re.split(" on | using ", _operator_complicated)
        ParseQueryPlanLogger.debug(_temp_split_res)
        if len(_temp_split_res) == 1:
            # # Gather, Hash, Hash Join, Memoize, Nested Loop, Parallel Hash, Parallel Hash Join, Partial Aggregate,
            # Aggregate
            self.op_type = OperatorType.from_str(_temp_split_res[0])
        elif len(_temp_split_res) == 2:
            # # Bitmap Heap Scan, Bitmap Index Scan, Parallel Bitmap Heap Scan, Parallel Seq Scan, Seq Scan
            self.op_type = OperatorType.from_str(_temp_split_res[0])
            self.op_on = _temp_split_res[1].split(" ")
        else:
            assert len(_temp_split_res) == 3
            # # Index Only Scan, Index Scan
            self.op_type = OperatorType.from_str(_temp_split_res[0])
            self.index_using = _temp_split_res[1]
            self.op_on = _temp_split_res[2].split(" ")
        if hasattr(self, f"_do_{self.op_type.name}"):
            # # only execute once, and one method
            ParseQueryPlanLogger.debug(f"_do_{self.op_type.name}")
            getattr(self, f"_do_{self.op_type.name}")(_additional_info_lines)
        else:
            ParseQueryPlanLogger.warning(f"no method named _do_{self.op_type.name}")

    def parse_filter_cond_exp(self, cond_exp):
        self.parse_cond_exp(cond_exp, self.filters, self.filters_sep)

    def parse_index_recheck_join_cond_exp(self, cond_exp):
        self.parse_cond_exp(cond_exp, self.index_or_join_cond, self.index_or_join_cond_sep)

    @staticmethod
    def remove_prefix(_exp, _prefix):
        if _exp.startswith(_prefix):
            _exp = _exp.lstrip(_prefix)
        return _exp

    @staticmethod
    def remove_suffix(_exp, _suffix):
        if _exp.endswith(_suffix):
            _exp = _exp.rstrip(_suffix)
        return _exp

    @staticmethod
    def remove_suffix_and_suffix(_exp, _prefix, _suffix):
        if _exp.startswith(_prefix):
            _exp = _exp.lstrip(_prefix)
        if _exp.endswith(_suffix):
            _exp = _exp.rstrip(_suffix)
        return _exp

    @staticmethod
    def parse_cond_exp(cond_exp, container, cond_sep: list[str]):
        """
            获取所有的条件表达式,
            0: IS NULL
            1: IS NOT NULL
            2: ANY/ALL复合表达式
            3: 简单判断表达式
        Args:
            cond_exp:
        Returns:

        """
        cond_exp = cond_exp[1:-1]  # # 去掉括号
        cond_sep.extend(re.findall("AND NOT|OR NOT|AND|OR", cond_exp))
        # # 找到全部的逻辑表达式
        if cond_sep:
            _filters: list[str] = list(
                map(lambda _s: _s[1:-1], re.sub("AND NOT|OR NOT|AND|OR", '', cond_exp).split("  ")))
        else:
            _filters: list[str] = [cond_exp]
        for _exp in _filters:
            _exp = Operator.remove_suffix_and_suffix(_exp, "(", "::text)")

            if re.match(".*?([>=<\']|~~|!~~|<>).*?", _exp) is None:
                if "IS NULL" in _exp:
                    container.append(
                        (CondExpType.IS_NULL, Operator.remove_suffix_and_suffix(_exp, "(", " IS NULL"), "IS NULL"))
                elif "IS NOT NULL" in _exp:
                    container.append(
                        (CondExpType.IS_NOT_NULL, Operator.remove_suffix_and_suffix(_exp, "(", " IS NOT NULL"),
                         "IS NOT NULL"))
            else:
                _match_res = re.match("(.*?) (.*?) (ANY|ALL|NOT) \(\'{(.*?)}\'::(.*?)\[\]", _exp)
                if _match_res is not None:
                    # # 事实上, 只有ANY, 也只有text一种 但是最好是都匹配出来
                    _t_keyword, _t_op, _t_pred, _words, _array_type = _match_res.groups()
                    _t_keyword = Operator.remove_suffix_and_suffix(_t_keyword, "(", ")::text")
                    _words_split = _words.split(",")
                    ParseQueryPlanLogger.debug(_match_res)
                    container.append(
                        (CondExpType.ANY_ALL_COMP_EXP, _t_keyword, _t_op, _t_pred, *_words_split, _array_type))
                else:
                    # # 匹配到一组条件, 形如: keyword = ANY ('{superhero,marvel-comics,based-on-comic,fight}'::text[])
                    # # 匹配到条件, 拆成词, 直接拆成词 就可以啦, 记住去掉一些text修饰
                    _match_res = re.sub("::text|\(|\)", "", _exp).split(" ")
                    _match_res = [_match_res[0], _match_res[1], " ".join(_match_res[2:])]
                    container.append((CondExpType.SIMPLE_COMPARE, *_match_res))

    def __str__(self):
        return f'{self.op_type} {self.cost_info} parallel: {self.parallel}'

    def _do_Aggregate(self, _additional_info_lines: list[str]):
        """
            如果SQL中有涉及到了聚合操作, 一般情况下顶部都是这个操作符
          ...
        Args:
            _additional_info_lines: 应该是一个空列表

        Returns:

        """
        # # make pep8 happy, self.__NO_SENSE_VARI__BUT_ZERO = 0
        # # do not ask me why do that, you know, I don't want
        # # 判断是否为0, 必须为0, 否则就出问题了
        assert len(_additional_info_lines) == self.__NO_SENSE_VARI__BUT_ZERO

    def _do_BitMapHeapScan(self, _additional_info_lines: list[str]):
        """
            BitMapHeapScan如下所示, 需要的是第一行的Recheck Cond, Heap Blocks 没用
        ->  Bitmap Heap Scan on movie_keyword mk  (cost=7.62..1475.63 rows=411 width=8) (actual time=1.186..29.129 rows=6453 loops=8)
          Recheck Cond: (keyword_id = k.id)
          Heap Blocks: exact=17767
          Filter: (info > '7.0'::text) // 可能会有
        Args:
            _additional_info_lines:

        Returns:

        """
        assert len(_additional_info_lines) > 0 and _additional_info_lines[0].startswith("Recheck Cond: ")
        # #
        self.parse_index_recheck_join_cond_exp(Operator.remove_prefix(_additional_info_lines[0], "Recheck Cond: "))
        # # 解析Filter表达式
        if len(_additional_info_lines) > 1 and _additional_info_lines[1].startswith("Filter: "):
            self.parse_filter_cond_exp(Operator.remove_prefix(_additional_info_lines[1], "Filter: "))

    def _do_BitMapIndexScan(self, _additional_info_lines: list[str]):
        """
            这是一个叶子结点操作, 一般是在BitMapHeapScan操作之后
          ->  Bitmap Index Scan on info_type_id_movie_info_idx  (cost=0.00..4.18 rows=4 width=0) (actual time=0.001..0.001 rows=0 loops=1)
            Index Cond: (info_type_id = it2.id)
        Args:
            _additional_info_lines:

        Returns:

        """
        assert len(_additional_info_lines) > 0 and _additional_info_lines[0].startswith("Index Cond: ")
        self.parse_index_recheck_join_cond_exp(Operator.remove_prefix(_additional_info_lines[0], "Index Cond: "))

    def _do_Gather(self, _additional_info_lines: list[str]):
        """
            这个操作也没有太多可说的, 啥也不需要写
          ->  Gather  (cost=479312.44..479312.65 rows=2 width=64) (actual time=3689.332..3748.970 rows=3 loops=1)
                Workers Planned: 2
                Workers Launched: 2
        Args:
            _additional_info_lines:

        Returns:

        """

    def _do_Hash(self, _additional_info_lines: list[str]):
        """
            暂时没有时间相关信息, 啥也不做, 下面的子节点情况比较复杂, 好像和 Hash Join操作相关
        ->  Parallel Hash  (cost=106530.36..106530.36 rows=1393 width=128) (actual time=12110.346..12110.464 rows=980091 loops=3)
            Buckets: 4096 (originally 4096)  Batches: 1024 (originally 1)  Memory Usage: 2880kB
        Args:
            _additional_info_lines:

        Returns:

        """

    def _do_HashJoin(self, _additional_info_lines: list[str]):
        """
        ->  Parallel Hash Join  (cost=7227.32..68169.09 rows=786713 width=4) (actual time=165.534..704.919 rows=738720 loops=3)
              Hash Cond: (mc.company_id = cn.id)
        Args:
            _additional_info_lines:

        Returns:

        """
        assert len(_additional_info_lines) > 0 and _additional_info_lines[0].startswith("Hash Cond: ")
        self.parse_index_recheck_join_cond_exp(Operator.remove_prefix(_additional_info_lines[0], "Hash Cond: "))

    def _do_IndexOnlyScan(self, _additional_info_lines: list[str]):
        """
        同理
          ->  Index Only Scan using name_pkey on name n1  (cost=0.43..1.35 rows=1 width=4) (actual time=0.002..0.002 rows=1 loops=648808)
                Index Cond: (id = ci.person_id)
                Heap Fetches: 18
        Args:
            _additional_info_lines:

        Returns:

        """
        assert len(_additional_info_lines) > 0 and _additional_info_lines[0].startswith("Index Cond: ")
        self.parse_index_recheck_join_cond_exp(Operator.remove_prefix(_additional_info_lines[0], "Index Cond: "))

    def _do_IndexScan(self, _additional_info_lines: list[str]):
        """
        同理
        ->  Index Scan using movie_info_idx_mid on movie_info mi  (cost=0.44..2.33 rows=1 width=4) (actual time=0.613..0.631 rows=0 loops=345)
              Index Cond: (movie_id = mk.movie_id)
              Filter: (info = ANY ('{Sweden,Norway,Germany,Denmark,Swedish,Denish,Norwegian,German}'::text[])) // 可能不存在
              Rows Removed by Filter: 48 // 可能不存在

        Args:
            _additional_info_lines:

        Returns:

        """
        assert len(_additional_info_lines) > 0 and _additional_info_lines[0].startswith("Index Cond: ")
        self.parse_index_recheck_join_cond_exp(Operator.remove_prefix(_additional_info_lines[0], "Index Cond: "))
        if len(_additional_info_lines) > 1:
            self.parse_filter_cond_exp(Operator.remove_prefix(_additional_info_lines[1], "Filter: "))

    def _do_Memorize(self, _additional_info_lines: list[str]):
        """
            这是一个缓存操作, 没有找到相关的例子
        Args:
            _additional_info_lines:

        Returns:

        """
        pass

    def _do_NestedLoop(self, _additional_info_lines: list[str]):
        """
        ->  Nested Loop  (cost=9.32..1639.68 rows=2 width=16) (actual time=5.076..472.341 rows=1233 loops=1)
            Join Filter: (mc.movie_id = t.id) // 可能有 可能没有
        Args:
            _additional_info_lines:

        Returns:

        """
        # # 此处是JOIN CONDITION
        if len(_additional_info_lines) > 0 and _additional_info_lines[0].startswith("Join Filter: "):
            self.parse_index_recheck_join_cond_exp(Operator.remove_prefix(_additional_info_lines[0], "Join Filter: "))

    def _do_PartialAggregate(self, _additional_info_lines: list[str]):
        """
            部分聚合 Gather下的第一个子节点,
            ->  Partial Aggregate  (cost=134331.74..134331.75 rows=1 width=64) (actual time=19418.839..19419.060 rows=1 loops=3)
        Args:
            _additional_info_lines:

        Returns:

        """
        pass

    def _do_SeqScan(self, _additional_info_lines: list[str]):
        """
            Seq Scan 经典子节点
          ->  Seq Scan on info_type it  (cost=0.00..2.41 rows=1 width=4) (actual time=0.027..0.030 rows=1 loops=3)
            Filter: ((info)::text = 'mini biography'::text) (可能有 可能没有)
            Rows Removed by Filter: 112
        Args:
            _additional_info_lines:

        Returns:

        """
        if len(_additional_info_lines) > 0 and _additional_info_lines[0].startswith("Filter: "):
            self.parse_filter_cond_exp(Operator.remove_prefix(_additional_info_lines[0], "Filter: "))

    def _do_Materialize(self, _additional_info_lines: list[str]):
        """
            materialize: 具体化, 具象化
            A `Materialize` node means the output of whatever is below it in the tree
            (which can be a scan, or a full set of joins or something like that)
             is materialized into memory before the upper node is executed.
             This is usually done when the outer node needs a source that it can re-scan for some reason or other.
             So in your case, the planner is determining that
             the result of a scan on one of your tables will fit in memory,
            and it till make it possible to choose an upper join operation
            that requires rescans while still being cheaper.
        Args:
            _additional_info_lines:

        Returns:

        """
        pass

    def _do_NotImplemented(self, _additional_info_lines: list[str]):
        """
            这里报一个非法操作
        Args:
            _additional_info_lines:

        Returns:

        """
        # # make pep8 happy
        ParseQueryPlanLogger.warning(f"not implemented operation found {'*' * self.__NO_SENSE_VARI__BUT_ZERO}")

    def _do_FinalizeAggregate(self, _additional_info_lines: list[str]):
        pass


@enum.unique
class QueryPlanNodeType(enum.IntEnum):
    LEAF_NODE = 0
    MIDDLE_NODE = 1
    ROOT_NODE = 2


class QueryPlanNode:
    __slots__ = [
        'parent', 'left', 'right', 'layer',
        'content', 'operator', 'id'
    ]

    def __init__(self, _id: int, _parent=None, _content: str = None, _layer=0):
        # # 二叉树的结构信息
        self.parent: typing.Optional[QueryPlanNode] = _parent
        self.left: typing.Optional[QueryPlanNode] = None
        self.right: typing.Optional[QueryPlanNode] = None
        self.layer = _layer
        # # 二叉树的内容
        self.content: str = _content.strip().strip("->")
        self.operator: Operator = Operator(_content)
        self.id = _id

    @property
    def is_root(self):
        return self.parent is None

    @property
    def is_leaf(self):
        return self.left is None and self.right is None

    def add_child(self, child: 'QueryPlanNode'):
        assert self.left is None or self.right is None, child
        if self.left is None:
            self.left = child
        else:
            self.right = child

    @property
    def node_type(self):
        if self.is_root:
            return QueryPlanNodeType.ROOT_NODE
        elif self.is_leaf:
            return QueryPlanNodeType.LEAF_NODE
        else:
            return QueryPlanNodeType.MIDDLE_NODE

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f'QueryPlanNode<{self.node_type.name}>[{self.operator}]: id: {self.id}, layer: {self.layer}, ' \
               f'parent: {self.parent.id if self.parent else None}'

    def node_ir(self):
        """
            作为节点的中间表示
        Returns:

        """
        return f'{self.node_type.name}[[{self.operator}]: id: {self.id}]'


class QueryPlan:
    COND_EXP = list()

    def __init__(self, _query_plan_raw: str):
        self.id_node: typing.Dict[int, QueryPlanNode] = dict()
        self.__g = SequenceGenerator()
        self.query_plan_raw = _query_plan_raw
        self.root_node: typing.Optional[QueryPlanNode] = None
        self.parse_content()
        self.planing_time = 0
        self.execution_time = 0

    def parse_content(self):
        cur_layer = 0
        cur_blank = 0
        blank2layer = {0: 0}
        gene = tuple(self.operators_generator(self.query_plan_raw))
        self.root_node = QueryPlanNode(next(self.__g), None, gene[0], cur_layer)
        self.id_node[self.root_node.id] = self.root_node
        cur_node = self.root_node
        for op_raw in gene[1:]:
            _arrow_pos = op_raw.find("->")
            if _arrow_pos > cur_blank:
                # # 达到下一层次
                cur_layer += 1
                cur_blank = _arrow_pos
                blank2layer[cur_blank] = cur_layer

                new_node = QueryPlanNode(next(self.__g), cur_node, op_raw, cur_layer)
                self.id_node[new_node.id] = new_node

                cur_node.add_child(new_node)
                cur_node = new_node
            elif _arrow_pos == cur_blank:
                # # 一个新的节点
                new_node = QueryPlanNode(next(self.__g), cur_node.parent, op_raw, cur_layer)
                self.id_node[new_node.id] = new_node

                cur_node.parent.add_child(new_node)
                cur_node = new_node
            else:
                cur_blank = _arrow_pos
                fallback_layer = blank2layer[cur_blank]
                for _ in range(cur_layer - fallback_layer + 1):
                    cur_node = cur_node.parent
                cur_layer = fallback_layer

                new_node = QueryPlanNode(next(self.__g), cur_node, op_raw, cur_layer)
                self.id_node[new_node.id] = new_node

                cur_node.add_child(new_node)
                cur_node = new_node
        try:
            plan_line, exec_line = gene[-1].split("\n")[-2:]
            self.planing_time = float(re.match(".*? Time: (.*?) ms", plan_line).groups()[0])
            self.execution_time = float(re.match(".*? Time: (.*?) ms", exec_line).groups()[0])
        except Exception as e:
            ParseQueryPlanLogger.warning(f"query plan raw error: {e}, no time output")

    @staticmethod
    def operators_generator(_content):
        _temp = list()
        for _line in _content.split("\n"):
            if '->' in _line:
                yield '\n'.join(_temp)
                _temp = [_line]
            else:
                _temp.append(_line)
        yield '\n'.join(_temp)

    def pre_order(self):
        _planing_time = 0

        def __pre_order(_cur_node: QueryPlanNode):
            if _cur_node is not None:
                print(_cur_node)

                # print(_cur_node.operator.filters)
                # print(_cur_node.operator.filters_sep)
                # print(_cur_node.operator.index_or_join_cond)
                # print(_cur_node.operator.index_or_join_cond_sep)
                self.COND_EXP.extend(_cur_node.operator.filters)
                self.COND_EXP.extend(_cur_node.operator.index_or_join_cond)
                __pre_order(_cur_node.left)
                __pre_order(_cur_node.right)

        __pre_order(self.root_node)

    def make_digraph(self, draw=True):
        g = nx.DiGraph()
        for n_id, node in self.id_node.items():
            g.add_node(n_id, intro=node.node_ir(), subset=node.node_type)
        for _, node in self.id_node.items():
            if node.left is not None:
                g.add_edge(node.id, node.left.id)
            if node.right is not None:
                g.add_edge(node.id, node.right.id)
        if draw:
            plt.title('draw_networkx')
            pos = nx.drawing.nx_agraph.graphviz_layout(g, prog='neato')
            nx.draw(g, pos, with_labels=True, arrows=True, node_size=500)
            plt.show()
            plt.savefig('nx_test.png')

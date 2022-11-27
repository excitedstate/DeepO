import enum
import logging
import re
import typing
from collections import namedtuple

from src.basic import GeneralLogger

ParseQueryPlanLogger = GeneralLogger(name="parse_query_plan", stdout_flag=True, stdout_level=logging.INFO,
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
    SeqSCAN = "Seq Scan"
    Materialize = "Materialize"
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
        self.with_actual = _with_actual
        self.parse_content()

    def parse_content(self):
        self.estimate_start_cost, self.estimate_end_cost, self.estimate_rows_involved, self.actual_start_cost, self.actual_end_cost, self.actual_rows_involved, self.actual_loops = re.match(
            "\(cost=(.*?)\.\.(.*?) rows=(.*?) width=.*?\) \(actual time=(.*?)\.\.(.*?) rows=(.*?) loops=(.*?)\)",
            self.content).groups() if self.with_actual else (
            *re.match("\(cost=(.*?)\.\.(.*?) rows=(.*?) width=.*?\)", self.content).groups(), 0, 0, 0, 0)

    def __str__(self):
        return f"estimate cost:{self.estimate_end_cost}, actual cost: {self.actual_end_cost}"


class Operator:
    __NO_SENSE_VARI__BUT_ZERO = 0

    def __init__(self, _content, _with_actual=True):
        """
            关注时间信息而不关注其他内存消耗信息,
        """
        self.content: str = _content
        self.with_actual = _with_actual
        self.op_type: OperatorType = OperatorType.NotImplemented
        # # for scan
        self.op_on: typing.Optional[list[str, str]] = list()
        self.index_using: str = ''
        # # ... filters and logic
        # # 不同的逻辑表达式的查询估计是不同的,
        # # todo 最好的方式还是做成将filter表达成树的形式
        # # 但这过于复杂了. 现在这种形式表达不了节点之间的关系
        self.filters: typing.List[typing.Tuple[str, ...]] = list()
        self.filters_sep: typing.List[str] = list()
        self.index_or_join_cond: typing.List[str] = list()
        # # cost信息
        self.cost_info: typing.Optional[CostInfo] = None

    def parse_content(self):
        """
            针对不同的指令做不同的分析
        Returns:

        """
        _lines = tuple(map(lambda _s: _s.strip(), self.content.split("\n")))
        _operator_line = _lines[0]
        _additional_info_lines = _lines[1:]
        # # 分析 operator_line
        _operator_complicated, _cost_info_raw = _operator_line.split("  ")
        # # 获取 该操作的代价信息
        self.cost_info = CostInfo(_cost_info_raw)
        # # 拆分操作信息
        _temp_split_res = re.split(" on | using ", _operator_complicated)
        if len(_temp_split_res) == 1:
            # # Gather, Hash, Hash Join, Memoize, Nested Loop, Parallel Hash, Parallel Hash Join, Partial Aggregate,
            # Aggregate
            self.op_type = Operator(_temp_split_res[0])
        elif len(_temp_split_res) == 2:
            # # Bitmap Heap Scan, Bitmap Index Scan, Parallel Bitmap Heap Scan, Parallel Seq Scan, Seq Scan
            self.op_type = Operator(_temp_split_res[0])
            self.op_on = _temp_split_res[2].split(" ")
        else:
            assert len(_temp_split_res) == 3
            # # Index Only Scan, Index Scan
            self.op_type = Operator(_temp_split_res[0])
            self.index_using = _temp_split_res[1]
            self.op_on = _temp_split_res[2].split(" ")
        if hasattr(self, f"_do_{self.op_type}"):
            getattr(self, f"_do_{self.op_type}")(_additional_info_lines)

    def parse_cond_exp(self, cond_exp):
        # # 做拆分
        self.filters_sep = re.findall("AND NOT|OR NOT|AND|OR", cond_exp)
        # # 找到全部的逻辑表达式
        _filters: list[str] = list(map(lambda _s: _s[1:-1], re.sub("AND NOT|OR NOT|AND|OR", '', cond_exp).split("  ")))
        for _exp in _filters:
            if re.match(">|=|<|'", _exp) is None:
                if "IS NULL" in _exp:
                    self.filters.append((_exp.replace(" IS NULL", ""), "IS NULL"))
                elif "IS NOT NULL" in _exp:
                    self.filters.append((_exp.replace(" IS NULL", ""), "IS NULL"))
            else:
                # # 匹配到条件, 拆成词, 直接拆成词 就可以啦
                self.filters.append(tuple(re.sub("::text|\(|\)", "", _exp).split(" ")))

    def _do_Aggregate(self, _additional_info_lines: list[str]):
        # # make pep8 happy, self.__NO_SENSE_VARI__BUT_ZERO = 0
        # # do not ask me why do that, you know, I don't want
        assert len(_additional_info_lines) == self.__NO_SENSE_VARI__BUT_ZERO

    def _do_BitMapHeapScan(self, _additional_info_lines: list[str]):
        pass

    def _do_BitMapIndexScan(self, _additional_info_lines: list[str]):
        pass

    def _do_Gather(self, _additional_info_lines: list[str]):
        pass

    def _do_Hash(self, _additional_info_lines: list[str]):
        pass

    def _do_HashJoin(self, _additional_info_lines: list[str]):
        pass

    def _do_IndexOnlyScan(self, _additional_info_lines: list[str]):
        pass

    def _do_IndexScan(self, _additional_info_lines: list[str]):
        pass

    def _do_Memorize(self, _additional_info_lines: list[str]):
        pass

    def _do_NestedLoop(self, _additional_info_lines: list[str]):
        # # 此处是JOIN CONDITION
        assert len(_additional_info_lines) == 1 and _additional_info_lines[0].startswith("Join Filter: ")
        # #
        _cond_exp = _additional_info_lines[0].strip("Join Filter: ")

    def _do_PartialAggregate(self, _additional_info_lines: list[str]):
        pass

    def _do_SeqSCAN(self, _additional_info_lines: list[str]):
        pass

    def _do_Materialize(self, _additional_info_lines: list[str]):
        pass

    def _do_NotImplemented(self, _additional_info_lines: list[str]):
        pass


@enum.unique
class QueryPlanNodeType(enum.StrEnum):
    LEAF_NODE = "leaf_node"
    MIDDLE_NODE = "middle_node"
    ROOT_NODE = "root_node"


class QueryPlanNode:
    __slots__ = [
        'parent', 'left', 'right', 'layer', 'is_root', 'is_leaf',
        'content', 'operator', 'add_child'
    ]

    def __init__(self, _parent=None, _content: str = None, _layer=0):
        # # 二叉树的结构信息
        self.parent: typing.Optional[QueryPlanNode] = _parent
        self.left: typing.Optional[QueryPlanNode] = None
        self.right: typing.Optional[QueryPlanNode] = None
        self.layer = _layer
        # # 二叉树的内容
        self.content: str = _content.strip().strip("->")
        self.operator: Operator = Operator(_content)

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

    def __str__(self):
        return f'QueryPlanNode<{self.node_type}>[{self.content}]'


class QueryPlan:
    def __init__(self, _query_plan_raw: str):
        self.query_plan_raw = _query_plan_raw
        self.root_node: typing.Optional[QueryPlanNode] = None
        self.parse_content()

    def parse_content(self):
        cur_layer = 0
        cur_blank = 0
        blank2layer = {0: 0}
        gene = tuple(self.operators_generator(self.query_plan_raw))
        self.root_node = QueryPlanNode(None, gene[0], cur_layer)
        cur_node = self.root_node
        for op_raw in gene[1:]:
            _arrow_pos = op_raw.find("->")
            if _arrow_pos > cur_blank:
                # # 达到下一层次
                cur_layer += 1
                cur_blank = _arrow_pos
                blank2layer[cur_blank] = cur_layer
                new_node = QueryPlanNode(cur_node, op_raw, cur_layer)
                cur_node.add_child(new_node)
                cur_node = new_node
            elif _arrow_pos == cur_blank:
                # # 一个新的节点
                new_node = QueryPlanNode(cur_node.parent, op_raw, cur_layer)
                cur_node.parent.add_child(new_node)
                cur_node = new_node
            else:
                cur_blank = _arrow_pos
                fallback_layer = blank2layer[cur_blank]
                for _ in range(cur_layer - fallback_layer + 1):
                    cur_node = cur_node.parent
                cur_layer = fallback_layer
                new_node = QueryPlanNode(cur_node, op_raw, cur_layer)
                cur_node.add_child(new_node)
                cur_node = new_node

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
        def __pre_order(_cur_node: QueryPlanNode):
            if _cur_node is not None:
                print(_cur_node)
                __pre_order(_cur_node.left)
                __pre_order(_cur_node.right)

        __pre_order(self.root_node)

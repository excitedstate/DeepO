"""
    main scrip, all invoke
"""
import logging
import os.path

from src.basic import PostgresDB, GeneralLogger
from src.config import (MODEL_PATH_COST_MODEL_PREFIX, MODEL_PATH_STD_SCALER, MODEL_PATH_EMBEDDING_MODEL,
                        DATA_PATH_LC_SQL_TRAIN_CSV, DATA_PATH_GOT_PLANS, DATA_PATH_LC_COLUMN_MIN_MAX_VALS_CSV,
                        DATA_PATH_PLANS_FOR_TRAIN, DB_LAB_VM_CONFIG, DATA_PATH_SQL_FOR_PLANS, DATA_PATH)
from torch import nn

from blitz.modules import BayesianLSTM
from blitz.utils import variational_estimator

MainLogger = GeneralLogger(name="main", stdout_flag=True, stdout_level=logging.INFO,
                           file_mode="a")


def tree_embedding_encode():
    """
    需要先获取训练数据

    Returns:

    """
    from src.embedding import ScanEmbedding, PlanSequential

    scan_embedding = ScanEmbedding(
        _fpath_column_min_max_vals=DATA_PATH_LC_COLUMN_MIN_MAX_VALS_CSV,
        _dir_query_plan=DATA_PATH_PLANS_FOR_TRAIN
    )
    scan_embedding.flow(
        _model_save_path=MODEL_PATH_EMBEDDING_MODEL,
        _save_path_output=f"data/scan_embedding_output.npy",
        _save_path_vectors="data/vectors.csv",
        _save_path_labels="data/labels.csv"
    )

    plan_sequential_encode = PlanSequential(
        _embedding_res_path=os.path.join(DATA_PATH, "output.npy"),
        _folder_name=DATA_PATH_PLANS_FOR_TRAIN
    )
    plan_sequential_encode.flow(
        fp_job_cardinality_sequence=os.path.join(DATA_PATH, "job-cardinality-sequence.pkl"),
        fp_cost_labels=os.path.join(DATA_PATH, "cost_label.npy")
    )


def train_cost_estimator():
    from src.cost_learner import CostEstimationTrainer

    cet = CostEstimationTrainer(
        _fp_plan_sequences_output=os.path.join(DATA_PATH, "job-cardinality-sequence.pkl"),
        _fp_cost_labels=os.path.join(DATA_PATH, "cost_label.npy"),
        _fp_std_scalar_labels=MODEL_PATH_STD_SCALER
    )
    cet.flow(
        _fp_model_save_prefix=MODEL_PATH_COST_MODEL_PREFIX
    )


def get_train_data():
    os.makedirs(DATA_PATH_PLANS_FOR_TRAIN, exist_ok=True)

    db = PostgresDB(_config=DB_LAB_VM_CONFIG)

    for _file in os.listdir(DATA_PATH_SQL_FOR_PLANS):
        if _file.endswith(".sql"):
            _base_name = os.path.basename(_file)
            if _base_name in ["fkindexes.sql", "schema.sql"]:
                MainLogger.info(f"file name: {_base_name}, filtered")
                continue
            MainLogger.info(f"file name: {_base_name}")
            with open(os.path.join(DATA_PATH_SQL_FOR_PLANS, _file), "r", encoding="utf-8") as _in_f:
                _plan = db.get_query_plan(_in_f.read())
            if isinstance(_plan, tuple):
                # # 获取成功 写入文件
                with open(os.path.join(DATA_PATH_PLANS_FOR_TRAIN, _base_name[:-4]), "w", encoding="utf-8") as _out_f:
                    _out_f.write("\n".join(tuple(zip(*_plan))[0]))


def hint_generate_and_cost_estimation():
    from src.hint_generator import CostEstimator, HintGenerator

    # # 先获取所有候选的查询计划, 这个是可以的
    hg = HintGenerator(
        _sql_path=DATA_PATH_LC_SQL_TRAIN_CSV
    )
    hg.flow()
    ce = CostEstimator(
        _fp_std_scaler=MODEL_PATH_STD_SCALER,
        cost_model_path=MODEL_PATH_COST_MODEL_PREFIX + "-4999"
    )
    ce.flow(
        plan_dir=os.path.join(DATA_PATH_GOT_PLANS, "0"),
        leaf_model_path=MODEL_PATH_EMBEDDING_MODEL,
        fp_column_statistics=DATA_PATH_LC_COLUMN_MIN_MAX_VALS_CSV,
        fp_vocab_dict=os.path.join(DATA_PATH, "vocab_dict.pkl")
    )


if __name__ == '__main__':
    # get_train_data()
    tree_embedding_encode()
    # train_cost_estimator()
    # hint_generate_and_cost_estimation()

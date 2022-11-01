"""
    stores some configs
"""
import os.path

PROJECTS_ROOT = ""
SRC_PATH = os.path.join(PROJECTS_ROOT, "src")
TEMP_PATH = os.path.join(PROJECTS_ROOT, "tmp")
DATA_PATH = os.path.join(PROJECTS_ROOT, "data")
DATA_PATH_SQL = os.path.join(DATA_PATH, "SQL")
DATA_PATH_LOGS = os.path.join(DATA_PATH, "logs")
DATA_PATH_SQL_WITH_HINTS = os.path.join(DATA_PATH, "SQL_with_hint")
DATA_PATH_SQL_WITH_HINTS_TEST = os.path.join(DATA_PATH_SQL_WITH_HINTS, "0")

DB_CONFIGS = {
    "test": {
        "host": "127.0.0.1",
        "user": "postgres",
        "db": "postgres"
    },
    "db_lab_vm_bing": {
        "host": "192.168.72.128",
        "user": "postgres",
        "db": "imdbload"
    }
}
TEST_DB_CONFIG = DB_CONFIGS["test"]
DB_LAB_VM_CONFIG = DB_CONFIGS["db_lab_vm_bing"]


# # STATEMENT TIMEOUT, PG执行的语句超时时间, 单位是ms
PG_STATEMENT_TIMEOUT = 60000

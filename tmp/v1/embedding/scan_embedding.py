import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Dense, Flatten, SimpleRNN

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import os
import csv

# #
file_name_column_min_max_vals = "../../../data/column_min_max_vals.csv"
# plan_path = "../data/JOB/cardinality/"
# test_path = "../data/JOB/synthetic/"
# test_path = "../data/JOB/job-light/'
# test_path = "../data/JOB/cardinality/"
plan_path = "/data/sunluming/datasets/JOB/cardinality"
test_path = plan_path


def extract_time(line):
    """
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
    return float(start_cost), float(end_cost), float(rows), float(width), float(a_start_cost), float(a_end_cost), float(
        a_rows)


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
        import unicodedata
        unicodedata.numeric(s)
        return False
    except (TypeError, ValueError):
        pass
    return True


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
    min_val = column_min_max_vals[column_name][0]
    max_val = column_min_max_vals[column_name][1]
    val = float(val)
    if (val > max_val):
        val = max_val
    elif (val < min_val):
        val = min_val
    val = float(val)
    val_norm = (val - min_val) / (max_val - min_val)
    return val_norm


def get_data_and_label(path):
    """
        这一段应该就是 本脚本的核心内容了, 用于获取训练数据
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
        path: 是 plan_path 或者 test_plan "./datasets/JOB/cardinality"
            暂时还不清楚是什么东西, 然后又来了一次排序, 每一个应该都是一个查询计划
            每个查询计划包含一个查询树, 预估时间和实际查询时间
            最后两行是执行时间 是不需要的

    Returns:
        sentences
        rows 返回涉及到的真实的行数
        pg 返回查询涉及到的行数
    """
    plans = sorted(os.listdir(path))
    sentences = []
    rows = []
    pg = []
    d = {}
    for file in sorted(plans):
        with open(path + '/' + file, 'r') as f:
            plan = f.readlines()
            for i in range(len(plan) - 2):
                # # 处理每一行查询计划, 发现Seq Scan
                if "Seq Scan" in plan[i]:
                    # # 获取 计划 消耗的时间 和 真实消耗的时间
                    _start_cost, _end_cost, _rows, _width, _a_start_cost_, _a_end_cost, _a_rows = extract_time(plan[i])
                    # # 获取table
                    if len(plan[i].strip().split("  ")) == 2:
                        _sentence = " ".join(plan[i].strip().split("  ")[0].split(" ")[:-1]) + " "
                        table = plan[i].strip().split("  ")[0].split(" ")[4]
                    else:
                        _sentence = " ".join(plan[i].strip().split("  ")[1].split(" ")[:-1]) + " "
                        table = plan[i].strip().split("  ")[1].split(" ")[4]
                    # # 把 Seq Scan的下一行也带上
                    if "actual" not in plan[i + 1] and "Plan" not in plan[i + 1]:
                        _sentence += plan[i + 1].strip()
                    else:
                        _sentence += table
                        # # 复制两份, 看不懂
                        _sentence = _sentence + ' ' + _sentence
                    # # 将sentence中的分隔符去掉 还有 bpchar和 numeric等等修饰, 最后还是把 Seq Scan on 去掉了
                    # # 作者这部分写的真的非常乱, 让人不明所以
                    _sentence = _sentence.replace(": ", " ").replace("(", "").replace(")", "").replace("'", "").replace(
                        "::bpchar", "") \
                        .replace("[]", "").replace(",", " ").replace("\\", "").replace("::numeric", "").replace("  ",
                                                                                                                " ") \
                        .replace("Seq Scan on ", "").strip()
                    # # 对 _sentences 中的每个词都进行处理
                    # # 如果是数字, 进行正则化, 如果不是, 那就直接添加
                    # # 可以改写成 map
                    sentence = []
                    ll = _sentence.split(" ")
                    for cnt in range(len(ll)):
                        if is_not_number(ll[cnt]):
                            sentence.append(ll[cnt])
                        else:
                            try:
                                sentence.append(
                                    normalize_data(ll[cnt], table + '.' + str(ll[cnt - 2]), column_min_max_vals))
                            except:
                                pass
                    sentences.append(tuple(sentence))
                    rows.append(_a_rows)
                    pg.append(_rows)
    return sentences, rows, pg


def prepare_data_and_label(sentences, rows):
    data = []
    label = []
    for sentence, row in zip(sentences, rows):
        _s = []
        for word in sentence:
            if (is_not_number(word)):
                _tmp = np.column_stack((np.array([0]), vocab_dict[word]))
                _tmp = np.reshape(_tmp, (vocab_size + 1))
                assert (len(_tmp) == vocab_size + 1)
                _s.append(_tmp)
            else:
                # print(word)
                _tmp = np.full((vocab_size + 1), word)
                # _tmp = np.column_stack((np.array([float(word)]),np.zeros((1,vocab_size))))
                # _tmp = np.reshape(_tmp,(vocab_size+1))
                # print(_tmp)
                assert (len(_tmp) == vocab_size + 1)
                _s.append(_tmp)
        data.append(np.array(_s))
        label.append(row)
    return data, label


def normalize_labels(labels, min_val=None, max_val=None):
    """
        log transformation without normalize
    Args:
        labels:
        min_val:
        max_val:

    Returns:

    """
    labels = np.array([np.log(float(l)) for l in labels]).astype(np.float32)
    return labels, 0, 1


# # 从文件中获取列的最大最小值, 用于正则化, 可以改写成迭代器形式的写法
with open(file_name_column_min_max_vals, 'r') as f:
    data_raw = list(list(rec) for rec in csv.reader(f, delimiter=','))
    column_min_max_vals = {}
    for i, row in enumerate(data_raw[1:]):
        column_min_max_vals[row[0]] = [float(row[1]), float(row[2])]

# # 获取数据
sentences, rows, pg = get_data_and_label(plan_path)

# # 获取所有的非数字词汇, 这到底是在做什么, 还去了重
vocabulary = []
for sentence in sentences:
    for word in sentence:
        if word not in vocabulary and is_not_number(word):
            vocabulary.append(word)
# print(len(vocabulary))
vocab_size = len(vocabulary)
# print(vocabulary)

_vocabulary = np.array(vocabulary)
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(_vocabulary)
encoded = to_categorical(integer_encoded)
vocab_dict = {}
for v, e in zip(vocabulary, encoded):
    vocab_dict[v] = np.reshape(np.array(e), (1, vocab_size))

data, label = prepare_data_and_label(sentences, rows)
label_norm, min_val, max_val = normalize_labels(label)

max_len = 0
for sentence in sentences:
    if (len(sentence) > max_len):
        max_len = len(sentence)
print(max_len)
padded_sentences = pad_sequences(data, maxlen=max_len, padding='post', dtype='float32')

# print(np.shape(padded_sentences))
# print(np.shape(label_norm))

X_train, X_test, y_train, y_test = train_test_split(padded_sentences, label_norm, test_size=0.8, random_state=40)
pg_train, pg_test, _y_train, _y_test = train_test_split(pg, label_norm, test_size=0.2, random_state=40)
print(np.shape(X_train), np.shape(X_test))
print(np.shape(y_train), np.shape(y_test))

model = Sequential()
model.add(SimpleRNN(128, return_sequences=True, activation='relu', input_shape=(max_len, vocab_size + 1)))
model.add(SimpleRNN(128, return_sequences=True, activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))  # ,kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(64, activation='relu'))  # ,kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(1, activation='relu'))

model.compile(optimizer=optimizers.Adagrad(lr=0.001), loss='mse', metrics=['mse', 'mae'])

model.summary()

model.fit(padded_sentences, label_norm, validation_split=0.2, epochs=100, batch_size=128, shuffle=True)

model.save('../model/embedding_model.h5')
# model.load_weights("../model/embedding_model.h5")

# For validation & feature extraction
test_sentences, test_rows, test_pg = get_data_and_label(test_path)
test_data, test_label = prepare_data_and_label(test_sentences, test_rows)
print(np.shape(test_data), np.shape(test_label))

test_padded_sentences = pad_sequences(test_data, maxlen=max_len, padding='post', dtype='float32')

intermediate_layer_model = Model(inputs=model.input,
                                 outputs=model.layers[4].output)
intermediate_output = intermediate_layer_model.predict(test_padded_sentences)

np.save("../data/{}.npy".format(test_path.split("/")[-2]), intermediate_output)

# For vector demo

scan_label = []
scan_label.append(['table', 'detail'])
for sentence in sentences:
    tmp = []
    #     print(sentence)
    table = sentence[0]
    tmp.append(table)
    if (len(sentence) > 2):
        tmp.append(' '.join(str(each) for each in sentence[2:]))
    else:
        tmp.append(table)
    scan_label.append(tmp)
#     break

np.savetxt("../data/vectors.csv", intermediate_output, delimiter="\t")
np.savetxt("../data/labels.csv", scan_label, fmt='%s', delimiter="\t")

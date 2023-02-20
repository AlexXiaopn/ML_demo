# -*- coding: utf-8 -*-

import numpy as np
import random


# 加载数据
def load_data(iris_path='./Iris.csv', rate=0.8):
    label2vec = {'Iris-setosa': 0., 'Iris-versicolor': 1., 'Iris-virginica': 2.}
    with open(iris_path) as f:
        data = f.readlines()
    dataset = list()
    for sample in data:
        sample = sample.replace('\n', '')
        row = sample.split(',')
        # 根据类别获取浮点数
        label = label2vec[row[-1]]
        row = row[:-1]
        row.append(label)
        dataset.append(row)
    random.shuffle(dataset)
    print(dataset)
    train_data = np.array(dataset[:int(len(dataset) * rate)], dtype=float)
    test_data = np.array(dataset[int(len(dataset) * rate):], dtype=float)
    print("test_date:weidaluan ", test_data)
    return np.rint(train_data), np.rint(test_data)


def get_rela_entropy(dataset, feature: int):
    def get_entropy(dataset):
        label_tags = list(set(dataset[:, -1]))
        label_length = len(dataset[:, -1])
        tmp_entropy = 0
        for label_tag in label_tags:
            tmp = sum([1 for d in dataset if d[-1] == label_tag])
            tmp_entropy += (tmp / label_length) * np.math.log(tmp / label_length, 2)
        entropy = -tmp_entropy
        return entropy

    feature_tags = list(set(dataset[:, feature]))
    sub_entropy = 0
    for feature_tag in feature_tags:
        sub_dataset = [d for d in dataset if d[feature] == feature_tag]
        sub_dataset = np.array(sub_dataset)
        tmp_entropy = get_entropy(sub_dataset)
        sub_entropy += (len(sub_dataset) / len(dataset)) * tmp_entropy
    rela_entropy = get_entropy(dataset) - sub_entropy
    return rela_entropy


def select_feature(dataset, features):
    rela_entropys = list()
    for feature in features:
        feature: int
        rela_entropy = get_rela_entropy(dataset, feature)
        rela_entropys.append(rela_entropy)
    return features[rela_entropys.index(max(rela_entropys))]


# 出现次数最多的数
def major_label(labels):
    # print(labels)
    tags = list(set(labels))
    # print("----------")
    # print(tags)
    # print("----------")
    # 统计每个元素各有多少个
    tag_num = [sum([1 for i in labels if i == label]) for label in tags]
    # print(tag_num)
    # 出现最多的元素 下标 k
    k = tag_num.index(max(tag_num))
    # 返回该元素值
    return tags[k]


# 构建决策树
def build_tree(dataset, features) -> dict:
    labels = dataset[:, -1]
    # 不重复集合长度为1时
    if len(set(labels)) == 1:
        return {'label': labels[0]}
    if not len(features):
        return {'label': major_label(labels)}
    best_feature = select_feature(dataset, features)
    tree = {'feature': best_feature, 'children': {}}
    feature_tags = list(set(dataset[:, best_feature]))
    for feature_tag in feature_tags:
        sub_dataset = [d for d in dataset if d[best_feature] == feature_tag]
        sub_dataset = np.array(sub_dataset)
        if len(sub_dataset) == 0:
            tree['children'][feature_tag] = {'label': major_label(labels)}
        else:
            sub_features = [i for i in features if i != best_feature]
            tree['children'][feature_tag] = build_tree(sub_dataset, sub_features)
    return tree


# 分类
def classifier(tree: dict, features_data, default):
    def classify(tree: dict, sample):
        for k, v in tree.items():
            if k != 'feature':
                return tree['label']
            else:
                return classify(tree['children'][sample[tree['feature']]],
                                sample)

    predict_vec = list()
    for features_sample in features_data:
        try:
            predict = classify(tree, features_sample)
        except KeyError:
            predict = default
        predict_vec.append(predict)
    return predict_vec


if __name__ == "__main__":
    train_data, test_data = load_data()
    # print(train_data)
    # print(train_data.shape[1]-1)
    # shape[1]-1 二维数组列数减一 shape (行数,列数)
    tree = build_tree(train_data, list(range(train_data.shape[1] - 1)))
    # print(tree)
    # [:,-1] 取倒数第一列 一维数组
    print("test_data: ----------------------")
    print(test_data)
    test_data_labels = test_data[:, -1]
    # print("--------------------")
    # print(test_data_labels)
    # print("--------------------")
    # [:,:-1]去掉最后一列的二位数组
    test_data_features = test_data[:, :-1]
    # print("--------------------")
    # print(test_data_features)
    # print("--------------------")
    default = major_label(test_data_labels)
    # print("default----", default)
    predict_vec = classifier(tree, test_data_features, default)
    #    print(predict_vec)
    accuracy = np.mean(np.array(predict_vec == test_data_labels))
    print(accuracy)

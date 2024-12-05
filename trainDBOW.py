import numpy as np
from sklearn.cluster import KMeans
import yaml

import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import yaml
from math import log


def load_features_from_npy(npy_path):
    """加载 .npy 文件中的特征"""
    features = np.load(npy_path)
    return features


from joblib import parallel_backend

import numpy as np
from sklearn.cluster import KMeans
#
# def recursive_kmeans(features, k, level, max_level, node_id=1, parent_id=0):
#     nodes = []
#
#     # 打印当前层的特征数量和层级信息
#     print(f"Level {level}: Features shape {features.shape}")
#
#
#     kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
#     kmeans.fit(features)
#
#     centroids = kmeans.cluster_centers_  # 质心
#     labels = kmeans.labels_
#
#     print(f"Level {level}: Created {k} clusters, centroids shape {centroids.shape}")
#
#     # 如果当前层级达到 max_level - 1，则创建叶子节点
#     if level == max_level - 1:
#         print(f"Level {level}: Creating leaf nodes...")
#         for i in range(k):
#             sub_features = features[labels == i]
#             # 使用质心作为描述子
#             descriptor = " ".join(f"{v:.7f}" for v in centroids[i])
#             node = {
#                 'nodeId': node_id,
#                 'parentId': parent_id,
#                 'weight': 0.0,
#                 'descriptor': f'dbw3 5 {features.shape[1]} {descriptor}',
#                 'is_leaf': True  # 设置为叶子节点
#             }
#             nodes.append(node)
#             print(f"   Processing level {level}, creating leaf node for cluster {i + 1}")
#             node_id += 1
#         return nodes, node_id
#
#     # 如果当前层级小于 max_level - 1，则继续递归
#     print(f"Level {level}: Creating non-leaf nodes...")
#
#     # 在当前层级中为每个节点添加一个计数器
#     cluster_counter = 1  # 记录当前层级内的节点顺序
#
#     for i in range(k):
#         sub_features = features[labels == i]
#         # 输出当前处理的是哪一层的哪一组特征
#         print(f"   Processing level {level}, branch {cluster_counter}")
#
#         # 创建当前层的节点，使用聚类的质心作为描述子
#         node = {
#             'nodeId': node_id,
#             'parentId': parent_id,
#             'weight': 0.0,
#             'descriptor': f'dbw3 5 {features.shape[1]} ' + " ".join(f"{v:.7f}" for v in centroids[i]),
#             'is_leaf': False  # 当前节点不是叶子节点
#         }
#         nodes.append(node)
#
#         # 打印当前层级的分支顺序
#         print(f"   Processing branch {cluster_counter} at level {level}, cluster {i + 1}")
#
#         node_id += 1
#         cluster_counter += 1  # 增加当前层级的分支编号
#
#         # 递归创建子节点
#         sub_nodes, node_id = recursive_kmeans(sub_features, k, level + 1, max_level, node_id, node['nodeId'])
#         nodes.extend(sub_nodes)
#
#     return nodes, node_id


def l2_normalize(vectors):
    """对输入的特征进行 L2 范数归一化"""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)  # 计算每行的 L2 范数
    return vectors / (norms + 1e-8)  # 除以 L2 范数进行归一化，避免除以零




def recursive_kmeans(features, k, level, max_level, node_id=1, parent_id=0):
    nodes = []

    # 打印当前层的特征数量和层级信息
    print(f"Level {level}: Features shape {features.shape}")

    # 使用 KMeans++ 初始化方法
    kmeans = KMeans(
        n_clusters=k,  # 假设聚类数为 k
        init='k-means++',  # 使用 k-means++ 初始化
        n_init=10,  # 尝试 10 次初始化
        max_iter=300,  # 设置最大迭代次数为 300
        tol=1e-4,  # 设置收敛容忍度
        random_state=42,  # 固定随机种子
        verbose=1,  # 输出详细日志
        algorithm='elkan'  # 使用 Elkan 算法加速收敛
    )

    kmeans.fit(features)

    centroids = kmeans.cluster_centers_  # 质心
    labels = kmeans.labels_

    print(f"Level {level}: Created {k} clusters, centroids shape {centroids.shape}")

    # 如果当前层级达到 max_level - 1，则创建叶子节点
    if level == max_level - 1:
        print(f"Level {level}: Creating leaf nodes...")
        for i in range(k):
            # 获取当前簇的质心
            cluster_center = centroids[i]

            # 使用质心作为描述子
            descriptor = " ".join(f"{v:.7f}" for v in cluster_center)
            node = {
                'nodeId': node_id,
                'parentId': parent_id,
                'weight': 0.0,
                'descriptor': f'dbw3 5 {features.shape[1]} {descriptor}',
                'is_leaf': True  # 设置为叶子节点
            }
            nodes.append(node)
            print(f"   Processing level {level}, creating leaf node for cluster {i + 1}")
            node_id += 1
        return nodes, node_id

    # 如果当前层级小于 max_level - 1，则继续递归
    print(f"Level {level}: Creating non-leaf nodes...")

    # 在当前层级中为每个节点添加一个计数器
    cluster_counter = 1  # 记录当前层级内的节点顺序

    for i in range(k):
        sub_features = features[labels == i]
        # 输出当前处理的是哪一层的哪一组特征
        print(f"   Processing level {level}, branch {cluster_counter}")

        # 创建当前层的节点，使用聚类的质心作为描述子
        cluster_center = centroids[i]  # 聚类的质心
        descriptor = " ".join(f"{v:.7f}" for v in cluster_center)

        node = {
            'nodeId': node_id,
            'parentId': parent_id,
            'weight': 0.0,
            'descriptor': f'dbw3 5 {features.shape[1]} {descriptor}',
            'is_leaf': False  # 当前节点不是叶子节点
        }
        nodes.append(node)

        # 打印当前层级的分支顺序
        print(f"   Processing branch {cluster_counter} at level {level}, cluster {i + 1}")

        node_id += 1
        cluster_counter += 1  # 增加当前层级的分支编号

        # 递归创建子节点
        sub_nodes, node_id = recursive_kmeans(sub_features, k, level + 1, max_level, node_id, node['nodeId'])
        nodes.extend(sub_nodes)

    return nodes, node_id




def generate_word_id_map(nodes):
    """
    生成wordId和nodeId的映射
    """
    word_id_map = []
    word_id = 0

    for node in nodes:
        if node.get('is_leaf', False):  # 检查是否是叶子节点
            word_id_map.append({
                'wordId': word_id,
                'nodeId': node['nodeId']
            })
            word_id += 1

    # 如果没有叶子节点，提醒
    if not word_id_map:
        print("Warning: No leaf nodes found. Check your clustering logic.")

    return word_id_map


def save_to_dbow3_yml(vocab, word_id_map, yml_path, k, L):
    with open(yml_path, 'w') as f:
        f.write("%YAML:1.0\n---\n")
        f.write("vocabulary:\n")
        f.write(f"   k: {k}\n")
        f.write(f"   L: {L}\n")
        f.write("   scoringType: 0\n")
        f.write("   weightingType: 0\n")
        f.write("   nodes:\n")
        for node in vocab:
            f.write(f"      - {{ nodeId:{node['nodeId']}, parentId:{node['parentId']}, weight:{node['weight']:.1f},\n")
            f.write(f"          descriptor:\"{node['descriptor']}\" }}\n")
        f.write("   words:\n")
        for mapping in word_id_map:
            f.write(f"      - {{ wordId:{mapping['wordId']}, nodeId:{mapping['nodeId']} }}\n")
    print(f"Vocabulary saved to {yml_path} in DBoW3-compatible format")



def main():
    npy_path = "COCOall_descriptors.npy"  # 特征文件路径
    yml_path = "vocabulary2.yml"  # 词汇文件输出路径

    k = 10  # 每个节点的分支数
    L = 5  # 层级

    # 加载特征
    features = load_features_from_npy(npy_path)
    print(f"Number of features loaded: {features.shape[0]}, Descriptor dimension: {features.shape[1]}")

    # 执行层级聚类
    print("Training vocabulary...")
    vocab, _ = recursive_kmeans(features, k, 0, L)
    print(f"Vocabulary created with {len(vocab)} nodes.")

    # 生成 wordId 和 nodeId 的映射
    word_id_map = generate_word_id_map(vocab)

    # 保存词典和映射
    save_to_dbow3_yml(vocab, word_id_map, yml_path, k, L)
    print("Finished!")


if __name__ == "__main__":
    main()

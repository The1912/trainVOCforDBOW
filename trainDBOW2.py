import numpy as np
import os
from sklearn.cluster import KMeans
import yaml
from math import log


def load_all_features_from_folder(folder_path):
    """加载文件夹中所有的 .npy 文件，并合并成一个字典，去除 keypoints"""
    combined_hierarchical_descriptors = {}

    # 遍历文件夹中的所有 .npy 文件
    for npy_file in os.listdir(folder_path):
        if npy_file.endswith('.npy'):
            file_path = os.path.join(folder_path, npy_file)
            # 加载当前文件的数据
            hierarchical_descriptors = np.load(file_path, allow_pickle=True).item()

            # 只保留 descriptors，删除 keypoints
            for image_name, data in hierarchical_descriptors.items():
                combined_hierarchical_descriptors[image_name] = {
                    'descriptors': data['descriptors']
                }

    return combined_hierarchical_descriptors


def recursive_kmeans(features, k, level, max_level, image_feature_map, node_id=1, parent_id=0):
    nodes = []

    # 输出当前层级和簇数
    print(f"Training level {level}, with {k} clusters...")

    # 使用 KMeans++ 初始化方法
    kmeans = KMeans(
        n_clusters=k,
        init='k-means++',
        n_init=10,
        max_iter=300,
        tol=1e-4,
        random_state=42,
        verbose=0,
        algorithm='elkan'
    )

    kmeans.fit(features)

    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    # 如果当前层级达到 max_level - 1，则创建叶子节点
    if level == max_level - 1:
        for i in range(k):
            cluster_center = centroids[i]
            descriptor = " ".join(f"{v:.7f}" for v in cluster_center)
            node = {
                'nodeId': node_id,
                'parentId': parent_id,
                'weight': 0.0,
                'descriptor': f'dbw3 5 {features.shape[1]} {descriptor}',
                'is_leaf': True,
                'included_images': []  # 每个叶子节点包含的图像
            }
            # 存储哪些图像属于该叶子节点
            for img_idx, label in enumerate(labels):
                if label == i:  # 如果该特征属于该簇
                    node['included_images'].append(image_feature_map[img_idx])

            # 输出节点的分叉信息
            # print(f"Level {level} - Fork {parent_id}-{node_id}: Cluster {i+1}/{k}")

            nodes.append(node)
            node_id += 1
        return nodes, node_id

    # 如果当前层级小于 max_level - 1，则继续递归
    for i in range(k):
        sub_features = features[labels == i]

        # 更新 image_feature_map 为当前层的子集图像对应的特征映射
        sub_image_feature_map = [image_feature_map[img_idx] for img_idx, label in enumerate(labels) if label == i]

        node = {
            'nodeId': node_id,
            'parentId': parent_id,
            'weight': 0.0,
            'descriptor': f'dbw3 5 {features.shape[1]} ' + " ".join(f"{v:.7f}" for v in centroids[i]),  # 保证格式一致
            'is_leaf': False,
            'included_images': []  # 当前非叶子节点包含的图像
        }

        # 存储哪些图像属于当前节点
        for img_idx, label in enumerate(labels):
            if label == i:  # 如果该特征属于该簇
                node['included_images'].append(image_feature_map[img_idx])

        # 输出节点的分叉信息
        # print(f"Level {level} - Fork {parent_id}-{node_id}: Cluster {i+1}/{k}")

        nodes.append(node)
        node_id += 1

        # 递归时使用更新后的 sub_features 和 sub_image_feature_map
        sub_nodes, node_id = recursive_kmeans(sub_features, k, level + 1, max_level, sub_image_feature_map, node_id,
                                              node['nodeId'])
        nodes.extend(sub_nodes)

    return nodes, node_id




def generate_word_id_map(nodes):
    word_id_map = []
    word_id = 0

    for node in nodes:
        if node.get('is_leaf', False):
            word_id_map.append({
                'wordId': word_id,
                'nodeId': node['nodeId']
            })
            word_id += 1

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
            f.write(f"      - {{ nodeId:{node['nodeId']}, parentId:{node['parentId']}, weight:{node['weight']:.10f},\n")
            f.write(f"          descriptor:\"{node['descriptor']}\" }}\n")
        f.write("   words:\n")
        for mapping in word_id_map:
            f.write(f"      - {{ wordId:{mapping['wordId']}, nodeId:{mapping['nodeId']} }}\n")
    print(f"Vocabulary saved to {yml_path} in DBoW3-compatible format")

def compute_single_node_tf(node_idx, image_feature_counts, nodes, image_names):
    """
    计算单个节点的 TF 值。
    :param node_idx: 要计算 TF 的节点的索引。
    :param image_feature_counts_dict: 包含所有图像特征数量的字典，键为图像名，值为特征数量。
    :param nodes: 所有的节点信息（包含每个节点包含的图像列表）。
    :param image_names: 图像名的唯一列表。
    :return: 返回节点在每张图像中的 TF 值。
    """
    # 获取当前节点
    node = nodes[node_idx]

    # 初始化 TF 值
    tf_values = []

    # 遍历所有图像并统计每张图像该节点的出现次数
    for image_name in image_names:
        # 获取该图像的特征数量
        num_features_in_image = image_feature_counts.get(image_name, 0)  # 从 image_feature_counts_dict 获取特征数量

        # 统计当前图像中该节点的出现次数
        tf_count = node['included_images'].count(image_name)

        # 归一化计算 TF 值
        if num_features_in_image > 0:
            tf = tf_count / num_features_in_image  # 计算该节点的 TF 值
        else:
            tf = 0.0
        tf_values.append(tf)

    return np.array(tf_values)


import math
def compute_single_node_idf(node_idx, nodes, total_images):
    """
    计算单个节点的 IDF 值。
    :param node_idx: 要计算 IDF 的节点的索引。
    :param hierarchical_descriptors: 包含所有图像及其描述符的字典。
    :param nodes: 所有的节点信息（包含每个节点包含的图像列表）。
    :param total_images: 图像总数量。
    :return: 返回单个节点的 IDF 值。
    """
    # 获取当前节点
    node = nodes[node_idx]

    # 获取当前节点中去重后的图像列表
    unique_images = set(node['included_images'])
    df = len(unique_images)

    # 计算 IDF 值
    if df == 0:
        idf = 0  # 如果节点没有出现过任何图像，IDF 设为 0
    else:
        idf = np.log((total_images + 1) / (df + 1))  # 加 1 进行平滑处理

    # 如果 IDF 小于 0，设置为 0 或其他小值
    if idf < 0:
        idf = 0  # 或者你也可以选择设置为一个小的常数

    return idf

from tqdm import tqdm  # 导入 tqdm 库

def compute_all_nodes_weights(image_feature_counts, nodes, image_names, total_images, method="sum"):
    """
    计算所有节点的权重，并更新节点的 weight，添加 TF 和 IDF 字段。
    :param hierarchical_descriptors: 包含所有图像及其描述符的字典。
    :param nodes: 所有的节点信息（包含每个节点包含的图像列表）。
    :param image_names: 所有图像的文件名列表。
    :param total_images: 总图像数量。
    :param method: 汇总方法，可以是 "sum" 或 "average"。
    """
    # 使用 tqdm 包装 nodes 列表，显示进度条
    for node_idx in tqdm(range(len(nodes)), desc="Calculating node weights", unit="node"):
        # 获取该节点的 IDF 和 TF，并计算其 TF-IDF 权重
        idf = compute_single_node_idf(node_idx, nodes, total_images)
        tf_values = compute_single_node_tf(node_idx, image_feature_counts, nodes, image_names)

        # 计算该节点的 TF-IDF 权重
        tf_idf_values = tf_values * idf  # 按逐元素方式相乘

        # 汇总 TF-IDF 值
        if method == "sum":
            node_weight = np.sum(tf_idf_values)  # 总和
        elif method == "average":
            node_weight = np.mean(tf_idf_values)  # 平均值

        # 更新节点的 weight 字段，并添加 TF 和 IDF 字段
        nodes[node_idx]['weight'] = node_weight
        nodes[node_idx]['TF'] = tf_values  # 添加 TF 字段
        nodes[node_idx]['IDF'] = idf  # 添加 IDF 字段

    return nodes


import gc
def main():
    # preprocessdata_path = "DBOWPreprocessdata"  # 输入文件夹路径
    # yml_path = "vocabulary.yml"  # 输出文件路径
    #
    # all_descriptors = np.load(os.path.join(preprocessdata_path, 'all_descriptors.npy'))
    # image_feature_map = np.load(os.path.join(preprocessdata_path, 'image_feature_map.npy'))
    # image_names = np.load(os.path.join(preprocessdata_path, 'image_names.npy'))
    # image_feature_counts = np.load(os.path.join(preprocessdata_path, 'image_feature_counts.npy'), allow_pickle=True).item()
    #
    # total_images = len(image_names)
    #
    # print(f"Loaded {len(image_names)} images for training.")
    # print(f"Total descriptors: {all_descriptors.shape[0]}")
    # print(f"Descriptor size: {all_descriptors.shape[1]}")
    #
    # print("Length of all_descriptors:", len(all_descriptors))
    # print("Length of image_feature_map:", len(image_feature_map))
    # print("Length of image_names:", len(image_names))
    # print("Length of image_feature_counts:", len(image_feature_counts))

    folder_path = "DBOWdescriptors2"  # 输入文件夹路径
    yml_path = "vocabulary.yml"  # 输出文件路径

    k = 10  # 每个节点的分支数
    L = 5 # 层级


    # 加载特征
    hierarchical_descriptors = load_all_features_from_folder(folder_path)
    print(f"Loaded {len(hierarchical_descriptors)} images.")

    print("Begin preprocess...")

    total_images = len(hierarchical_descriptors)
    image_feature_map = []
    image_names = []
    all_descriptors = []
    image_feature_counts = {}

    # Step 3: 遍历每张图像并提取其描述符
    for image_name, data in hierarchical_descriptors.items():
        descriptors = data['descriptors']  # 获取该图像的描述符
        all_descriptors.append(descriptors)  # 将描述符添加到 all_descriptors 列表中

        # 对每个描述符，记录其所属的图像
        num_features = len(descriptors)
        image_feature_map.extend([image_name] * num_features)  # 将图像名重复 num_features 次
        image_names.append(image_name)

    print(f"Length of image_feature_map: {len(image_feature_map)}")
    print(f"Length of image_names: {len(image_names)}")


    for image_name, data in hierarchical_descriptors.items():
        # 获取该图像的描述符
        descriptors = data['descriptors']
        # 计算描述符的数量
        num_features = len(descriptors)
        # 将该图像的特征数量保存到字典中
        image_feature_counts[image_name] = num_features
    print(f"Length of image_feature_counts: {len(image_feature_counts)}")

    # Step 4: 将 all_descriptors 合并成一个大的 NumPy 数组
    all_descriptors = np.vstack(all_descriptors)
    print(f"Total number of descriptors: {all_descriptors.shape[0]}")  # 总特征点数
    print(f"Descriptor size: {all_descriptors.shape[1]}")  # 描述符维度

    hierarchical_descriptors.clear()
    hierarchical_descriptors = None  # 或 del hierarchical_descriptors

    gc.collect()
    print("Hierarchical_descriptors dict been cleared")

    from pympler import asizeof

    print("Detailed memory usage of image_feature_map:", asizeof.asizeof(image_feature_map))
    print("Detailed memory usage of image_names:", asizeof.asizeof(image_names))
    print("Detailed memory usage of all_descriptors:", asizeof.asizeof(all_descriptors))
    print("Detailed memory usage of image_feature_counts:", asizeof.asizeof(image_feature_counts))
    print("Detailed memory usage of hierarchical_descriptors:", asizeof.asizeof(hierarchical_descriptors))

    print("End preprocess.")

    print("Training vocabulary...")

    nodes, _ = recursive_kmeans(all_descriptors, k, 0, L, image_feature_map)
    print(f"Vocabulary created with {len(nodes)} nodes.")

    print("Caculate TF-IDF weights...")
    nodes = compute_all_nodes_weights(image_feature_counts, nodes, image_names, total_images, method="sum")
    print("TF-IDF weights finished")

    # 生成 wordId 和 nodeId 的映射
    word_id_map = generate_word_id_map(nodes)

    # 保存词典和映射
    save_to_dbow3_yml(nodes, word_id_map, yml_path, k, L)
    print("Train Finished!")




if __name__ == "__main__":
    main()

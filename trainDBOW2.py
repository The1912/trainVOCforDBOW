import numpy as np
import os
from sklearn.cluster import KMeans
import yaml
from math import log
from tqdm import tqdm


def load_all_features_from_folder(folder_path):
    """Load all .npy files in the folder and merge them into a dictionary, removing keypoints"""
    combined_hierarchical_descriptors = {}

    # Traverse all .npy files in the folder
    for npy_file in os.listdir(folder_path):
        if npy_file.endswith('.npy'):
            file_path = os.path.join(folder_path, npy_file)
            # Load the data from the current file
            hierarchical_descriptors = np.load(file_path, allow_pickle=True).item()

            # Retain only descriptors and remove keypoints
            for image_name, data in hierarchical_descriptors.items():
                combined_hierarchical_descriptors[image_name] = {
                    'descriptors': data['descriptors']
                }

    return combined_hierarchical_descriptors


def recursive_kmeans(features, k, level, max_level, image_feature_map, node_id=1, parent_id=0):
    nodes = []

    # Output current level and number of clusters
    print(f"Training level {level}, with {k} clusters...")

    # Use KMeans++ initialization method
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

    # If current level reaches max_level - 1, create leaf nodes
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
                'included_images': []  # Images included in each leaf node
            }
            # Store which images belong to this leaf node
            for img_idx, label in enumerate(labels):
                if label == i:  # If the feature belongs to this cluster
                    node['included_images'].append(image_feature_map[img_idx])

            # Output the forking information of the node
            # print(f"Level {level} - Fork {parent_id}-{node_id}: Cluster {i+1}/{k}")

            nodes.append(node)
            node_id += 1
        return nodes, node_id

    # If current level is less than max_level - 1, continue recursively
    for i in range(k):
        sub_features = features[labels == i]

        # Update image_feature_map to map the features of the subset images
        sub_image_feature_map = [image_feature_map[img_idx] for img_idx, label in enumerate(labels) if label == i]

        node = {
            'nodeId': node_id,
            'parentId': parent_id,
            'weight': 0.0,
            'descriptor': f'dbw3 5 {features.shape[1]} ' + " ".join(f"{v:.7f}" for v in centroids[i]),  # Ensure format consistency
            'is_leaf': False,
            'included_images': []  # Images included in the current non-leaf node
        }

        # Store which images belong to the current node
        for img_idx, label in enumerate(labels):
            if label == i:  # If the feature belongs to this cluster
                node['included_images'].append(image_feature_map[img_idx])

        # Output the forking information of the node
        # print(f"Level {level} - Fork {parent_id}-{node_id}: Cluster {i+1}/{k}")

        nodes.append(node)
        node_id += 1

        # Recursively use updated sub_features and sub_image_feature_map
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
    Compute the TF value for a single node.
    :param node_idx: The index of the node for which to compute the TF.
    :param image_feature_counts_dict: A dictionary containing the number of features for each image, key is image name, value is feature count.
    :param nodes: All node information (contains a list of images for each node).
    :param image_names: A list of unique image names.
    :return: Return the TF value of the node in each image.
    """
    # Get the current node
    node = nodes[node_idx]

    # Initialize TF values
    tf_values = []

    # Traverse all images and count the occurrences of this node in each image
    for image_name in image_names:
        # Get the number of features in this image
        num_features_in_image = image_feature_counts.get(image_name, 0)  # Get feature count from image_feature_counts_dict

        # Count the occurrences of this node in the current image
        tf_count = node['included_images'].count(image_name)

        # Normalize and compute the TF value
        if num_features_in_image > 0:
            tf = tf_count / num_features_in_image  # Compute TF for this node
        else:
            tf = 0.0
        tf_values.append(tf)

    return np.array(tf_values)


import math
def compute_single_node_idf(node_idx, nodes, total_images):
    """
    Compute the IDF value for a single node.
    :param node_idx: The index of the node for which to compute the IDF.
    :param hierarchical_descriptors: A dictionary containing all images and their descriptors.
    :param nodes: All node information (contains a list of images for each node).
    :param total_images: Total number of images.
    :return: Return the IDF value of the node.
    """
    # Get the current node
    node = nodes[node_idx]

    # Get the list of unique images in this node
    unique_images = set(node['included_images'])
    df = len(unique_images)

    # Compute IDF value
    if df == 0:
        idf = 0  # If no images contain this node, set IDF to 0
    else:
        idf = np.log((total_images + 1) / (df + 1))  # Smooth with +1

    # If IDF is less than 0, set it to 0 or another small value
    if idf < 0:
        idf = 0  # Or you can set it to a small constant

    return idf

def compute_all_nodes_weights(image_feature_counts, nodes, image_names, total_images, method="sum"):
    """
    Compute the weights for all nodes and update the node's weight, adding TF and IDF fields.
    :param hierarchical_descriptors: A dictionary containing all images and their descriptors.
    :param nodes: All node information (contains a list of images for each node).
    :param image_names: A list of all image filenames.
    :param total_images: Total number of images.
    :param method: Summing method, can be "sum" or "average".
    """
    # Use tqdm to wrap nodes list and show progress bar
    for node_idx in tqdm(range(len(nodes)), desc="Calculating node weights", unit="node"):
        # Get the IDF and TF for this node and compute its TF-IDF weight
        idf = compute_single_node_idf(node_idx, nodes, total_images)
        tf = compute_single_node_tf(node_idx, image_feature_counts, nodes, image_names)

        # Calculate weight based on the method
        if method == "sum":
            node_weight = np.sum(tf * idf)
        elif method == "average":
            node_weight = np.mean(tf * idf)
        else:
            node_weight = 0.0  # If the method is invalid, return 0.0

        # Update node's weight
        nodes[node_idx]['weight'] = node_weight

    print(f"Finished calculating weights for {len(nodes)} nodes.")


def main():
    # Example paths and parameters
    folder_path = '/path/to/features'  # Folder containing .npy files
    output_yml = 'vocabulary.yml'
    k = 10  # Number of clusters per level
    max_level = 5  # Maximum depth of the tree

    # Load all features
    combined_hierarchical_descriptors = load_all_features_from_folder(folder_path)

    # Prepare the image names and feature map
    image_names = list(combined_hierarchical_descriptors.keys())
    image_feature_map = image_names  # In this case, just the list of image names
    
    # Assuming all features from all images are stacked into a single array for clustering
    all_features = np.vstack([data['descriptors'] for data in combined_hierarchical_descriptors.values()])

    # Perform KMeans clustering recursively
    nodes, _ = recursive_kmeans(all_features, k, 0, max_level, image_feature_map)

    # Generate word ID map for leaf nodes
    word_id_map = generate_word_id_map(nodes)

    # Save the vocabulary to YAML in DBoW3 format
    save_to_dbow3_yml(nodes, word_id_map, output_yml, k, max_level)

    # Example: Compute node weights based on TF-IDF
    image_feature_counts = {image_name: len(descriptor['descriptors']) for image_name, descriptor in combined_hierarchical_descriptors.items()}
    total_images = len(image_names)
    compute_all_nodes_weights(image_feature_counts, nodes, image_names, total_images, method="sum")

if __name__ == "__main__":
    main()

import numpy as np
import os
import gc

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

def preprocess_and_save_features(folder_path, save_path):
    hierarchical_descriptors = load_all_features_from_folder(folder_path)
    print(f"Loaded {len(hierarchical_descriptors)} images.")

    total_images = len(hierarchical_descriptors)
    image_feature_map = []
    image_names = []
    all_descriptors = []
    image_feature_counts = {}

    for image_name, data in hierarchical_descriptors.items():
        descriptors = data['descriptors']  # 获取该图像的描述符
        all_descriptors.append(descriptors)  # 将描述符添加到 all_descriptors 列表中

        num_features = len(descriptors)
        image_feature_map.extend([image_name] * num_features)  # 将图像名重复 num_features 次
        image_names.append(image_name)

        image_feature_counts[image_name] = num_features  # 记录该图像的特征数量

    # Step 4: 将 all_descriptors 合并成一个大的 NumPy 数组
    all_descriptors = np.vstack(all_descriptors)
    print(f"Total number of descriptors: {all_descriptors.shape[0]}")  # 总特征点数
    print(f"Descriptor size: {all_descriptors.shape[1]}")  # 描述符维度

    # 保存数据到 .npy 文件
    np.save(os.path.join(save_path, 'all_descriptors.npy'), all_descriptors)
    np.save(os.path.join(save_path, 'image_feature_map.npy'), np.array(image_feature_map))
    np.save(os.path.join(save_path, 'image_names.npy'), np.array(image_names))
    np.save(os.path.join(save_path, 'image_feature_counts.npy'), image_feature_counts)

    # 清理内存
    hierarchical_descriptors.clear()
    hierarchical_descriptors = None  # 或 del hierarchical_descriptors
    gc.collect()
    print("Preprocessing complete and features saved.")

def main():
    folder_path = "DBOWdescriptors"  # 输入文件夹路径
    save_path = "DBOWPreprocessdata"

    preprocess_and_save_features(folder_path, save_path)



    print("End preprocess.")




if __name__ == "__main__":
    main()


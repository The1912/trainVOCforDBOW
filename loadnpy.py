import numpy as np

# Load the hierarchical descriptors
hierarchical_descriptors = np.load('/home/dragonz/ADaryl/Codes/Python/superpoint22222/pytorch-superpoint22222222/DBOWdescriptors/hierarchical_descriptors_batch_6.npy', allow_pickle=True).item()

# data = np.load('/home/dragonz/ADaryl/Codes/Python/superpoint22222/pytorch-superpoint22222222/DBOWdescriptors/hierarchical_descriptors_batch_1.npy', allow_pickle=True)
# print(type(data))  # 查看加载的对象类型
# print(data.shape)  # 如果是数组，打印其形状

for image_name, data in hierarchical_descriptors.items():
    keypoints = data['keypoints']
    descriptors = data['descriptors']

    # 输出特征点数量和描述符尺寸
    num_keypoints = keypoints.shape[0]  # 特征点数量
    descriptor_size = [descriptors.shape[0], descriptors.shape[1]]  # 描述符的维度

    print(f"Image: {image_name}")
    print(f"Number of keypoints: {num_keypoints}")
    print(f"Descriptor size: {descriptor_size}")
    print("-" * 30)

print(f"Number of images in the dictionary: {len(hierarchical_descriptors)}")


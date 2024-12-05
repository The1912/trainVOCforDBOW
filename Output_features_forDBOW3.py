
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, zeros_
from models.unet_parts import *
import numpy as np
from torch import nn
import os
import cv2

def depth_to_space(input: torch.Tensor, block_size: int) -> torch.Tensor:
    output = input.permute(0, 2, 3, 1)
    (batch_size, d_height, d_width, d_depth) = output.size()
    s_depth = d_depth // (block_size * block_size)
    s_width = d_width * block_size
    s_height = d_height * block_size

    t_1 = output.reshape(batch_size, d_height, d_width, block_size * block_size, s_depth)
    spl = t_1.split(block_size, 3)
    stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
    output = torch.stack(stack, 0).transpose(0, 1).permute(0, 2, 1, 3, 4).reshape(batch_size, s_height, s_width,
                                                                                  s_depth)
    output = output.permute(0, 3, 1, 2)
    return output


def flattenDetection_out(semi: torch.Tensor) -> torch.Tensor:
    '''
    Flatten detection output

    :param semi:
        output from detector head
        tensor (batch_size, 65, Hc, Wc)

    :return:
        3D heatmap
        tensor (batch_size, H, W)
    '''

    # 应用 softmax 函数
    dense = nn.functional.softmax(semi, dim=1)  # [batch_size, 65, Hc, Wc]

    # 移除 dustbin 类
    nodust = dense[:, :-1, :, :]  # [batch_size, 64, Hc, Wc]

    # 使用 DepthToSpace

    heatmap = depth_to_space(nodust, 8)  # [batch_size, C, H, W]

    return heatmap


# from models.SubpixelNet import SubpixelNet
class SuperPointNet_gauss22(torch.nn.Module):
    """ Pytorch definition of SuperPoint Network. """

    def __init__(self, subpixel_channel=1):
        super(SuperPointNet_gauss22, self).__init__()
        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        det_h = 65
        self.inc = inconv(1, c1)
        self.down1 = down(c1, c2)
        self.down2 = down(c2, c3)
        self.down3 = down(c3, c4)
        # self.down4 = down(c4, 512)
        # self.up1 = up(c4+c3, c2)
        # self.up2 = up(c2+c2, c1)
        # self.up3 = up(c1+c1, c1)
        # self.outc = outconv(c1, subpixel_channel)
        self.relu = torch.nn.ReLU(inplace=True)
        # self.outc = outconv(64, n_classes)
        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnPa = nn.BatchNorm2d(c5)
        self.convPb = torch.nn.Conv2d(c5, det_h, kernel_size=1, stride=1, padding=0)
        self.bnPb = nn.BatchNorm2d(det_h)
        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.bnDa = nn.BatchNorm2d(c5)
        self.convDb = torch.nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)
        self.bnDb = nn.BatchNorm2d(d1)
        self.output = None

    def forward(self, x):
        """ Forward pass that jointly computes unprocessed point and descriptor
        tensors.
        Input
          x: Image pytorch tensor shaped N x 1 x patch_size x patch_size.
        Output
          semi: Output point pytorch tensor shaped N x 65 x H/8 x W/8.
          desc: Output descriptor pytorch tensor shaped N x 256 x H/8 x W/8.
        """
        # Let's stick to this version: first BN, then relu
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Detector Head.
        cPa = self.relu(self.bnPa(self.convPa(x4)))
        semi = self.bnPb(self.convPb(cPa))
        # Descriptor Head.
        cDa = self.relu(self.bnDa(self.convDa(x4)))
        desc = self.bnDb(self.convDb(cDa))

        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

        output = {'semi': semi, 'desc': desc}

        confidence_map = flattenDetection_out(output['semi'])

        resized_desc = F.interpolate(desc, size=confidence_map.shape[2:], mode='bicubic', align_corners=True)



        dn = torch.norm(resized_desc, p=2, dim=1, keepdim=True)  # 计算范数
        fused_descriptor = resized_desc.div(dn + 1e-8)  # 除以范数进行归一化，避免除以零

        return confidence_map, fused_descriptor


def nms_and_filter(keypoints, descriptors, confidence_map, nms_radius=5, border_margin=8):
    """
    对特征点进行非极大值抑制和边界点过滤
    :param keypoints: 原始特征点坐标
    :param descriptors: 原始特征点描述符
    :param confidence_map: 置信度图
    :param nms_radius: 非极大值抑制的半径
    :param border_margin: 边界过滤的像素宽度
    :return: 经过NMS和边界过滤后的特征点和描述符
    """
    H, W = confidence_map.shape
    mask = np.zeros_like(confidence_map, dtype=bool)

    # 非极大值抑制: 遍历每个特征点，在其半径内检查更高的置信度值
    for y, x in keypoints:
        if mask[y, x] or y < border_margin or x < border_margin or y >= H - border_margin or x >= W - border_margin:
            continue

        # 获取局部区域的置信度
        local_patch = confidence_map[max(0, y - nms_radius):min(H, y + nms_radius + 1),
                      max(0, x - nms_radius):min(W, x + nms_radius + 1)]

        # 如果中心点不是局部最大值，则忽略此点
        if confidence_map[y, x] < np.max(local_patch):
            continue

        mask[y, x] = True

    # 根据mask过滤特征点和描述符
    keypoints_filtered = keypoints[mask[keypoints[:, 0], keypoints[:, 1]]]
    descriptors_filtered = descriptors[mask[keypoints[:, 0], keypoints[:, 1]]]

    return keypoints_filtered, descriptors_filtered


def extract_superpoint_features(confidence_map, descriptor_map, confidence_threshold=0.005, nms_radius=4,
                                border_margin=8):
    """
    提取 SuperPoint 特征点和描述符，使用非极大值抑制和边界过滤
    :param confidence_map: SuperPoint 网络的置信度图 (H, W)
    :param descriptor_map: SuperPoint 网络的描述符图 (H, W, D) D 表示描述符维度
    :param confidence_threshold: 保留特征点的置信度阈值
    :param nms_radius: NMS的半径
    :param border_margin: 边界过滤的像素宽度
    :return: 筛选后的特征点坐标和对应的描述符
    """
    # 找到所有 confidence_map 大于指定阈值的点
    keypoints = np.argwhere(confidence_map > confidence_threshold)
    descriptors = descriptor_map[keypoints[:, 0], keypoints[:, 1]]

    # 执行 NMS 和 边界过滤
    keypoints, descriptors = nms_and_filter(keypoints, descriptors, confidence_map, nms_radius, border_margin)

    return keypoints, descriptors


def resize_and_crop(image: np.ndarray, target_height: int = 480, target_width: int = 640) -> np.ndarray:
    """
    检查图像尺寸，并将其裁剪并等比缩放至目标尺寸。

    :param image: 输入的图像，形状为 (H, W)
    :param target_height: 目标高度
    :param target_width: 目标宽度
    :return: 处理后的图像，形状为 (target_height, target_width)
    """
    h, w = image.shape[:2]

    # 计算缩放比例，保持长宽比
    scale = min(target_height / h, target_width / w)

    # 先进行缩放，保持图像的长宽比
    new_h, new_w = int(h * scale), int(w * scale)
    resized_image = cv2.resize(image, (new_w, new_h))

    # 计算裁剪的起始位置，以便将图像居中
    crop_y = (new_h - target_height) // 2 if new_h > target_height else 0
    crop_x = (new_w - target_width) // 2 if new_w > target_width else 0

    # 裁剪图像
    cropped_image = resized_image[crop_y:crop_y + target_height, crop_x:crop_x + target_width]

    return cropped_image



def main():
    import cv2
    from tqdm import tqdm

    save_folder = '/home/dragonz/ADaryl/Codes/Python/superpoint22222/pytorch-superpoint22222222/DBOWdescriptors/'  # 指定保存文件夹路径
    os.makedirs(save_folder, exist_ok=True)  # 如果文件夹不存在则创建

    with torch.no_grad():
        device = torch.device('cuda:0')
        model = SuperPointNet_gauss22()
        model = model.to(device)

        weights_path = '/home/dragonz/ADaryl/Codes/Python/superpoint22222/pytorch-superpoint22222222/logs/superpoint_coco_heat2_0/checkpoints/superPointNet_170000_checkpoint.pth.tar'
        print("weights_path:", weights_path)

        weights = torch.load(weights_path, map_location=device)
        model.load_state_dict(weights['model_state_dict'])
        model.eval()

        image_folder = '/home/dragonz/ADaryl/Datasets/COCO2017/test2017'

        hierarchical_descriptors = {}
        batch_size = 5000
        image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # 处理图像文件，按批次分组
        for i, image_name in tqdm(enumerate(image_files), total=len(image_files), desc="Processing images"):
            image_path = os.path.join(image_folder, image_name)

            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"无法读取图像文件: {image_path}")
                continue

            # 确保图像尺寸符合要求：裁剪并等比缩放至480x640
            img = resize_and_crop(img, target_height=480, target_width=640)

            # 将图像转换为张量并规范化
            img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0

            confidence_map, descriptor_map = model(img)
            confidence_map = confidence_map.squeeze().cpu().numpy()
            descriptor_map = descriptor_map.squeeze().cpu().numpy().transpose(1, 2, 0)

            keypoints, descriptors = extract_superpoint_features(confidence_map, descriptor_map)
            # Store hierarchical descriptors (keypoints and descriptors for each image)
            hierarchical_descriptors[image_name] = {
                'keypoints': keypoints,
                'descriptors': descriptors
            }

            # 每处理完 `batch_size` 张图像，保存一次
            if (i + 1) % batch_size == 0 or (i + 1) == len(image_files):
                batch_filename = os.path.join(save_folder, f'hierarchical_descriptors_batch_{i // batch_size + 1}.npy')
                np.save(batch_filename, hierarchical_descriptors, allow_pickle=True)
                print(f"已保存 {batch_filename}")
                hierarchical_descriptors.clear()  # 清空当前批次的描述符，准备处理下一批

        print("所有批次的描述符已保存。")


if __name__ == '__main__':
    main()



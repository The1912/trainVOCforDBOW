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
        Output from detector head
        tensor (batch_size, 65, Hc, Wc)

    :return:
        3D heatmap
        tensor (batch_size, H, W)
    '''

    # Apply softmax
    dense = nn.functional.softmax(semi, dim=1)  # [batch_size, 65, Hc, Wc]

    # Remove dustbin class
    nodust = dense[:, :-1, :, :]  # [batch_size, 64, Hc, Wc]

    # Use DepthToSpace
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

        dn = torch.norm(resized_desc, p=2, dim=1, keepdim=True)  # Compute the norm
        fused_descriptor = resized_desc.div(dn + 1e-8)  # Divide by norm to normalize, avoid division by zero

        return confidence_map, fused_descriptor


def nms_and_filter(keypoints, descriptors, confidence_map, nms_radius=5, border_margin=8):
    """
    Perform Non-Maximum Suppression and boundary point filtering
    :param keypoints: Original keypoint coordinates
    :param descriptors: Original keypoint descriptors
    :param confidence_map: Confidence map
    :param nms_radius: Non-Maximum Suppression radius
    :param border_margin: Border margin width in pixels
    :return: Filtered keypoints and descriptors after NMS and boundary filtering
    """
    H, W = confidence_map.shape
    mask = np.zeros_like(confidence_map, dtype=bool)

    # Non-Maximum Suppression: Iterate over keypoints and check for higher confidence values within the radius
    for y, x in keypoints:
        if mask[y, x] or y < border_margin or x < border_margin or y >= H - border_margin or x >= W - border_margin:
            continue

        # Get local patch confidence
        local_patch = confidence_map[max(0, y - nms_radius):min(H, y + nms_radius + 1),
                      max(0, x - nms_radius):min(W, x + nms_radius + 1)]

        # If the center is not the local maximum, skip this point
        if confidence_map[y, x] < np.max(local_patch):
            continue

        mask[y, x] = True

    # Filter keypoints and descriptors using the mask
    keypoints_filtered = keypoints[mask[keypoints[:, 0], keypoints[:, 1]]]
    descriptors_filtered = descriptors[mask[keypoints[:, 0], keypoints[:, 1]]]

    return keypoints_filtered, descriptors_filtered


def extract_superpoint_features(confidence_map, descriptor_map, confidence_threshold=0.005, nms_radius=4,
                                border_margin=8):
    """
    Extract SuperPoint keypoints and descriptors, using Non-Maximum Suppression and boundary filtering
    :param confidence_map: SuperPoint network confidence map (H, W)
    :param descriptor_map: SuperPoint network descriptor map (H, W, D) D is descriptor dimension
    :param confidence_threshold: Confidence threshold for keypoints
    :param nms_radius: NMS radius
    :param border_margin: Border filtering width in pixels
    :return: Filtered keypoint coordinates and corresponding descriptors
    """
    # Find all points with confidence_map greater than the threshold
    keypoints = np.argwhere(confidence_map > confidence_threshold)
    descriptors = descriptor_map[keypoints[:, 0], keypoints[:, 1]]

    # Perform NMS and border filtering
    keypoints, descriptors = nms_and_filter(keypoints, descriptors, confidence_map, nms_radius, border_margin)

    return keypoints, descriptors


def resize_and_crop(image: np.ndarray, target_height: int = 480, target_width: int = 640) -> np.ndarray:
    """
    Check the image size, and crop and scale it to the target size while maintaining the aspect ratio.

    :param image: Input image, shape (H, W)
    :param target_height: Target height
    :param target_width: Target width
    :return: Processed image, shape (target_height, target_width)
    """
    h, w = image.shape[:2]

    # Calculate scale to maintain aspect ratio
    scale = min(target_height / h, target_width / w)

    # First resize to maintain aspect ratio
    new_h, new_w = int(h * scale), int(w * scale)
    resized_image = cv2.resize(image, (new_w, new_h))

    # Calculate cropping position to center the image
    crop_y = (new_h - target_height) // 2 if new_h > target_height else 0
    crop_x = (new_w - target_width) // 2 if new_w > target_width else 0

    cropped_image = resized_image[crop_y:crop_y + target_height, crop_x:crop_x + target_width]

    return cropped_image
def main():
    import cv2
    from tqdm import tqdm

    save_folder = 'DBOWdescriptors/'  # Specify the folder path to save the files
    os.makedirs(save_folder, exist_ok=True)  # Create the folder if it doesn't exist

    with torch.no_grad():
        device = torch.device('cuda:0')
        model = SuperPointNet_gauss22()
        model = model.to(device)

        weights_path = 'superPointNet_170000_checkpoint.pth.tar'
        print("weights_path:", weights_path)

        weights = torch.load(weights_path, map_location=device)
        model.load_state_dict(weights['model_state_dict'])
        model.eval()

        image_folder = 'Datasets/COCO2017/test2017'

        hierarchical_descriptors = {}
        batch_size = 5000
        image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # Process image files and group them in batches
        for i, image_name in tqdm(enumerate(image_files), total=len(image_files), desc="Processing images"):
            image_path = os.path.join(image_folder, image_name)

            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Unable to read image file: {image_path}")
                continue

            # Ensure the image size is as required: crop and scale it to 480x640
            img = resize_and_crop(img, target_height=480, target_width=640)

            # Convert the image to a tensor and normalize
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

            # Save after processing each batch of images (batch_size)
            if (i + 1) % batch_size == 0 or (i + 1) == len(image_files):
                batch_filename = os.path.join(save_folder, f'hierarchical_descriptors_batch_{i // batch_size + 1}.npy')
                np.save(batch_filename, hierarchical_descriptors, allow_pickle=True)
                print(f"Saved {batch_filename}")
                hierarchical_descriptors.clear()  # Clear the current batch's descriptors and prepare for the next batch

        print("All batches of descriptors have been saved.")


if __name__ == '__main__':
    main()

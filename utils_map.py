import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
from torch.nn import functional as F
from torchvision import transforms
import os
import cv2


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes

        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0

# 注意力可视化
def feature_vis(features,save_path, name, stage):  # feaats形状: [b,c,h,w]
    output_shape = (224, 224)  # 输出形状
    # channel_mean = torch.mean(features, dim=1, keepdim=True)
    channel_mean,_ = torch.max(features, dim=1, keepdim=True)
    channel_mean = F.interpolate(channel_mean, size=output_shape, mode='bilinear', align_corners=False)
    channel_mean = channel_mean.squeeze(0).squeeze(0).cpu().detach().numpy()  # 四维压缩为二维
    channel_mean = (
                ((channel_mean - np.min(channel_mean)) / (np.max(channel_mean) - np.min(channel_mean))) * 255).astype(
        np.uint8)
    savedir = os.path.join(save_path, name)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    channel_mean = cv2.applyColorMap(channel_mean, cv2.COLORMAP_JET)
    channel_mean = cv2.rotate(channel_mean, cv2.ROTATE_90_CLOCKWISE)
    channel_mean = cv2.flip(channel_mean, 1)
    cv2.imwrite(savedir + '/' + stage + '.png', channel_mean)


def test_single_volume(image, label, net, classes, patch_size=[256, 256],
                       test_save_path=None, case=None, z_spacing=1, attention_map_save_path=None):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            x_transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            print(slice.shape, case, ind, test_save_path)
            input = x_transforms(slice).unsqueeze(0).float().cuda()

            net.eval()
            with torch.no_grad():
                outputs, map0, map1, map2, map3, map4 = net(input)
                #print(map1.shape, map2.shape, map3.shape)
                ########################################################
                #print(input.shape)
                feature_vis(input, attention_map_save_path, case, str(ind) + '_img')
                feature_vis(map0, attention_map_save_path, case, str(ind)+'_map0')
                feature_vis(map1, attention_map_save_path, case, str(ind)+'_map1')
                feature_vis(map2, attention_map_save_path, case, str(ind)+'_map2')
                feature_vis(map3, attention_map_save_path, case, str(ind)+'_map3')
                feature_vis(map4, attention_map_save_path, case, str(ind)+'_map4')
                ########################################################

                # outputs = F.interpolate(outputs, size=slice.shape[:], mode='bilinear', align_corners=False)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            out = torch.argmax(torch.softmax(net(input), dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list
import torch
from PIL import Image
import pandas as pd
from monai.transforms import (
    NormalizeIntensity,
)
import os
import SimpleITK as sitk
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

from visualizer import get_local
get_local.activate() # 激活装饰器

from model import BiomedClipModel

def read_information(id):
    data_info_path = "/home/hubin/Datasets/ISLES2024/Text_information.xlsx"
    df = pd.read_excel(data_info_path)
    row = df[df['id'] == id]
    label = row.iloc[0]['labels']
    patient_prompt = row.iloc[0]['patient_prompt']

    norm = NormalizeIntensity()
    img_path = os.path.join("/home/hubin/Datasets/ISLES2024/MRS_DATA/strip_MNI_MIP_40", id + '_MCA_rotate.nii.gz')
    itk = sitk.ReadImage(img_path)
    arr = sitk.GetArrayFromImage(itk).astype(float)
    arr = arr[16:240, 16:240]  # Vit
    arr = torch.Tensor(arr).unsqueeze(0)
    arr = np.concatenate((arr, arr, arr), axis=0)
    arr = norm(arr).unsqueeze(0)
    label = torch.Tensor(label)
    return arr, patient_prompt, label


def visualize_map(original_image,feature_map,alpha=0.7,save_path_image='',save_path_map=''):
    feature_map = torch.tensor(feature_map).squeeze(0)
    #attention_map = feature_map
    attention_map = feature_map.max(dim=2)[0]
    #attention_map = feature_map.mean(dim=2)

    # Step 2: Interpolate to (224, 224)
    attention_map = F.interpolate(attention_map.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear',
                                  align_corners=False).squeeze()

    # Normalize the attention map
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min()) #第0层1-，第5层1-.
    attention_map = (attention_map.cpu().detach().numpy() * 255).astype(np.uint8)

    # Apply colormap
    cmap = plt.get_cmap('jet')
    attention_colored = (cmap(attention_map / 255.0)[:, :, :3] * 255).astype(np.uint8) #(224,224,3)

    # Convert original image to array and resize to (224, 224)
    original_image_array = np.array(original_image.squeeze(0).permute(1,2,0).cpu()) #(224,224,3)
    original_image_array = (original_image_array - original_image_array.min()) / (original_image_array.max() - original_image_array.min())
    original_image_array = (original_image_array * 255.0).astype(np.uint8)
    # Overlay the attention map
    overlay = (alpha * original_image_array + (1 - alpha) * attention_colored).astype(np.uint8)
    overlay_image = Image.fromarray(overlay)
    overlay_image.save(save_path_map)

    # image = original_image_array
    # image = Image.fromarray(image)
    # image.save(save_path_image)

if __name__ == '__main__':
    id = 'sub-stroke0137'
    images, patient_text, label = read_information(id)

    path = "/home/hubin/Codes/ISLES2024/image_text_models/savemodels/Biomedclip"
    model_path = os.path.join(path, "0fold_16epoch_80.77%acc_0.71%f1_0.622%loss.pth")

    images, patient_text, label = images.to(torch.float32).cuda(), patient_text, label.cuda()  # LLM时候使用
    save_path_map = os.path.join('image', id + '.png')

    model = BiomedClipModel().cuda()
    checkpoint = torch.load(model_path, map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    get_local.clear()
    out = model(images,patient_text)
    print(out)

    feature_map = get_local.cache
    feature_map = feature_map['Attention.forward'][0] #12个(1,12,197,64)
    feature_map = torch.from_numpy(feature_map)
    B, H, N, D = feature_map.shape  # 1, 12, 197, 64
    # 去掉 cls token，只保留 196 个 patch
    feature_map = feature_map[:, :, 1:, :]  # (1, 12, 196, 64)
    # 合并 head 维度
    feature_map = feature_map.permute(0, 2, 1, 3).reshape(B, 196, H * D)  # (1, 196, 768)
    # reshape 成 14x14 格式
    feature_map = feature_map.reshape(B, 14, 14, H * D).numpy()  # (1, 14, 14, 768)
    visualize_map(original_image=images, feature_map=feature_map, alpha=0.5, save_path_map=save_path_map)





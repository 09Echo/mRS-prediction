import torch
import os
import SimpleITK as sitk
from sklearn import metrics
import warnings
import numpy as np
import pandas as pd
from monai.transforms import (
    OneOf,
    LoadImage,
    Transpose,
    RandGaussianNoise,
    RandAdjustContrast,
    RandGaussianSmooth,
    RandGaussianSharpen,
    RandHistogramShift,
    NormalizeIntensity,
    Rand3DElastic,
    RandAffine,
    Zoom,
    Compose,
)
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = "3"

# 定义训练的设备
device = 'cuda'
from model import BiomedClipModel


class ProVe():
    def __init__(self):
        # data_info_path = "/home/hubin/Datasets/prove_home/mrs2d/295_information_new.xlsx"
        # self.label = pd.read_excel(data_info_path, sheet_name="LLM_sentence_final")['labels'].values
        # self.data = pd.read_excel(data_info_path, sheet_name="LLM_sentence_final")[['Prove-it ID','information','diff_prompt']].values

        # data_info_path =  "/home/hubin/Datasets/ISLES2024/Text_information.xlsx"
        # self.label = pd.read_excel(data_info_path)[['labels']].values
        # self.data = pd.read_excel(data_info_path)[['id','patient_prompt','VQA_prompt']].values

        data_info_path = "/home/hubin/Datasets/external/109_information.xlsx"
        self.label = pd.read_excel(data_info_path, sheet_name="LLM_78")['label'].values
        self.data = pd.read_excel(data_info_path, sheet_name="LLM_78")[
            ['id', 'patient_prompt']].values

        self.transform = Compose([
                NormalizeIntensity(),
        ])

    def __getitem__(self, item):
        data_list = self.data[item]
        patient_text = data_list[1]

        #img_path = os.path.join("/home/hubin/Datasets/prove_home/mrs2d/", data_list[0],'mca_mCTA1_brain_rotate.nii.gz')
        # img_path = os.path.join("/home/hubin/Datasets/ISLES2024/MRS_DATA/strip_MNI_MIP_40/", data_list[0]+'_MCA_rotate.nii.gz')
        img_path = os.path.join("/home/hubin/Datasets/external/", str(data_list[0]), 'MCA.nii.gz')

        itk = sitk.ReadImage(img_path)
        arr = sitk.GetArrayFromImage(itk).astype(float)
        arr = arr[16:240,16:240] #Vit
        arr = self.transform(arr)
        arr = torch.Tensor(arr).unsqueeze(0)
        arr = np.concatenate((arr, arr, arr), axis=0)
        label = int(self.label[item].item())
        return arr, patient_text, label ,str(data_list[0])

    def __len__(self):
        return len(self.label)

def calculate_specificity(confusion_matrix):
    true_positives = confusion_matrix[0, 0]
    false_negatives = confusion_matrix[0, 1]
    specificity = true_positives / (true_positives + false_negatives)
    return specificity

def calculate_sensitivity(confusion_matrix):
    true_negatives = confusion_matrix[1, 1]
    false_positives = confusion_matrix[1, 0]
    sensitivity = true_negatives / (true_negatives + false_positives)
    return sensitivity

test_batch_size=1

def extern_test():
    weights = []
    path = '/home/hubin/Codes/ISLES2024/image_text_models/savemodels/Biomedclip/'

    txt_path = '/home/hubin/Codes/ISLES2024/big_model/print/external_results_78'
    txt_result_path = os.path.join(txt_path, 'Biomedclip')
    for filename in os.listdir(path):
        if filename.endswith(".pth"):
            weights.append(path + filename)

    #weights = [path + '2fold_22epoch_76.00%acc_0.57%f1_0.657%loss.pth'] #biomedclip
    weights = [path + '4fold_23epoch_72.00%acc_0.59%f1_0.569%loss.pth'] #biomedclip

    id_all = []
    p_y_all = []
    t_y_all = []
    prob_all = []
    for ki in range(len(weights)):
        print(weights[ki])
        warnings.filterwarnings('ignore')  # 忽视warning
        test_ds = ProVe()
        test_dl = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, drop_last=False)

        model = BiomedClipModel().cuda()
        # load model weights
        weights_path =weights[ki]
        msg = model.load_state_dict(torch.load(weights_path)['model_state_dict']) #strict=False表示可以不精准匹配，保存的参数里有模型不需要的部分时可以使用
        print(msg)

        id = []
        p_y = []
        t_y = []
        outprob = []
        model.eval()
        with torch.no_grad():
            for step, test_data in enumerate(test_dl):
                test_images, test_patient_text, test_labels, test_list = test_data
                test_images, test_patient_text, test_labels = test_images.to(torch.float32).cuda(), test_patient_text, test_labels.cuda() #LLM时候使用
                id.append(test_list[0])  # id为样本名

                t_y = t_y + test_labels.tolist()
                outputs = model(test_images, test_patient_text)  # 输出值，有正有负 #维度从【2】变成【1，2】
                predict_y = torch.max(outputs, dim=1)[1]  # 类别 0，1。dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值
                p_y += predict_y.tolist()

                ex = np.exp(outputs.cpu())
                ex1 = (ex / ex.sum()).tolist()
                # out = ex1[0][test_labels.tolist()[0]]
                out = ex1[0][1]
                outprob.append(out)  # 类别概率
        id_all.extend(id)
        p_y_all.extend(p_y)
        t_y_all.extend(t_y)
        prob_all.extend(outprob)

        cmk = metrics.confusion_matrix(t_y, p_y)
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(t_y_all, p_y_all, average='binary')
        ACC = metrics.accuracy_score(t_y, p_y)
        AUC = metrics.roc_auc_score(t_y, outprob)
        sens = calculate_sensitivity(cmk)
        spe = calculate_specificity(cmk)

        print("混淆矩阵:", cmk)
        print("{}分类报告:".format(ki), metrics.classification_report(t_y, p_y, digits=3))  # digits保留小数点位数
        print("ACC:", '{:.4f}'.format(ACC))
        print("precision:", '{:.4f}'.format(precision))
        print("recall:", '{:.4f}'.format(recall))
        print("f1:", '{:.4f}'.format(f1))
        print("AUC:", '{:.4f}'.format(AUC))
        print("sensitivity:", '{:.4f}'.format(sens))
        print("specificity:", '{:.4f}'.format(spe))

        with open(txt_result_path, 'a') as file:
            # 持续写入内容
            file.write("混淆矩阵:" + '\n')
            for row in cmk:
                row_str = str(row).replace('[', '').replace(']', '')  # 去除列表括号
                file.write(row_str + '\n')
            file.write("分类报告：" + '\n')
            file.write(metrics.classification_report(t_y, p_y, digits=3) + '\n')
            file.write("ACC:" + str(ACC) + '\n')
            file.write("precision:" + str(precision) + '\n')
            file.write("recall:" + str(recall) + '\n')
            file.write("f1:" + str(f1) + '\n')
            file.write("AUC:" + str(AUC) + '\n')
            file.write("sensitivity:" + str(sens) + '\n')
            file.write("specificity:" + str(spe) + '\n')
            file.write('\n')

    text = pd.DataFrame({'id': id_all, 'true_label': t_y_all, 'predict_label': p_y_all, 'out_prob': prob_all})
    text.to_csv(os.path.join(path, "index_external_78.csv"), index=None, encoding='utf8')


if __name__ == '__main__':
    extern_test()

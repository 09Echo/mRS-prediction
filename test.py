import torch
import os
from torch.utils.data import DataLoader
from data import ProVe
from sklearn import metrics
import warnings
import numpy as np
import pandas as pd
import argparse
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# 定义训练的设备
device = 'cuda'
from multistage_diff import SAMVIT

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t_size", type=int, default=1, help="train batch size")
    parser.add_argument("--k_fold", type=int, default=5)
    parser.add_argument('--bert_chekpoint', default='/biolinkbert', help='huggingface_bert')
    parser.add_argument('--LVM_Med_checkpoint', default="/lvmmed_vit.pth")
    parser.add_argument("--checkpoint_name", default=['best_fold_0_model.pt','best_fold_1_model.pt','best_fold_2_model.pt','best_fold_3_model.pt','best_fold_4_model.pt'], help='model_name')
    parser.add_argument("--save_path",default='/mrs_prediction')
    parser.add_argument('--num_classes', type=int, default=3)
    args = parser.parse_args()
    return args

def mrstest(args):
    k_fold_num= args.k_fold
    txt_path = '/results'
    txt_path = os.path.join(txt_path,'model')
    id_all = []
    p_y_all = []
    t_y_all = []
    prob_all = []
    average = [[],[],[],[],[],[],[]]
    for ki in range(k_fold_num):
        weights = os.path.join(args.save_path, args.checkpoint_name[ki])
        warnings.filterwarnings('ignore')  # 忽视warning
        test_ds = ProVe(fold=ki, mode='test', n_splits=k_fold_num)
        test_dl = DataLoader(test_ds, batch_size=args.t_size, shuffle=False, drop_last=False)

        # create model
        model = SAMVIT(num_classes=2,textencoder_path = '/biolinkbert',\
                       imgencoder_path = "/lvmmed_vit.pth").cuda()
        # load model weights
        weights_path =weights[ki]
        model.load_state_dict(torch.load(weights_path)['model_state_dict']) #strict=False表示可以不精准匹配，保存的参数里有模型不需要的部分时可以使用

        id = []
        p_y = []
        t_y = []
        outprob = []
        model.eval()
        with torch.no_grad():
            for step, test_data in enumerate(test_dl):
                test_images, test_patient_text, test_diff_text, test_labels, test_list = test_data
                test_images, test_patient_text, test_diff_text, test_labels = test_images.to(torch.float32).cuda(), test_patient_text, test_diff_text, test_labels.cuda() #LLM时候使用
                id.append(test_list)

                t_y = t_y + test_labels.tolist()
                outputs = model(test_images, test_diff_text, test_patient_text)
                predict_y = torch.max(outputs, dim=1)[1]
                p_y += predict_y.tolist()

                ex = np.exp(outputs.cpu())
                ex1 = (ex / ex.sum()).tolist()
                out = ex1[0][1]
                outprob.append(out)
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
        print("ACC:", '{:.3f}'.format(ACC))
        average[0].append(ACC)
        print("precision:", '{:.3f}'.format(precision))
        average[1].append(precision)
        print("recall:", '{:.3f}'.format(recall))
        average[2].append(recall)
        print("f1:", '{:.3f}'.format(f1))
        average[3].append(f1)
        print("AUC:", '{:.3f}'.format(AUC))
        average[4].append(AUC)
        print("sensitivity:", '{:.3f}'.format(sens))
        average[5].append(sens)
        print("specificity:", '{:.3f}'.format(spe))
        average[6].append(spe)

        with open(txt_path, 'a') as file:
            file.write(str(ki) + "fold:" + '\n')
            file.write( "混淆矩阵:" + '\n')
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

    cm = metrics.confusion_matrix(t_y_all, p_y_all)
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(t_y_all, p_y_all, average='binary')

    ACC_std = np.std(average[0])
    precision_std = np.std(average[1])
    recall_std = np.std(average[2])
    f1_std = np.std(average[3])
    AUC_std = np.std(average[4])
    sen_std = np.std(average[5])
    spe_std = np.std(average[6])

    print("************平均**********" + '\n')
    print("ACC:", '{:.3f}'.format(sum(average[0])/k_fold_num))
    print("precision:", '{:.3f}'.format(sum(average[1])/k_fold_num))
    print("recall:", '{:.3f}'.format(sum(average[2])/k_fold_num))
    print("f1:", '{:.3f}'.format(sum(average[3])/k_fold_num))
    print("AUC:", '{:.3f}'.format(sum(average[4])/k_fold_num))
    print("sensitivity:", '{:.3f}'.format(sum(average[5])/k_fold_num))
    print("specificity:", '{:.3f}'.format(sum(average[6])/k_fold_num))

    with open(txt_path, 'a') as file:
        file.write("************平均**********" + '\n')
        file.write("ACC:" + str(np.mean(average[0])) + '\n')
        file.write("precision:" + str(np.mean(average[1])) + '\n')
        file.write("recall:" + str(np.mean(average[2])) + '\n')
        file.write("f1:" + str(np.mean(average[3])) + '\n')
        file.write("AUC:" + str(np.mean(average[4])) + '\n')
        file.write("sensitivity:" + str(np.mean(average[5])) + '\n')
        file.write("specificity:" + str(np.mean(average[6])) + '\n')
        file.write('\n')

    print("************总**********")
    print("混淆矩阵:", cm)
    print("ACC:", '{:.3f}'.format(metrics.accuracy_score(t_y_all, p_y_all)))
    print("precision:", '{:.3f}'.format(precision))
    print("recall:", '{:.3f}'.format(recall))
    print("f1:", '{:.3f}'.format(f1))
    print("AUC:", '{:.3f}'.format(metrics.roc_auc_score(t_y_all, prob_all)))
    print("sensitivity:", '{:.3f}'.format(calculate_sensitivity(cm)))
    print("specificity:", '{:.3f}'.format(calculate_specificity(cm)))
    text = pd.DataFrame({'id': id_all, 'true_label': t_y_all, 'predict_label': p_y_all, 'out_prob': prob_all})
    text.to_csv("index.csv", index=None, encoding='utf8')

    with open(txt_path, 'a') as file:
        # 持续写入内容
        file.write("************总**********" + '\n')
        file.write("混淆矩阵:" + '\n')
        for row in cm:
            row_str = str(row).replace('[', '').replace(']', '')  # 去除列表括号
            file.write(row_str + '\n')
        file.write("ACC：" + str(metrics.accuracy_score(t_y_all, p_y_all)) + '\n')
        file.write("precision：" + str(precision) + '\n')
        file.write("recall：" + str(recall) + '\n')
        file.write("f1：" + str(f1) + '\n')
        file.write("AUC：" + str(metrics.roc_auc_score(t_y_all, prob_all)) + '\n')
        file.write("specificity：" + str(calculate_specificity(cm)) + '\n')
        file.write("sensitivity：" + str(calculate_sensitivity(cm)) + '\n')
        file.write('\n')

    with open(txt_path, 'a') as file:
        file.write("************标准差**********" + '\n')
        file.write("ACC_std: " + str(ACC_std) + '\n')
        file.write("precision_std: " + str(precision_std) + '\n')
        file.write("recall_std: " + str(recall_std) + '\n')
        file.write("f1_std: " + str(f1_std) + '\n')
        file.write("AUC_std: " + str(AUC_std) + '\n')
        file.write("Sensitivity_std: " + str(sen_std) + '\n')
        file.write("Specificity_std: " + str(spe_std) + '\n')
        file.write('\n')
if __name__ == '__main__':
    args = parse_args()
    mrstest(args)

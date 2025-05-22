import torch
import os
from torch.utils.data import DataLoader
from utils_text_LLM import ProVe  #LLM时使用
from sklearn import metrics
import warnings
import numpy as np
import pandas as pd
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

# 定义训练的设备
device = 'cuda'
from model import BiomedClipModel

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

def mrstest():
    k_fold_num=5
    path = '/home/hubin/Codes/ISLES2024/image_text_models/savemodels/Biomedclip/'
    weights = [path + '0fold_16epoch_80.77%acc_0.71%f1_0.622%loss.pth', path + '1fold_20epoch_69.23%acc_0.64%f1_0.622%loss.pth',
               path + '2fold_22epoch_76.00%acc_0.57%f1_0.657%loss.pth', path + '3fold_29epoch_76.00%acc_0.67%f1_0.504%loss.pth',
               path + '4fold_23epoch_72.00%acc_0.59%f1_0.569%loss.pth']
    txt_path = '/home/hubin/Codes/ISLES2024/image_text_models/results'
    txt_path = os.path.join(txt_path,'Biomedclip')
    with open(txt_path, 'a') as file:
        # 持续写入内容
        file.write('\n')
        file.write("比例1:1或1:1.5,学习率1e-4或8e-4,Epochs:200,patience:50" + '\n')
        file.write("path:"+path+'\n')
    id_all = []
    p_y_all = []
    t_y_all = []
    prob_all = []
    average = [[],[],[],[],[],[],[]]
    for ki in range(k_fold_num):
        warnings.filterwarnings('ignore')  # 忽视warning
        test_ds = ProVe(fold=ki, mode='test', n_splits=k_fold_num)
        test_dl = DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, drop_last=False)

        # create model
        model = BiomedClipModel().cuda()

        # load model weights
        weights_path =weights[ki]
        #model = torch.nn.DataParallel(model).cuda() #多卡训练后的测试load也要分布式并行
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
                #test_images, test_text, test_labels = test_images.to(torch.float32).cuda(), test_text.to(torch.float32).cuda(), test_labels.cuda()
                id.append(test_list)  # id为样本名

                t_y = t_y + test_labels.tolist()
                outputs = model(test_images, test_patient_text)  # 输出值，有正有负 #维度从【2】变成【1，2】
                predict_y = torch.max(outputs, dim=1)[1]  # 类别 0，1。dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值
                p_y += predict_y.tolist()

                # ex = np.exp(outputs.cpu())
                # ex1 = (ex / ex.sum()).tolist()
                # # out = ex1[0][test_labels.tolist()[0]]
                # out = ex1[0][1]
                # outprob.append(out)  # 类别概率
                probs = torch.softmax(outputs, dim=1)[:, 1]  # 假设正类为第 1 类
                outprob.extend(probs.cpu().tolist())
        id_all.extend(id)
        p_y_all.extend(p_y)
        t_y_all.extend(t_y)
        prob_all.extend(outprob)
        # print('t_y:',t_y) #标签
        # print('p_y:', p_y) #预测值
        # print('outprob:', outprob) #预测概率
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
            # 持续写入内容
            file.write(weights[ki]+'\n')
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
    # print("{}分类报告:".format(ki),metrics.classification_report(t_y,p_y,digits=3)) #digits保留小数点位数
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
    text.to_csv(path + "index.csv", index=None, encoding='utf8')

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
    mrstest()

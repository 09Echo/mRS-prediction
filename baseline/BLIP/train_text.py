import torch
import os
import sys
import torch.nn as nn
from torch.utils.data import DataLoader
#from utils_text import ProVe  #使用1*22向量文本时
from utils_text_LLM import ProVe  #使用LLM做文本特征时
from tqdm import tqdm
from sklearn import metrics
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
import argparse
import random
import numpy as np
import wandb
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from model import Prediction


train_losses = [[], [], [], [], []]
test_losses = [[], [], [], [], []]
train_acc = [[], [], [], [], []]
max_acc=[0,0,0,0,0]


def get_args_parser():
    parser = argparse.ArgumentParser(description="2D classification tasks")
    parser.add_argument('--k_fold',default=5,type=int,help='cross validation')
    parser.add_argument('--lr',default=2e-4,help='learning rate')
    parser.add_argument('--train_bs',default=8,type=int,help='train batch size')
    parser.add_argument('--test_bs', default=16, type=int, help='test batch size')
    parser.add_argument('--epochs',default=200,type=int)
    parser.add_argument('--weights',default=[1,2],help='cross entropyloss') #1.65
    parser.add_argument('--save_path',default='/home/hubin/Codes/ISLES2024/image_text_models/savemodels/BLIP')
    return parser

RANDOM_SEED = 42 # any random number

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现

def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print(f"目录 {dir_path} 被创建.")
    else:
        print(f"目录 {dir_path} 已存在.")

if __name__ == '__main__':
    warnings.filterwarnings('ignore')#忽视warning
    parser = get_args_parser()
    args = parser.parse_args()
    set_seed(RANDOM_SEED)

    #for ki in range(args.k_fold):
    for ki in [1]:
        best_val_loss = float('inf')
        epochs_no_improve = 0
        patience = 40

        ensure_dir_exists(args.save_path)  # 确保model_save文件夹存在

        print("*******第---{}---折*******".format(ki + 1))
        train_ds = ProVe(fold=ki,mode='train', n_splits=args.k_fold)
        train_dl = DataLoader(train_ds, batch_size=args.train_bs, shuffle=False, drop_last=False,num_workers=4)

        test_ds = ProVe(fold=ki, mode='test',n_splits=args.k_fold)
        test_dl = DataLoader(test_ds, batch_size=args.test_bs, shuffle=False, drop_last=False,num_workers=4)

        # batch数量
        train_num = len(train_dl)
        test_num = len(test_dl)

        #call the model

        model = Prediction(pretrained='./model_base.pth')
        total = sum(param.numel() for param in model.parameters() if param.requires_grad)
        print('  + Number of Backbone Params: %.4f(e6)' % (total / 1e6))
        # for param_name, param in model.named_parameters():
        #     print(param_name, param.requires_grad)

        model = model.cuda()

        weights = args.weights
        class_weights = torch.FloatTensor(weights).cuda()
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        loss_fn = loss_fn.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_acc = 0.0
        best_f1=0.0
        train_counter = [(i + 1) for i in range(args.epochs)]
        test_counter = train_counter

        num = []
        xsl=0
        for epoch in range(args.epochs):
            # train
            model.train()
            running_loss = 0.0
            acc1 = 0
            for step, data in enumerate(train_dl):
                images, patient_text, labels, train_list = data
                images, patient_text, labels = images.to(torch.float32).cuda(), patient_text, labels.cuda() #使用LLM时
                #images, text, labels = images.to(torch.float32).cuda(), text.to(torch.float32).cuda(), labels.cuda() #使用1*22向量时，要放在cuda上
                #logits, T= model(images)
                logits = model(images, patient_text, mode = 'multimodal')
                predict_y = torch.max(logits, dim=1)[1]

                loss = loss_fn(logits, labels) #是nn.logSoftmax()和nn.nllloss()的整合
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                acc1 += torch.eq(predict_y, labels).sum().item()

            train_accurate = acc1 / len(train_ds)
            t_loss = running_loss / train_num

            train_acc[ki].append(train_accurate)
            train_losses[ki].append(t_loss)

            # test
            acc = 0.0  # accumulate accurate number / epoch
            r_loss = 0.0
            p_y = []
            t_y = []
            q_all=[]
            id_all = []
            outprob = []
            # eval
            model.eval()
            with torch.no_grad():
                for step, test_data in enumerate(test_dl):
                    test_images, test_patient_text, test_labels, test_list= test_data

                    test_images, test_patient_text, test_labels = test_images.to(torch.float32).cuda(), test_patient_text, test_labels.cuda()
                    #test_images, test_text, test_labels = test_images.to(torch.float32).cuda(), test_text.to(torch.float32).cuda(), test_labels.cuda()
                    for t in test_list:
                        id_all.append(t)
                    t_y = t_y + test_labels.tolist()
                    outputs = model(test_images,test_patient_text,mode='multimodal')
                    predict_y = torch.max(outputs, dim=1)[1]
                    loss = loss_fn(outputs, test_labels)
                    acc += torch.eq(predict_y, test_labels).sum().item()
                    p_y = p_y + predict_y.tolist()
                    r_loss += loss.item()

            test_accurate = acc / len(test_ds)
            v_loss = r_loss / test_num

            #wandb.log({'train_loss': t_loss, 'test_loss': v_loss, 'learning_rate': optimizer.param_groups[0]['lr']})
            scheduler.step()
            precision, recall, test_f1, _ = metrics.precision_recall_fscore_support(t_y, p_y, average='binary')
            cm = metrics.confusion_matrix(t_y, p_y)
            #auc = metrics.roc_auc_score(t_y, outprob)

            test_losses[ki].append(v_loss)
            print('[epoch %d] train_loss: %.3f  test_loss: %.3f  train_accuracy: %.3f test_accuracy: %.3f test_f1: %.3f' %
                  (epoch + 1, t_loss,v_loss,train_accurate, test_accurate, test_f1))

            OUT = True
            if test_accurate >= best_acc:
                best_acc = test_accurate
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': v_loss, },
                    os.path.join(args.save_path, "{}fold_{}epoch_{:.2f}%acc_{:.2f}%f1_{:.3f}%loss.pth").format(
                        ki, epoch + 1, test_accurate * 100, test_f1, v_loss))
                print("最优模型已保存.")
                print("混淆矩阵:", cm)
                OUT = False

            if test_f1 >= best_f1:
                best_f1 = test_f1
                if OUT:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': v_loss, },
                        os.path.join(args.save_path, "{}fold_{}epoch_{:.2f}%acc_{:.2f}%f1_{:.3f}%loss.pth").format(
                            ki, epoch + 1, test_accurate * 100, test_f1, v_loss))
                    print("最优模型已保存.")
                    print("混淆矩阵:", cm)

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print(f'Early stopping after {epoch + 1} epochs')
                    break
        max_acc[ki] = best_acc
        print('best acc:', best_acc)
        #wandb.finish()

    av_acc = sum(max_acc) / args.k_fold
    print("==============    K Fold validation    ================")
    print("acc:{}".format(av_acc))



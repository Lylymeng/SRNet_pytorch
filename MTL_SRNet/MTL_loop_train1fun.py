import torch
import random
from MTL_dataloader1 import generate_train_data
from MTL_dataloader1 import generate_valid_data
#from rand_dataloader import generate_train_data
#from rand_dataloader import generate_valid_data
import copy
from torch.utils.tensorboard import SummaryWriter
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import numpy as np

#学习率调整
def adjust_learning_rate(optimizer, gamma, global_step):
    lr = 1e-3 * (gamma ** global_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def plot_embedding_2D(data, label, title):
    #绘制tsne图
    color_map = ['r','b'] # 2个类，准备2种颜色，其中红色是载体图像，然后蓝色是隐写图像
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    #print("data_shape[0]:", data.shape[0]) 
    #print("data_shape:", data.shape)  
    #print("label_shape:", label.shape)  
    for i in range(data.shape[0]):
        plt.plot(data[i, 0], data[i, 1], marker='o', markersize=1, color=color_map[label[i]])
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

#验证
def validation(model, valid_loader, device, criterion, writer, train_step,bpp,epoch):
    cnt = 0  # validation中所有图片的数量
    total_acc = 0  # 所有图片的accuracy的和
    total_loss = 0  # 所有图片的loss的和
    model.to(device)
    feature_list = []
    label_list = []
    #评估模式
    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.view(-1, 256, 256, 1).to(device)
            labels = labels.view(-1, 1).to(device)
            labels = torch.squeeze(labels)
            cnt += images.shape[0]
            #模型输出结果
            model_output1,model_output2,feature_temp = model(images)
            #记录特征图与标签
            feature_list.append(feature_temp)
            label_list.append(labels)
            #计算损失
            loss = criterion(model_output1, labels)
            total_loss += loss.item()
            # 计算accuracy
            model_label = torch.max(model_output1, dim=1)[1]
            temp_acc = torch.eq(model_label, labels).sum().item()
            total_acc += temp_acc

        avg_acc = total_acc / cnt
        avg_loss = total_loss / cnt
        writer.add_scalar('valid_accuracy', avg_acc, epoch)
        writer.add_scalar('valid_loss', avg_loss, epoch)
        print('__________currIter: %d || valid_loss: %.6f || valid_acc: %.6f__________' % (train_step, avg_loss, avg_acc))
    #在每次验证时记录valid数据集的tsne图
    '''
    print('Begining TSNE......')
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0)
    feature_list = torch.cat(feature_list,axis = 0).cpu().numpy()
    label_list = torch.cat(label_list,axis = 0).cpu().numpy()
    n_samples, n_features = feature_list.shape
    print("n_samples: ", n_samples)  #[4000]
    print("n_features: ", n_features)  #[256]
    result_2D = tsne_2D.fit_transform(feature_list)
    print('Finished......')
    fig1 = plot_embedding_2D(result_2D, label_list, f'valid_{bpp}_epoch{epoch}')    # 将二维数据用plt绘制出来
    fig1.show()
    plt.savefig(f'../../../autodl-tmp/image_tsne/MIPOD/valid_{bpp}_epoch{epoch}.jpg')
    #plt.pause(50)
    n_samples, n_features = feature_list.shape
    plt.close(fig1)
    '''
    return avg_acc


def train(model,
           bpp_datasets,
           batch_size,
           optimizer,
           criterion,
           device,
           EPOCHS,
           valid_interval=5000,
           write_interval=100,
           save_interval=5000,
           load_path=None,
           ):

    best_model_state_path ='./SUNI_bestModel_0.4_epoch180_0.8075.pth'
    #数据集循环
    for bpp, data_paths in bpp_datasets.items():
        print(f"Training on bpp={bpp} dataset...")
        highest_accuracy = 0.0
        total_loss = 0
        total_loss1 = 0
        total_loss2 = 0
        total_task1_acc=0
        total_task2_acc=0
        lr_adjust_step = 0
        model_save_cnt = 0
        acc_history = []
        converged = False
        train_step = 0

        #生成tensorboard文件
        writer = SummaryWriter(log_dir=f'./tflogs/SUNI_{bpp}')
        #数据处理
        train_loader = generate_train_data(data_paths, batch_size)
        
        valid_loader = generate_valid_data(data_paths, batch_size)
        
        print('model down successfully')
        #导入之前训练的最优模型
        if best_model_state_path:
            model_state_dict = torch.load(best_model_state_path)
            model.load_state_dict(model_state_dict)
            print("Loaded model from", best_model_state_path)
        data_gen = train_loader.dataset
        model.to(device)
        #epoch循环
        for epoch in range(1, EPOCHS + 1):
            print('start-epoch:%d||train_step:%d' % (epoch, train_step))
            if converged:#若收敛，则跳出该循环，进入下一个数据集
                break

            for index, (images, labels) in enumerate(train_loader):
                train_step += 1

                images = images.view(-1, 256, 256, 1).to(device)
                labels = labels.view(64,-1).to(device)
                labels = torch.squeeze(labels)
                #print('images.shape: ', images.shape, 'labels.shape: ', labels.shape)
                #print('labels.center',labels)
                optimizer.zero_grad()
                main_task_output,aux_task_output,feature_temp=model(images)
                #print(main_task_output)
                #任务1索引
                task1_label = torch.max(main_task_output, dim=1)[1]
                #print('task1_label',task1_label)
                #任务2索引
                task2_label = torch.max(aux_task_output,dim=1)[1]
                #任务1准确率计算
                task1_acc = torch.eq(task1_label, labels[:, 0]).sum().item()
                total_task1_acc += task1_acc
                #任务2准确率计算
                task2_acc = torch.eq(task2_label, labels[:, 1]).sum().item()
                total_task2_acc += task2_acc
                #计算损失
                loss1=criterion(main_task_output,labels[:,0])
                loss2=criterion(aux_task_output,labels[:,1])
                loss=0.9*loss1 + 0.1*loss2
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                total_loss1 += loss1.item()
                total_loss2 += loss2.item()


                if train_step % write_interval == 0:
                    #任务1的准确率与损失
                    temp_acc_1 = total_task1_acc / (write_interval*images.shape[0])
                    temp_loss1=total_loss1 / (write_interval * images.shape[0])
                    #任务2的准确率与损失
                    temp_acc_2 = total_task2_acc / (write_interval*images.shape[0])
                    temp_loss2=total_loss2 / (write_interval * images.shape[0])
                    print('EPOCH: %d/%d || currIter: %d || train_loss_task1: %.6f || train_acc_task1: %.6f' %
                          (epoch, EPOCHS, train_step, temp_loss1, temp_acc_1))
                    writer.add_scalar('train_accuracy_task1', temp_acc_1, epoch)
                    writer.add_scalar('train_loss_task1', temp_loss1, epoch)
                    writer.add_scalar('train_accuracy_task2', temp_acc_2, epoch)
                    writer.add_scalar('train_loss_task2', temp_loss2, epoch)
                    total_loss1 = 0
                    total_loss2 = 0
                    total_task1_acc = 0
                    total_task2_acc = 0
                    total_loss = 0
                    
                    
                if train_step % valid_interval == 0:
                    
                    v_acc = validation(model, valid_loader, device, criterion, writer, train_step, bpp,epoch)
                    acc_history.append(v_acc)
                   #判断收敛
                    if v_acc > 0.85 and len(acc_history) >= 15 and max(acc_history[-15:]) - min(
                            acc_history[-15:]) < 0.01:
                        #若收敛，则跳出循环，保存模型，进入下一个数据集
                        converged = True
                        break
                    #若不收敛，则更新最大准确率和最佳模型
                    if v_acc > highest_accuracy:
                        highest_accuracy = v_acc
                        best_model_state = copy.deepcopy(model.state_dict())
                        
                if train_step % save_interval == 0:
                    
                    torch.save(model.state_dict(), f'./Model/SUNI/Model_{bpp}_epoch{epoch}.pth')
                    model_save_cnt += 1

                if epoch == 80 or train_step == 120:
                    lr_adjust_step += 1
                    adjust_learning_rate(optimizer, 0.1, lr_adjust_step)

        if best_model_state:
            torch.save(best_model_state,
                       f'./Model/SUNI/best/bestModel_{bpp}_epoch{epoch}_{highest_accuracy:.4f}.pth')
            best_model_state_path = f'./Model/SUNI/best/bestModel_{bpp}_epoch{epoch}_{highest_accuracy:.4f}.pth'
            print(f'Saved best model with accuracy {highest_accuracy:.4f} for bpp {bpp}')

    writer.close()




def test1(model, test_loader, device, weight_path=None):
    model.to(device)

    # 加载之前训练好的模型
    assert weight_path is not None, 'weight_path is None, please change weight_path'
    model.load_state_dict(torch.load(weight_path))

    # 二分类对应的四种结果
    TTCounter = 0
    TFCounter = 0
    FTCounter = 0
    FFCounter = 0

    # 含密图像和原始图像的数量
    TCounter = 0
    FCounter = 0

    step_cnt = 0
    model.eval()
    with torch.no_grad():
        for index, (images, labels) in enumerate(test_loader):
            step_cnt += 1
            cover_img, stego_img = images.view(-1, 256, 256, 1).to(device)
            cover_img = cover_img.unsqueeze(0)
            stego_img = stego_img.unsqueeze(0)
            # print(cover_img.shape, stego_img.shape)

            cover_label, stego_label = labels.view(-1, 1).to(device)
            cover_label = cover_label.item()
            stego_label = stego_label.item()
            # print(cover_label, stego_label)

            flag = random.randint(0, 1)

            if flag == 0:
                # 选择原始图像
                FCounter += 1
                model_output = model(cover_img)
                model_label = torch.max(model_output, dim=1)[1].item()
                if model_label == 0:
                    FFCounter += 1
                else:
                    FTCounter += 1
            else:
                # 选择含密图像
                TCounter += 1
                model_output = model(stego_img)
                model_label = torch.max(model_output, dim=1)[1].item()
                if model_label == 0:
                    TFCounter += 1
                else:
                    TTCounter += 1

            if step_cnt % 50 == 0:
                print(
                    'cnt: %d || TT: %d/%d, FF: %d/%d, TF: %d/%d, FT: %d/%d || PosCount: %d, NegCount: %d, correct: %.4f' %
                    (step_cnt,
                     TTCounter, TCounter,
                     FFCounter, FCounter,
                     TFCounter, TCounter,
                     FTCounter, FCounter,
                     TCounter, FCounter,
                     (TTCounter + FFCounter) * 1.0 / step_cnt))

        print('\nTOTAL RESULT: ')
        print('TT: %d/%d, FF: %d/%d, TF: %d/%d, FT: %d/%d || PosCount: %d, NegCount: %d, correct: %.4f' %
              (TTCounter, TCounter,
               FFCounter, FCounter,
               TFCounter, TCounter,
               FTCounter, FCounter,
               TCounter, FCounter,
               (TTCounter + FFCounter) * 1.0 / step_cnt))


def test(model, test_loader, device, criterion,weight_path=None):
    model.to(device)
    # 加载之前训练好的模型
    assert weight_path is not None, 'weight_path is None, please change weight_path'
    model.load_state_dict(torch.load(weight_path))
    cnt = 0  # 测试集中所有图片的数量
    total_acc = 0  # 所有图片的accuracy的总和
    total_loss = 0  # 所有图片的loss的总和
    model.to(device)
    feature_list = []
    label_list = []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 256, 256, 1).to(device)
            labels = labels.view(-1, 1).to(device)
            labels = torch.squeeze(labels)

            cnt += images.shape[0]

            model_output1,model_output2,feature_temp = model(images)
            #记录特征图与标签
            feature_list.append(feature_temp)
            label_list.append(labels)

            loss = criterion(model_output1, labels)
            total_loss += loss.item()
            # 计算accuracy
            model_label = torch.max(model_output1, dim=1)[1]
            temp_acc = torch.eq(model_label, labels).sum().item()
            total_acc += temp_acc

        avg_acc = total_acc / cnt
        avg_loss = total_loss / cnt

        print('test_loss: %.6f || test_acc: %.6f' % (avg_loss, avg_acc))

    print('Begining TSNE......')
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0)
    feature_list = torch.cat(feature_list,axis = 0).cpu().numpy()
    label_list = torch.cat(label_list,axis = 0).cpu().numpy()
    n_samples, n_features = feature_list.shape
    print("n_samples: ", n_samples)  #[4000]
    print("n_features: ", n_features)  #[256]
    result_2D = tsne_2D.fit_transform(feature_list)
    print('Finished......')
    fig1 = plot_embedding_2D(result_2D, label_list, 'test')    # 将二维数据用plt绘制出来
    fig1.show()
    plt.savefig(f'./image_tsne/test_SUNI02.jpg')
    #plt.pause(50)
    n_samples, n_features = feature_list.shape
    plt.close(fig1)
    return avg_acc

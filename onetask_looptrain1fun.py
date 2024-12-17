import torch
import random
import copy
from dataloader import generate_data, generate_test_data
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
#import src
import numpy as np

def adjust_learning_rate(optimizer, gamma, global_step):
    lr = 1e-3 * (gamma ** global_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#绘制tsne图

def plot_embedding_2D(data, label, title):
    color_map = ['r','b'] # 2个类，准备2种颜色，其中红色是载体图像，然后蓝色是隐写图像
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    print("data_shape[0]:", data.shape[0]) 
    print("data_shape:", data.shape)  
    print("label_shape:", label.shape)  
    for i in range(data.shape[0]):
        plt.plot(data[i, 0], data[i, 1], marker='o', markersize=1, color=color_map[label[i]])
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def validation(model, valid_loader, device, criterion, writer, train_step,epoch,bpp):
    cnt = 0  # validation中所有图片的数量
    total_acc = 0  # 所有图片的accuracy的和
    total_loss = 0  # 所有图片的loss的和
    model.to(device)
    feature_list = []
    label_list = []
    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.view(-1, 256, 256, 1).to(device)
            labels = labels.view(-1, 1).to(device)
            labels = torch.squeeze(labels)

            cnt += images.shape[0]

            model_output,feature_temp = model(images)
            
            feature_list.append(feature_temp)
            label_list.append(labels)

            loss = criterion(model_output, labels)
            total_loss += loss.item()
            # 计算accuracy
            model_label = torch.max(model_output, dim=1)[1]
            temp_acc = torch.eq(model_label, labels).sum().item()
            total_acc += temp_acc

        avg_acc = total_acc / cnt
        avg_loss = total_loss / cnt
        writer.add_scalar('valid_accuracy', avg_acc, epoch)
        writer.add_scalar('valid_loss', avg_loss, epoch)
        print('currIter: %d || valid_loss: %.6f || valid_acc: %.6f' % (train_step, avg_loss, avg_acc))
    print('Begining TSNE......')
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0)
    feature_list = torch.cat(feature_list,axis = 0).cpu().numpy()
    label_list = torch.cat(label_list,axis = 0).cpu().numpy()
    n_samples, n_features = feature_list.shape
    print("n_samples: ", n_samples)  #[4000]
    print("n_features: ", n_features)  #[256]
    result_2D = tsne_2D.fit_transform(feature_list)
    print('Finished......')
    fig1 = plot_embedding_2D(result_2D, label_list, f'valid_epoch{epoch}')    # 将二维数据用plt绘制出来
    fig1.show()
    plt.savefig(f'./mtlonetaskSUNI0302/image_tsne/valid_{bpp}_{epoch}.jpg')
    #plt.pause(50)
    n_samples, n_features = feature_list.shape
    plt.close(fig1)

    return avg_acc


def train1(model,
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
    best_model_state_path = './SUNI04Best_Model04.pth'

    for bpp, data_paths in bpp_datasets.items():
        print(f"Training on bpp={bpp} dataset...")
        highest_accuracy = 0.0
        total_loss = 0
        total_acc = 0
        lr_adjust_step = 0
        model_save_cnt = 0
        writer = SummaryWriter(log_dir=f'./tflogs/mtlonetaskSUNI{bpp}')
        train_loader, valid_loader = generate_data(data_paths, batch_size)
        print('model down successfully')
        if best_model_state_path:
            model_state_dict = torch.load(best_model_state_path)
            model.load_state_dict(model_state_dict)
            print("Loaded model from", best_model_state_path)

        model.to(device)
        acc_history = []
        converged = False
        train_step = 0

        for epoch in range(1, EPOCHS + 1):
            print('start-epoch:%d||train_step:%d' % (epoch, train_step))
            if converged:
                break

            for index, (images, labels) in enumerate(train_loader):
                train_step += 1

                images = images.view(-1, 256, 256, 1).to(device)
                labels = labels.view(-1, 1).to(device)
                labels = torch.squeeze(labels)
                # print('images.shape: ', images.shape, 'labels.shape: ', labels.shape)

                optimizer.zero_grad()
                model_output,feature_temp = model(images)
                loss = criterion(model_output, labels)
                # loss = output_loss + reg_loss(model)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                # 计算accuracy
                model_label = torch.max(model_output, dim=1)[1]
                temp_acc = torch.eq(model_label, labels).sum().item()
                total_acc += temp_acc

                if train_step % valid_interval == 0:
                    # 在validation set上测试一次
                    v_acc = validation(model, valid_loader, device, criterion, writer, train_step,epoch,bpp)
                    acc_history.append(v_acc)
                    # 检查是否收敛

                    if v_acc > 0.85 and len(acc_history) >= 20 and max(acc_history[-20:]) - min(
                            acc_history[-20:]) < 0.01:
                        converged = True
                        break
                       

                        # 更新最高准确率和最佳模型状态
                    if v_acc > highest_accuracy:
                        highest_accuracy = v_acc
                        best_model_state = copy.deepcopy(model.state_dict())

                if train_step % write_interval == 0:
                    # 在writer中保存accuracy和loss的值
                    temp_acc = total_acc / (write_interval * images.shape[0])
                    temp_loss = total_loss / (write_interval * images.shape[0])
                    print('EPOCH: %d/%d || currIter: %d || train_loss: %.6f || train_acc: %.6f' %
                          (epoch, EPOCHS, train_step, temp_loss, temp_acc))

                    writer.add_scalar(f'train_accuracy_{bpp}', temp_acc, epoch)
                    writer.add_scalar(f'train_loss_{bpp}', temp_loss, epoch)
                    total_loss = 0
                    total_acc = 0

                if train_step % save_interval == 0:
                    # 保存模型
                    torch.save(model.state_dict(), f'./mtlonetaskSUNI0302/Model_{bpp}_{train_step}.pth')
                    model_save_cnt += 1


                if epoch == 80 or train_step == 120:
                    lr_adjust_step += 1
                    adjust_learning_rate(optimizer, 0.1, lr_adjust_step)

        if best_model_state:
            torch.save(best_model_state,
                       f'./mtlonetaskSUNI0302/Mo_onetask_SUNI_best/bestModel_{bpp}_{train_step}_{highest_accuracy:.4f}.pth')
            best_model_state_path = f'./mtlonetaskSUNI0302/Mo_onetask_SUNI_best/bestModel_{bpp}_{train_step}_{highest_accuracy:.4f}.pth'
            print(f'Saved best model with accuracy {highest_accuracy:.4f} for bpp {bpp}')
            


    writer.close()




def test(model, test_loader, device, weight_path=None):

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


def test1(model, test_loader, device, criterion,weight_path=None):
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

            model_output,feature_temp = model(images)

            feature_list.append(feature_temp)
            label_list.append(labels)

            loss = criterion(model_output, labels)
            total_loss += loss.item()

            # 计算accuracy
            model_label = torch.max(model_output, dim=1)[1]
            temp_acc = torch.eq(model_label, labels).sum().item()
            total_acc += temp_acc

        avg_acc = total_acc / cnt
        avg_loss = total_loss / cnt

        print('Test Loss: {:.4f}, Test Accuracy: {:.4f}'.format(avg_loss, avg_acc))
    print('Begining TSNE......')
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0)
    feature_list = torch.cat(feature_list,axis = 0).cpu().numpy()
    label_list = torch.cat(label_list,axis = 0).cpu().numpy()
    n_samples, n_features = feature_list.shape
    print("n_samples: ", n_samples)  #[4000]
    print("n_features: ", n_features)  #[256]
    result_2D = tsne_2D.fit_transform(feature_list)
    print('Finished......')
    fig1 = plot_embedding_2D(result_2D, label_list, f'test_suni_0.2')    # 将二维数据用plt绘制出来
    fig1.show()
    plt.savefig(f'./test_image/suni02.jpg')
    #plt.pause(50)
    n_samples, n_features = feature_list.shape
    return avg_acc




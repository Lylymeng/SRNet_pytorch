import torch
import random
import copy
from torch.utils.tensorboard import SummaryWriter

#调整优化器的学习率，lr*gamma的global_step次方
def adjust_learning_rate(optimizer, gamma, global_step):
    lr = 1e-3 * (gamma ** global_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#验证
def validation(model, valid_loader, device, criterion, writer, train_step):
    cnt = 0  # validation中所有图片的数量
    total_acc = 0  # 所有图片的accuracy的和
    total_loss = 0  # 所有图片的loss的和
    model.to(device)
    #设置为评估模式
    model.eval()
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.view(-1, 256, 256, 1).to(device)
            labels = labels.view(-1, 1).to(device)
            labels = torch.squeeze(labels)
            cnt += images.shape[0]
            #model_output1代表隐写任务的输出
            model_output1,model_output2 = model(images)
            #计算损失
            loss = criterion(model_output1, labels)
            total_loss += loss.item()
            # 计算accuracy
            model_label = torch.max(model_output1, dim=1)[1]
            #比较任务1输出最大值的索引与隐写任务的标签
            temp_acc = torch.eq(model_label, labels).sum().item()
            total_acc += temp_acc
        #计算平均准确率与损失
        avg_acc = total_acc / cnt
        avg_loss = total_loss / cnt
        writer.add_scalar('valid_accuracy', avg_acc, train_step)
        writer.add_scalar('valid_loss', avg_loss, train_step)
        print('__________currIter: %d || valid_loss: %.6f || valid_acc: %.6f__________' % (train_step, avg_loss, avg_acc))
        return avg_acc


def train(model,
          train_loader,
          valid_loader,
          optimizer,
          criterion,
          device,
          EPOCHS,
          valid_interval=5000,
          save_interval=5000,
          write_interval=100,
          load_path=None):

    model.to(device)
    total_loss = 0
    total_loss1 = 0
    total_loss2 = 0
    
    total_task1_acc = 0
    total_task2_acc = 0
    lr_adjust_step = 0
    model_save_cnt = 0
    #生成tensorboard文件
    writer = SummaryWriter(log_dir='3x3_0.4bpp_HUGO')
    # 加载之前的模型
    if load_path is not None:
        print("_________load model______________")
        model.load_state_dict(torch.load(load_path))
    best_model_state_path = None    
    highest_accuracy = 0.0
    acc1_history = []
    converged = False #设置收敛标志
    train_step = 0
    for epoch in range(1, EPOCHS + 1):
        print('start-epoch:%d||train_step:%d'%(epoch,train_step))
        if converged: #如果收敛就跳出循环
            break
        for index, (images, labels) in enumerate(train_loader):
            train_step += 1
            images = images.view(-1, 256, 256, 1).to(device)
            labels = labels.view(64,-1).to(device)
            labels = torch.squeeze(labels)
            #print('images.shape: ', images.shape, 'labels.shape: ', labels.shape)
            #print('labels.center',labels)
            optimizer.zero_grad()
            main_task_output,aux_task_output=model(images)
            #print(main_task_output)
            #任务1索引，得出0、1值来判断是否隐写
            task1_label = torch.max(main_task_output, dim=1)[1]
            #print('task1_label',task1_label)
            #任务2索引，得出0、1值来判断是否滤波
            task2_label = torch.max(aux_task_output,dim=1)[1]
            # 计算任务1的准确率
            task1_acc = torch.eq(task1_label, labels[:, 0]).sum().item()#对比任务1索引与训练集标签的第一列
            total_task1_acc += task1_acc
            # 计算任务2的准确率
            task2_acc = torch.eq(task2_label, labels[:, 1]).sum().item()#对比任务2索引与训练集标签的第二列
            total_task2_acc += task2_acc
            #计算损失值
            loss1=criterion(main_task_output,labels[:,0])
            loss2=criterion(aux_task_output,labels[:,1])
            #固定权重方法
            loss=0.9*loss1 + 0.1*loss2     
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()

            if train_step % valid_interval == 0:
                # 在validation set上测试一次
                v_acc1=validation(model, valid_loader, device, criterion, writer, train_step)
                #存储测试准确率
                acc1_history.append(v_acc1)
                # 检查是否收敛，准确率连续25次大于0.85且波动不超过0.01
                if v_acc1 > 0.85 and len(acc1_history) >= 25 and max(acc1_history[-25:]) - min(acc1_history[-25:]) < 0.01:
                    #若收敛，将收敛标志置为true，跳出循环
                    converged = True
                    break
                # 若不收敛，更新最高准确率和最佳模型状态
                if v_acc1 > highest_accuracy:
                    highest_accuracy = v_acc1
                    best_model_state = copy.deepcopy(model.state_dict())

            if train_step % write_interval == 0:
                # 在writer中保存accuracy和loss的值
                temp_acc_1 = total_task1_acc / (write_interval*images.shape[0])
                temp_loss1=total_loss1 / (write_interval * images.shape[0])
                print('EPOCH: %d/%d || currIter: %d || train_loss_task1: %.6f || train_acc_task1: %.6f' %
                      (epoch, EPOCHS, train_step, temp_loss1, temp_acc_1))
                writer.add_scalar('train_accuracy_task1', temp_acc_1, train_step)
                writer.add_scalar('train_loss_task1', temp_loss1, train_step)
                total_loss1 = 0
                total_loss2 = 0
                total_task1_acc = 0
                total_task2_acc = 0
                total_loss = 0
                
            if train_step % save_interval == 0:
                # 保存模型
                torch.save(model.state_dict(), './Mo_3x3_0.4bpp_HUGO/Model_' + str(train_step) + '.pth')
                model_save_cnt += 1

            if epoch == 80 or train_step == 120:
                lr_adjust_step += 1
                adjust_learning_rate(optimizer, 0.1, lr_adjust_step)
    #收敛后，保存最佳模型            
    if best_model_state:
        torch.save(best_model_state,
                       f'./Mo_3x3_0.4bpp_HUGO_best/bestModel_{train_step}_{highest_accuracy:.4f}.pth')
        best_model_state_path = f'./Mo_3x3_0.4bpp_HUGO_best/bestModel_{train_step}_{highest_accuracy:.4f}.pth'
        print(f'Saved best model with accuracy {highest_accuracy:.4f} ')

    writer.close()


def test(model, test_loader, device, criterion,weight_path=None):
    model.to(device)
    # 加载之前训练好的模型
    assert weight_path is not None, 'weight_path is None, please change weight_path'
    model.load_state_dict(torch.load(weight_path))
    cnt = 0  # 测试集中所有图片的数量
    total_acc = 0  # 所有图片的accuracy的总和
    total_loss = 0  # 所有图片的loss的总和
    model.to(device)

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 256, 256, 1).to(device)
            labels = labels.view(-1, 1).to(device)
            labels = torch.squeeze(labels)

            cnt += images.shape[0]

            model_output1,model_output2 = model(images)
            loss = criterion(model_output1, labels)
            total_loss += loss.item()
            # 计算accuracy
            model_label = torch.max(model_output1, dim=1)[1]
            temp_acc = torch.eq(model_label, labels).sum().item()
            total_acc += temp_acc

        avg_acc = total_acc / cnt
        avg_loss = total_loss / cnt

        print('test_loss: %.6f || test_acc: %.6f' % (avg_loss, avg_acc))
        return avg_acc





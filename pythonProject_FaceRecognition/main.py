# --------------------------------------------------------------------------------------------------------------------------------------------------
# 作者：       赵敏琨
# 日期：       2021年6月
# 说明：
#     预训练模型迁移学习：加载相关预训练模型，并可视化输出网络的结构，能够利用模型的不同层和相关预训练模型的参数值实现图像的特征提取；
#     特征可视化：自行选择一张图片，将其通过卷积神经网络进行特征提取，对图片经过不同层次网络得到的特征进行可视化，对比分析低层特征和高层特征的特点。
# --------------------------------------------------------------------------------------------------------------------------------------------------

# Reference: https://blog.csdn.net/jiaoyangwm/article/details/80011656 卷积神经网络超详细介绍
# Reference: https://blog.csdn.net/weixin_41278720/article/details/80759933 pytorch学习笔记之加载预训练模型
# Reference: https://zhuanlan.zhihu.com/p/25980324 PyTorch预训练
# Reference: https://dl.apachecn.org/#/docs/pt-tut-17/20 计算机视觉的迁移学习教程
# Reference: https://blog.csdn.net/github_28260175/article/details/103436020 PyTorch2ONNX2TensorRT 踩坑日志
# Reference: https://blog.csdn.net/nan355655600/article/details/106245563 网络可视化工具netron详细安装流程
# Reference: https://blog.csdn.net/qq_41167777/article/details/109013155 利用pytorch训练好的模型测试单张图片
# Reference: https://blog.csdn.net/LXX516/article/details/80132228 pytorch模型中间层特征的提取
# Reference: https://blog.csdn.net/zhangphil/article/details/103599615 CNN神经网络猫狗分类经典案例，深度学习过程中间层激活特征图可视化
# Reference: https://blog.csdn.net/weixin_42782150/article/details/107015617 Python中使用plt.ion()和plt.ioff()画动态图
# Reference: https://blog.csdn.net/github_39172337/article/details/82463184 深度学习模型的可视化技术总结
# Reference: https://blog.csdn.net/qxqsunshine/article/details/108020963 深度学习特征图可视化
# Reference: https://blog.csdn.net/qq_34789262/article/details/82904762 Inception v3 输入图片尺寸大小
# Reference: https://blog.csdn.net/sinat_33487968/article/details/83622128 【pytorch torchvision源码解读系列—3】Inception V3
# Reference: https://blog.csdn.net/mathlxj/article/details/105136643 Pytorch源码学习之七：torchvision.models.googlenet
# Reference: https://blog.csdn.net/weixin_44538273/article/details/88856239 shuffle-net的pytorch代码
# Reference: https://blog.csdn.net/scut_salmon/article/details/82391320 torch x = x.view(-1, ...)理解
# Referebce: https://blog.csdn.net/qq_31347869/article/details/100566719 Pytorch实现ResNet代码解析


from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from PIL import Image
import time
import os
import copy

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #   ----------任务选择----------
    print("预训练模型迁移学习，请输入'PM'")  # Pre-Trained Model
    print("特征可视化，请输入'FV'")  # Feature Visualization
    while 1:
        TASK = input('请选择任务：')
        if TASK == 'PM' or TASK == 'pm':
            break
        elif TASK == 'FV' or TASK == 'fv':
            break
        else:
            print('非法，请重新输入')

    #   ----------任务2----------
    if TASK == 'PM' or TASK == 'pm':

        plt.ion()  # interactive mode

        # #########Choose Net######### #
        print("选择网络ResNet，请输入：'res'")
        print("选择网络VGGNet，请输入：'vgg'")
        print("选择网络AlexNet，请输入：'alex'")
        print("选择网络GoogLeNet，请输入：'google'")
        print("选择网络InceptionNet，请输入：'inception'")
        print("选择网络DenseNet，请输入：'dense'")
        print("选择网络SqueezeNet，请输入：'squeeze'")
        print("选择网络MobileNet，请输入：'mobile'")
        print("选择网络ShuffleNet，请输入：'shuffle'")
        print("选择网络MNASNet，请输入：'mnas'")
        while 1:
            NetName = input('请选择网络：')
            if NetName == 'res':
                NetNum = input('请输入编号：')
                if NetNum != '18' and NetNum != '34' and NetNum != '152':
                    print('非法，请重新输入')
                else:
                    break
            elif NetName == 'vgg':
                NetNum = input('请输入编号：')
                if NetNum != '16' and NetNum != '19':
                    print('非法，请重新输入')
                else:
                    break
            elif NetName == 'alex':
                break
            elif NetName == 'google':
                break
            elif NetName == 'inception':
                break
            elif NetName == 'dense':
                NetNum = input('请输入编号：')
                if NetNum != '121' and NetNum != '161':
                    print('非法，请重新输入')
                else:
                    break
            elif NetName == 'squeeze':
                break
            elif NetName == 'mobile':
                break
            elif NetName == 'shuffle':
                break
            elif NetName == 'mnas':
                break
            else:
                print('非法，请重新输入')

        # Data augmentation and normalization for training
        # Just normalization for validation
        # Attention: torchvision model inception v3 needs 299x299 size image
        if NetName == 'inception':
            resize_train = 299
            resize_test = 299
        else:
            resize_train = 224
            resize_test = 224

        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize((resize_train, resize_train)),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.442, 0.442, 0.442], [0.196, 0.196, 0.196])
            ]),
            'val': transforms.Compose([
                transforms.Resize(resize_test),
                # transforms.CenterCrop(resize_train),
                transforms.ToTensor(),
                transforms.Normalize([0.442, 0.442, 0.442], [0.196, 0.196, 0.196])
            ]),
        }

        data_dir = 'data/eigfaces_aug'
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                  data_transforms[x])
                          for x in ['train', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                     shuffle=True)
                      for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        class_names = image_datasets['train'].classes

        ######################################################################
        # Visualize a few images
        # ^^^^^^^^^^^^^^^^^^^^^^
        #

        def imshow(inp, title=None):
            """Imshow for Tensor."""
            inp = inp.numpy().transpose((1, 2, 0))
            mean = np.array([0.442, 0.442, 0.442])
            std = np.array([0.196, 0.196, 0.196])
            inp = std * inp + mean
            inp = np.clip(inp, 0, 1)
            plt.imshow(inp)
            if title is not None:
                plt.title(title)
            plt.pause(0.001)  # pause a bit so that plots are updated


        # Get a batch of training data
        inputs, classes = next(iter(dataloaders['train']))

        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)

        imshow(out, title=[class_names[x] for x in classes])


        ######################################################################
        # Training the model
        # ------------------
        #
        # Write a general function to train a model. Here, I will illustrate:
        #
        # -  Scheduling the learning rate
        # -  Saving the best model
        #
        # In the following, parameter ``scheduler`` is an LR scheduler object from
        # ``torch.optim.lr_scheduler``.


        def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
            since = time.time()

            best_model_wts = copy.deepcopy(model.state_dict())
            best_acc = 0.0

            fig_train, axs_train = plt.subplots(1, 2)
            fig_val, axs_val = plt.subplots(1, 2)
            fig_train.suptitle('Train Process')
            fig_val.suptitle('Val Process')
            xdata = range(num_epochs)
            loss_train, acc_train = [], []
            loss_val, acc_val = [], []

            for epoch in range(num_epochs):
                print('Epoch {}/{}'.format(epoch, num_epochs - 1))
                print('-' * 10)

                # Each epoch has a training and validation phase
                for phase in ['train', 'val']:
                    if phase == 'train':
                        model.train()  # Set model to training mode
                    else:
                        model.eval()   # Set model to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0

                    # Iterate over data.
                    for inputs, labels in dataloaders[phase]:
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            # Attention, models.inception_v3's outputs include
                            # x and auxs, two dims;
                            # Though, models.googlenet's
                            # outputs is (x, aux), one dim.
                            if phase == 'train' and NetName == 'inception':
                                outputs = model(inputs)[0]
                            else:
                                outputs = model(inputs)
                            _, preds = torch.max(outputs, 1)

                            loss = criterion(outputs, labels)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                    if phase == 'train':
                        scheduler.step()

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects.cpu() / dataset_sizes[phase]

                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                        phase, epoch_loss, epoch_acc))

                    if phase == 'train':
                        loss_train.append(epoch_loss)
                        acc_train.append(epoch_acc)
                    else:
                        loss_val.append(epoch_loss)
                        acc_val.append(epoch_acc)

                    # deep copy the model
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())

                print()

            axs_train[0].grid()
            axs_train[0].plot(xdata, loss_train, '.-')
            axs_train[0].set_title('Loss Curve')
            axs_train[1].grid()
            axs_train[1].plot(xdata, acc_train, '.-')
            axs_train[1].set_title('Accuracy Curve')
            axs_train[0].set_ylabel('Value')
            axs_train[0].set_xlabel("Epoch")
            axs_train[1].set_xlabel("Epoch")
            axs_val[0].grid()
            axs_val[0].plot(xdata, loss_val, '.-')
            axs_val[0].set_title('Loss Curve')
            axs_val[1].grid()
            axs_val[1].plot(xdata, acc_val, '.-')
            axs_val[1].set_title('Accuracy Curve')
            axs_val[0].set_ylabel("Value")
            axs_val[0].set_xlabel("Epoch")
            axs_val[1].set_xlabel("Epoch")

            time_elapsed = time.time() - since
            print('Training complete in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))
            print('Best val Acc: {:4f}'.format(best_acc))

            # load best model weights
            model.load_state_dict(best_model_wts)
            return model


        ######################################################################
        # Visualizing the model predictions
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        #
        # Generic function to display predictions for a few images
        #

        def visualize_model(model, num_images=6):
            was_training = model.training
            model.eval()
            images_so_far = 0
            fig = plt.figure()

            with torch.no_grad():
                for i, (inputs, labels) in enumerate(dataloaders['val']):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)

                    for j in range(inputs.size()[0]):
                        images_so_far += 1
                        ax = plt.subplot(num_images//2, 2, images_so_far)
                        ax.axis('off')
                        ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                        imshow(inputs.cpu().data[j])

                        if images_so_far == num_images:
                            model.train(mode=was_training)
                            return model.train(mode=was_training)

        ######################################################################
        # Finetuning the convnet
        # ----------------------
        #
        # Load a pretrained model and reset final fully connected layer.
        #

        if NetName == 'res' and NetNum == '18':
            model_ft = models.resnet18(pretrained=True)
        elif NetName == 'res' and NetNum == '34':
            model_ft = models.resnet34(pretrained=True)
        elif NetName == 'res' and NetNum == '152':
            model_ft = models.resnet152(pretrained=True)
        elif NetName == 'vgg' and NetNum == '16':
            model_ft = models.vgg16(pretrained=True)
        elif NetName == 'vgg' and NetNum == '19':
            model_ft = models.vgg19(pretrained=True)
        elif NetName == 'alex':
            model_ft = models.alexnet(pretrained=True)
        elif NetName == 'google':
            model_ft = models.googlenet(pretrained=True)
        elif NetName == 'inception':
            model_ft = models.inception_v3(pretrained=True)
        elif NetName == 'dense' and NetNum == '121':
            model_ft = models.densenet121(pretrained=True)
        elif NetName == 'dense' and NetNum == '161':
            model_ft = models.densenet161(pretrained=True)
        elif NetName == 'squeeze':
            model_ft = models.squeezenet1_1(pretrained=True)
        elif NetName == 'mobile':
            model_ft = models.mobilenet_v2(pretrained=True)
        elif NetName == 'shuffle':
            model_ft = models.shufflenet_v2_x1_0(pretrained=True)
        elif NetName == 'mnas':
            model_ft = models.mnasnet1_0(pretrained=True)

        if NetName == 'res' or NetName == 'google' or NetName == 'inception':
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, 40)
        elif NetName == 'vgg' or NetName == 'alex':
            num_ftrs = model_ft.classifier._modules['6'].in_features
            # The full connection layers of vgg/alex net are in classifier module, needed to be changed
            model_ft.classifier._modules['6'] = nn.Linear(num_ftrs, 40)
        elif NetName == 'dense':
            num_ftrs = model_ft.classifier.in_features
            # The full connection layers of dense net are in classifier module, needed to be changed
            model_ft.classifier = nn.Linear(num_ftrs, 40)
        elif NetName == 'squeeze':
            num_ftrs = model_ft.classifier._modules['1'].in_channels
            tuple_kernel = model_ft.classifier._modules['1'].kernel_size
            tuple_stride = model_ft.classifier._modules['1'].stride
            # The squeeze net has no full connection layers, needed to be changed
            model_ft.classifier._modules['1'] = nn.Conv2d(num_ftrs, 40, tuple_kernel, tuple_stride)
        elif NetName == 'mobile' or NetName == 'mnas':
            num_ftrs = model_ft.classifier._modules['1'].in_features
            # The mobile/mnas net has no full connection layers, needed to be changed
            model_ft.classifier._modules['1'] = nn.Linear(num_ftrs, 40)
        elif NetName == 'shuffle':
            num_ftrs = model_ft._modules['fc'].in_features
            model_ft._modules['fc'] = nn.Linear(num_ftrs, 40)

            # Here the size of each output sample is set to 2.
            # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).

        model_ft = model_ft.to(device)

        criterion = nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        ######################################################################
        # Train and evaluate
        # ^^^^^^^^^^^^^^^^^^
        #
        # It takes about two minute on GPU.
        #

        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                               num_epochs=13)

        ######################################################################
        #

        visualize_model(model_ft)


        ######################################################################
        # ConvNet as fixed feature extractor
        # ----------------------------------
        #
        # Here, we need to freeze all the network except the final layer. We need
        # to set ``requires_grad == False`` to freeze the parameters so that the
        # gradients are not computed in ``backward()``.
        #
        # Read more about this in the documentation
        # `here <https://pytorch.org/docs/notes/autograd.html#excluding-subgraphs-from-backward>`__.
        #

        # if NetName == 'res' and NetNum == '18':
        #     model_conv = models.resnet18(pretrained=True)
        # elif NetName == 'res' and NetNum == '34':
        #     model_conv = models.resnet34(pretrained=True)
        # elif NetName == 'res' and NetNum == '152':
        #     model_conv = models.resnet152(pretrained=True)
        # elif NetName == 'vgg' and NetNum == '16':
        #     model_conv = models.vgg16(pretrained=True)
        # elif NetName == 'vgg' and NetNum == '19':
        #     model_conv = models.vgg19(pretrained=True)
        # elif NetName == 'alex':
        #     model_conv = models.alexnet(pretrained=True)
        # elif NetName == 'google':
        #     model_conv = models.googlenet(pretrained=True)
        # elif NetName == 'inception':
        #     model_conv = models.inception_v3(pretrained=True)
        # elif NetName == 'dense' and NetNum == '121':
        #     model_conv = models.densenet121(pretrained=True)
        # elif NetName == 'dense' and NetNum == '161':
        #     model_conv = models.densenet161(pretrained=True)
        # elif NetName == 'squeeze':
        #     model_conv = models.squeezenet1_1(pretrained=True)
        # elif NetName == 'mobile':
        #     model_conv = models.mobilenet_v2(pretrained=True)
        # elif NetName == 'shuffle':
        #     model_conv = models.shufflenet_v2_x1_0(pretrained=True)
        # elif NetName == 'mnas':
        #     model_conv = models.mnasnet1_0(pretrained=True)
        #
        # for param in model_conv.parameters():
        #     param.requires_grad = False
        #
        # # Parameters of newly constructed modules have requires_grad=True by default
        # if NetName == 'res' or NetName == 'google' or NetName == 'inception':
        #     num_ftrs = model_conv.fc.in_features
        #     model_conv.fc = nn.Linear(num_ftrs, 40)  # res/google/inception net
        # elif NetName == 'vgg' or NetName == 'alex':
        #     num_ftrs = model_conv.classifier._modules['6'].in_features
        #     model_conv.classifier._modules['6'] = nn.Linear(num_ftrs, 40)    # vgg/alex net
        # elif NetName == 'dense':
        #     num_ftrs = model_conv.classifier.in_features
        #     model_conv.classifier = nn.Linear(num_ftrs, 40)  # dense net
        # elif NetName == 'squeeze':
        #     num_ftrs = model_conv.classifier._modules['1'].in_channels
        #     tuple_kernel = model_conv.classifier._modules['1'].kernel_size
        #     tuple_stride = model_conv.classifier._modules['1'].stride
        #     model_conv.classifier._modules['1'] = nn.Conv2d(num_ftrs, 40, tuple_kernel, tuple_stride)    # squeeze net
        # elif NetName == 'mobile' or NetName == 'mnas':
        #     num_ftrs = model_conv.classifier._modules['1'].in_features
        #     model_conv.classifier._modules['1'] = nn.Linear(num_ftrs, 40)  # mobile/mnas net
        # elif NetName == 'shuffle':
        #     num_ftrs = model_conv._modules['fc'].in_features
        #     model_conv._modules['fc'] = nn.Linear(num_ftrs, 40)  # shuffle net
        #
        # model_conv = model_conv.to(device)
        #
        # criterion = nn.CrossEntropyLoss()
        #
        # # Observe that only parameters of final layer are being optimized as
        # # opposed to before.
        # if NetName == 'res' or NetName == 'google' or NetName == 'inception':
        #     optimizer_conv = optim.SGD(model_conv.fc.parameters(),
        #                                lr=0.001, momentum=0.9)
        # elif NetName == 'vgg' or NetName == 'alex':
        #     optimizer_conv = optim.SGD(
        #         model_conv.classifier._modules['6'].parameters(),
        #         lr=0.001, momentum=0.9)
        # elif NetName == 'dense':
        #     optimizer_conv = optim.SGD(model_conv.classifier.parameters(),
        #                                lr=0.001, momentum=0.9)
        # elif NetName == 'squeeze' or NetName == 'mobile' or NetName == 'mnas':
        #     optimizer_conv = optim.SGD(
        #         model_conv.classifier._modules['1'].parameters(),
        #         lr=0.001, momentum=0.9)
        # elif NetName == 'shuffle':
        #     optimizer_conv = optim.SGD(
        #         model_conv._modules['fc'].parameters(),
        #         lr=0.001, momentum=0.9)
        #
        # # Decay LR by a factor of 0.1 every 7 epochs
        # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


        ######################################################################
        # Train and evaluate
        # ^^^^^^^^^^^^^^^^^^
        #
        # On CPU this will take about half the time compared to previous scenario.
        # This is expected as gradients don't need to be computed for most of the
        # network. However, forward does need to be computed.
        #

        # model_conv = train_model(model_conv, criterion, optimizer_conv,
        #                          exp_lr_scheduler, num_epochs=13)

        # visualize_model(model_conv)

        ######################################################################
        #
        while 1:
            SAVEFLAG = input('请输入迁移学习训练网络是否保存(ON/OFF)：')
            if SAVEFLAG == 'ON' or SAVEFLAG == 'on':
                break
            elif SAVEFLAG == 'OFF' or SAVEFLAG == 'off':
                break
            else:
                print('非法，请重新输入')

        # Process above shows two transfer learning methods: 1) FineTuning; 2) ConvFrozen.
        # Normally, training time of the first way is less than the second way.
        # The model of the first way is 'model_ft', and the second way is 'model_conv'.
        # Transfer learning have not change the architecture of Net Model,
        # It only changes parameters of that.
        # So, I only use 'model_ft' to export the ONNX model.

        if SAVEFLAG == 'ON' or SAVEFLAG == 'on':
            ff = torch.randn(4, 3, 224, 224)
            ff = ff.type(torch.cuda.FloatTensor)

            if NetName == 'res' and NetNum == '18':
                torch.save(model_ft, './models/resnet18_ft_TL.pth')
                torch.onnx.export(model_ft, ff, './nets/resnet18_TL.onnx')
            elif NetName == 'res' and NetNum == '34':
                torch.save(model_ft, './models/resnet34_ft_TL.pth')
                torch.onnx.export(model_ft, ff, './nets/resnet34_TL.onnx')
            elif NetName == 'res' and NetNum == '152':
                torch.save(model_ft, './models/resnet152_ft_TL.pth')
                torch.onnx.export(model_ft, ff, './nets/resnet152_TL.onnx')
            elif NetName == 'vgg' and NetNum == '16':
                torch.save(model_ft, './models/vgg16_ft_TL.pth')
                torch.onnx.export(model_ft, ff, './nets/vgg16_TL.onnx', operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
            elif NetName == 'vgg' and NetNum == '19':
                torch.save(model_ft, './models/vgg19_ft_TL.pth')
                torch.onnx.export(model_ft, ff, './nets/vgg19_TL.onnx', operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
            elif NetName == 'alex':
                torch.save(model_ft, './models/alexnet_ft_TL.pth')
                torch.onnx.export(model_ft, ff, './nets/alexnet_TL.onnx', operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
            elif NetName == 'google':
                torch.save(model_ft, './models/googlenet_ft_TL.pth')
                torch.onnx.export(model_ft, ff, './nets/googlenet_TL.onnx')
            elif NetName == 'inception':
                torch.save(model_ft, './models/inception_ft_TL.pth')
                torch.onnx.export(model_ft, ff, './nets/inception_TL.onnx')
            elif NetName == 'dense' and NetNum == '121':
                torch.save(model_ft, './models/densenet121_ft_TL.pth')
                torch.onnx.export(model_ft, ff, './nets/densenet121_TL.onnx')
            elif NetName == 'dense' and NetNum == '161':
                torch.save(model_ft, './models/densenet161_ft_TL.pth')
                torch.onnx.export(model_ft, ff, './nets/densenet161_TL.onnx')
            elif NetName == 'squeeze':
                torch.save(model_ft, './models/squeezenet_ft_TL.pth')
                torch.onnx.export(model_ft, ff, './nets/squeezenet_TL.onnx')
            elif NetName == 'mobile':
                torch.save(model_ft, './models/mobilenet_ft_TL.pth')
                torch.onnx.export(model_ft, ff, './nets/mobilenet_TL.onnx')
            elif NetName == 'shuffle':
                torch.save(model_ft, './models/shufflenet_ft_TL.pth')
                torch.onnx.export(model_ft, ff, './nets/shufflenet_TL.onnx')
            elif NetName == 'mnas':
                torch.save(model_ft, './models/mnasnet_ft_TL.pth')
                torch.onnx.export(model_ft, ff, './nets/mnasnet_TL.onnx')
            print('网络已保存')
        else:
            print('保存已关闭')


        plt.ioff()
        plt.show()


        ######################################################################
        # Further Learning
        # -----------------
        #


    #   ----------任务3----------
    if TASK == 'FV' or TASK == 'fv':

        plt.ion()  # interactive mode

        # #########Choose Net######### #
        print("选择网络ResNet，请输入：'res'")
        print("选择网络VGGNet，请输入：'vgg'")
        print("选择网络AlexNet，请输入：'alex'")
        print("选择网络GoogLeNet，请输入：'google'")
        print("选择网络InceptionNet，请输入：'inception'")
        print("选择网络DenseNet，请输入：'dense'")
        print("选择网络SqueezeNet，请输入：'squeeze'")
        print("选择网络MobileNet，请输入：'mobile'")
        print("选择网络ShuffleNet，请输入：'shuffle'")
        print("选择网络MNASNet，请输入：'mnas'")
        while 1:
            NetName = input('请选择网络：')
            if NetName == 'res':
                my_model = torch.load('./models/resnet34_ft_TL.pth')
                break
            elif NetName == 'vgg':
                my_model = torch.load('./models/vgg16_ft_TL.pth')
                break
            elif NetName == 'alex':
                my_model = torch.load('./models/alexnet_ft_TL.pth')
                break
            elif NetName == 'google':
                my_model = torch.load('./models/googlenet_ft_TL.pth')
                break
            elif NetName == 'inception':
                my_model = torch.load('./models/inception_ft_TL.pth')
                break
            elif NetName == 'dense':
                my_model = torch.load('./models/densenet121_ft_TL.pth')
                break
            elif NetName == 'squeeze':
                my_model = torch.load('./models/squeezenet_ft_TL.pth')
                break
            elif NetName == 'mobile':
                my_model = torch.load('./models/mobilenet_ft_TL.pth')
                break
            elif NetName == 'shuffle':
                my_model = torch.load('./models/shufflenet_ft_TL.pth')
                break
            elif NetName == 'mnas':
                my_model = torch.load('./models/mnasnet_ft_TL.pth')
                break
            else:
                print('非法，请重新输入')

        # Load Transfer Learning trained model from local path
        my_model.to(device)
        my_model.eval()

        data_dir = 'data/eigfaces_aug'
        image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x))
                          for x in ['train', 'val']}
        class_names = image_datasets['train'].classes

        # img_dir = './data/fvtest_data/a.jpg'
        # img_dir = './data/fvtest_data/b.jpg'
        # img_dir = './data/fvtest_data/f.jpg'
        # img_dir = './data/fvtest_data/N_Face_Gray.png'
        # img_dir = './data/fvtest_data/My_Face_Gray.jpg'
        # FILENUM = str(16)
        # FILENUM = str(10)
        FILENUM = str(np.random.randint(0, 40, 1)[0])
        # img_dir = './data/fvtest_data/eigfaces/' + FILENUM + '.jpg'
        # img_dir = './data/fvtest_data/eigfaces_noise_SP1/' + FILENUM + '.jpg'  #Salt & Pepper Noise 1% density
        img_dir = './data/fvtest_data/eigfaces_noise_SP3/' + FILENUM + '.jpg'  #Salt & Pepper Noise 3% density
        # img_dir = './data/fvtest_data/eigfaces_noise_SP5/' + FILENUM + '.jpg'  #Salt & Pepper Noise 5% density
        if NetName == 'inception':
            resize_train = 299
            resize_test = 299
        else:
            resize_train = 224
            resize_test = 224

        transforms_act = transforms.Compose([
                transforms.Resize(resize_test),
                # transforms.CenterCrop(resize_train),
                transforms.ToTensor(),
                transforms.Normalize([0.442, 0.442, 0.442], [0.196, 0.196, 0.196])
            ])

        img = Image.open(img_dir)
        plt.figure('Input Image')
        plt.axis('off')
        plt.imshow(img)
        plt.pause(0.001)

        img_trans = transforms_act(img).unsqueeze(0)    # Augmentation
        img_trans = img_trans.to(device)

        outputs = my_model(img_trans)

        _, preds = torch.sort(outputs, 1, descending=True)
        preds = preds[0]

        preds_top1 = class_names[preds[0]]
        preds_top3 = class_names[preds[0]], class_names[preds[1]], class_names[preds[2]]
        print('预测结果Top1:', preds_top1)
        print('预测结果Top3:', preds_top3)

        if len(preds_top1) == 7:
            num_check = preds_top1[6]
        else:
            num_check = preds_top1[6] + preds_top1[7]
        datapath_check = './data/fvtest_data/eigfaces/' + num_check + '.jpg'
        img_check = Image.open(datapath_check)
        plt.figure('Check Image')
        plt.axis('off')
        plt.title(preds_top1)
        plt.imshow(img_check)
        plt.pause(0.001)

        print('-----------------------------------------------------------------------------------------------------------')

        if NetName == 'res' or NetName == 'inception' or NetName == 'shuffle':
            module_name_list = []
            print('↓-----------------------------------------------------------------------------------------------------↓')
            print('Model Architecture:')
            for module_name in my_model._modules:
                print(module_name)
                module_name_list.append(module_name)
            print('↑-----------------------------------------------------------------------------------------------------↑')

            # Feature Extractor Class
            class FeatureExtractor(nn.Module):
                def __init__(self, submodule, extracted_layers):
                    super(FeatureExtractor, self).__init__()
                    self.submodule = submodule
                    self.extracted_layers = extracted_layers

                def forward(self, x):
                    outputs = []
                    for name, module in self.submodule._modules.items():
                        if name == "fc":
                            if NetName == 'shuffle':
                                x = x.view(-1, 1024)
                            else:
                                x = x.view(x.size(0), -1)
                        if NetName == 'inception' and name == 'AuxLogits':
                            continue
                        x = module(x)
                        # print(name)
                        if name in self.extracted_layers:
                            outputs.append(x)
                    return outputs


            exact_list = []
            while 1:
                exact = input('请输入抽取特征层名称：')
                if exact in module_name_list:
                    exact_list.append(exact)
                    while 1:
                        ctn = input('继续输入？(Y/N)：')
                        if ctn == 'Y' or ctn == 'y':
                            break
                        elif ctn == 'N' or ctn == 'n':
                            break
                        else:
                            print('非法，请重新输入')
                    if ctn == 'N' or ctn == 'n':
                        break
                else:
                    print('非法，请重新输入')

        if NetName == 'google':
            module_name_list = []
            submodule_name_list = []
            print('↓-----------------------------------------------------------------------------------------------------↓')
            print('Model Architecture:')
            for module_name in my_model._modules:
                print(module_name)
                module_name_list.append(module_name)
            print('↑-----------------------------------------------------------------------------------------------------↑')

            # Feature Extractor Class
            class FeatureExtractor(nn.Module):
                def __init__(self, submodule, extracted_layers):
                    super(FeatureExtractor, self).__init__()
                    self.submodule = submodule
                    self.extracted_layers = extracted_layers

                def forward(self, x):
                    outputs = []
                    for name, module in self.submodule._modules.items():
                        if name == 'aux1' or name == 'aux2':
                            continue
                        if name == "fc":
                            x = x.view(x.size(0), -1)
                        x = module(x)
                        # print(name)
                        if name in self.extracted_layers:
                            outputs.append(x)
                    return outputs


            exact_list = []
            while 1:
                exact = input('请输入抽取特征层名称：')
                if exact in module_name_list:
                    exact_list.append(exact)

                    while 1:
                        ctn = input('继续输入？(Y/N)：')
                        if ctn == 'Y' or ctn == 'y':
                            break
                        elif ctn == 'N' or ctn == 'n':
                            break
                        else:
                            print('非法，请重新输入')
                    if ctn == 'N' or ctn == 'n':
                        break
                else:
                    print('非法，请重新输入')

        if NetName == 'vgg' or NetName == 'alex' or NetName == 'squeeze'\
                or NetName == 'mobile':
            module_name_list = []
            print('↓-----------------------------------------------------------------------------------------------------↓')
            print('Model Architecture:')
            for module_name in my_model.features._modules:
                print(module_name, ': ', my_model.features[int(module_name)])
                module_name_list.append(module_name)
            print('↑-----------------------------------------------------------------------------------------------------↑')

            # Feature Extractor Class
            class FeatureExtractor(nn.Module):
                def __init__(self, submodule, extracted_layers):
                    super(FeatureExtractor, self).__init__()
                    self.submodule = submodule
                    self.extracted_layers = extracted_layers

                def forward(self, x):
                    outputs = []
                    for name, module in self.submodule.features._modules.items():
                        if name == "fc":
                            x = x.view(x.size(0), -1)
                        x = module(x)
                        # print(name)
                        if name in self.extracted_layers:
                            outputs.append(x)
                    return outputs


            exact_list = []
            while 1:
                exact = input('请输入抽取特征层编号：')
                print(exact)
                if exact in module_name_list:
                    exact_list.append(exact)
                    while 1:
                        ctn = input('继续输入？(Y/N)：')
                        if ctn == 'Y' or ctn == 'y':
                            break
                        elif ctn == 'N' or ctn == 'n':
                            break
                        else:
                            print('非法，请重新输入')
                    if ctn == 'N' or ctn == 'n':
                        break
                else:
                    print('非法，请重新输入')

        if NetName == 'dense':
            module_name_list = []
            print('↓-----------------------------------------------------------------------------------------------------↓')
            print('Model Architecture:')
            for module_name in my_model.features._modules:
                print(module_name)
                module_name_list.append(module_name)
            print('↑-----------------------------------------------------------------------------------------------------↑')

            # Feature Extractor Class
            class FeatureExtractor(nn.Module):
                def __init__(self, submodule, extracted_layers):
                    super(FeatureExtractor, self).__init__()
                    self.submodule = submodule
                    self.extracted_layers = extracted_layers

                def forward(self, x):
                    outputs = []
                    for name, module in self.submodule.features._modules.items():
                        if name == "fc":
                            x = x.view(x.size(0), -1)
                        x = module(x)
                        # print(name)
                        if name in self.extracted_layers:
                            outputs.append(x)
                    return outputs


            exact_list = []
            while 1:
                exact = input('请输入抽取特征层名称：')
                print(exact)
                if exact in module_name_list:
                    exact_list.append(exact)
                    while 1:
                        ctn = input('继续输入？(Y/N)：')
                        if ctn == 'Y' or ctn == 'y':
                            break
                        elif ctn == 'N' or ctn == 'n':
                            break
                        else:
                            print('非法，请重新输入')
                    if ctn == 'N' or ctn == 'n':
                        break
                else:
                    print('非法，请重新输入')

        if NetName == 'mnas':
            module_name_list = []
            print('↓-----------------------------------------------------------------------------------------------------↓')
            print('Model Architecture:')
            for module_name in my_model.layers._modules:
                print(module_name, ': ', my_model.layers[int(module_name)])
                module_name_list.append(module_name)
            print('↑-----------------------------------------------------------------------------------------------------↑')

            # Feature Extractor Class
            class FeatureExtractor(nn.Module):
                def __init__(self, submodule, extracted_layers):
                    super(FeatureExtractor, self).__init__()
                    self.submodule = submodule
                    self.extracted_layers = extracted_layers

                def forward(self, x):
                    outputs = []
                    for name, module in self.submodule.layers._modules.items():
                        if name == "fc":
                            x = x.view(x.size(0), -1)
                        x = module(x)
                        # print(name)
                        if name in self.extracted_layers:
                            outputs.append(x)
                    return outputs


            exact_list = []
            while 1:
                exact = input('请输入抽取特征层编号：')
                print(exact)
                if exact in module_name_list:
                    exact_list.append(exact)
                    while 1:
                        ctn = input('继续输入？(Y/N)：')
                        if ctn == 'Y' or ctn == 'y':
                            break
                        elif ctn == 'N' or ctn == 'n':
                            break
                        else:
                            print('非法，请重新输入')
                    if ctn == 'N' or ctn == 'n':
                        break
                else:
                    print('非法，请重新输入')

        my_exactor = FeatureExtractor(my_model, exact_list)
        x_tensor_list = my_exactor(img_trans)

        for k in range(len(exact_list)):
            x_cpu = x_tensor_list[k].data.cpu()
            print('LAYER:' + exact_list[k] + "'s", 'output shape is', x_cpu.shape)

            if x_cpu.shape[1] < 128:
                subplot_rows = 4
            elif x_cpu.shape[1] < 256:
                subplot_rows = 8
            elif x_cpu.shape[1] <= 512:
                subplot_rows = 16
            else:
                subplot_rows = 32
            subplot_cols = int(x_cpu.shape[1] / subplot_rows)

            # Feature visualization method 1: Traversal
            x_fusion = np.zeros((x_cpu.shape[2], x_cpu.shape[3]))
            plt.figure('Feature Maps of LAYER:' + exact_list[k])
            for i in range(x_cpu.shape[1]):
                ax = plt.subplot(subplot_rows, subplot_cols, i + 1)
                # ax.set_title('Sample #{}'.format(i))
                ax.axis('off')
                plt.imshow(x_cpu.data.numpy()[0, i, :, :], cmap='viridis')
                x_fusion += x_cpu.data.numpy()[0, i, :, :]

            plt.figure('Random 4 Feature Maps of LAYER:' + exact_list[k])
            x_random = np.random.randint(0, x_cpu.shape[1], 4)
            for i in range(4):
                ax = plt.subplot(2, 2, i + 1)
                ax.set_title('Sample #{}'.format(x_random[i]))
                ax.axis('off')
                plt.imshow(x_cpu.data.numpy()[0, x_random[i], :, :], cmap='viridis')

            # Feature visualization method 2: Based on traversal, feature maps fusion
            plt.figure('Fusion Feature Map of LAYER:' + exact_list[k])
            plt.imshow(x_fusion, cmap='viridis')
            plt.title(x_cpu.shape)
            plt.axis('off')
            plt.colorbar()

        plt.ioff()  # turn off interactive mode
        plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

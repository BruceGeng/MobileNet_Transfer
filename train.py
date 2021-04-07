import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from tqdm import tqdm

from MobileV3 import mobilenet_v3_large
from GhostNet import ghostnet

model_name = 'ghostnet' # 'mobilenet'
save_dir = "/apdcephfs/share_774517/shared_info/brucegeng/transfer_learning"
weight_dir = "/apdcephfs/share_774517/shared_info/brucegeng/pretrained_model"
train_dir = "/apdcephfs/share_774517/data/videoqa/fufankui_data/fufankui_train_data_v6"
val_dir = "/apdcephfs/share_774517/data/videoqa/subtype_data/video_val_data_multi"
num_classes = 18
workers = 20
batch_size = 256
epochs = 50


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.Resize((256, 256)),
                                     transforms.RandomCrop((224,224)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    train_dataset = datasets.ImageFolder(train_dir, data_transform["train"])
    train_num = len(train_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=workers)

    val_dataset = datasets.ImageFolder(val_dir, data_transform["val"])
    val_num = len(val_dataset)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=workers)

    print("using {} images for training, {} images for validation.".format(train_num,val_num))

    if model_name == 'mobilenet':
        # create model
        net = mobilenet_v3_large(num_classes = num_classes)
        model_weight_path = os.path.join(weight_dir, "mobilenet_v3_large.pth")
        print("You chose mobilenet model!")
    elif model_name == 'ghostnet':
        net = ghostnet(num_classes = num_classes)
        model_weight_path = os.path.join(weight_dir, "ghostnet_dict.pth")
        print("You chose ghostnet model!")
    else:
        print("You Must Chose A Model!!!")

    # load pretrain weights
    assert os.path.exists(model_weight_path), "file {} dose not exist.".format(model_weight_path)
    pre_weights = torch.load(model_weight_path, map_location=device)

    # delete classifier weights
    pre_dict = {k: v for k, v in pre_weights.items() if net.state_dict()[k].numel() == v.numel()}
    missing_keys, unexpected_keys = net.load_state_dict(pre_dict, strict=False)

    # freeze features weights
    if model_name == 'mobilenet':
        for param in net.features.parameters():
            param.requires_grad = False
    elif model_name == 'ghostnet':
        for param in net.blocks.parameters():
            param.requires_grad = False
    else:
        print("You Must Choose A Model!!!")

    net.to(device)

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    best_acc = 0.0
    if model_name == 'mobilenet':
        save_path = os.path.join(save_dir, 'Transfered_MobileNetV3.pth')
    elif model_name == 'ghostnet':
        save_path = os.path.join(save_dir, 'Transfered_GhostNet.pth')
    else:
        print("You Must Choose A Model!!!")

    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader)
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        # validate
        if epoch % 10 == 0 or epoch == epochs - 1:
            net.eval()
            acc = 0.0  # accumulate accurate number / epoch
            with torch.no_grad():
                val_bar = tqdm(val_loader)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outputs = net(val_images.to(device))
                    # loss = loss_function(outputs, test_labels)
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                    val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                               epochs)
            val_accurate = acc / val_num
            print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                  (epoch + 1, running_loss / train_steps, val_accurate))

            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
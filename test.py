import os
import json

import torch
from PIL import Image
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from MobileV3 import mobilenet_v3_large
from GhostNet import ghostnet
import csv

model_name = 'ghostnet' # 'mobilenet'
save_dir = "/apdcephfs/share_774517/shared_info/brucegeng/transfer_learning"
test_dir = "/apdvqacephfs/share_774517/data/videoqa/horror/data11/fafachen/lowquality/discomfort_detect/video_test_data"
num_classes = 18
workers = 20
batch_size = 256

if model_name == 'mobilenet':
    model_weight_path = os.path.join(save_dir, 'Transfered_MobileNetV3.pth')
    result_file_path = os.path.join(save_dir, 'mobilenet_res.csv')
elif model_name == 'ghostnet':
    model_weight_path = os.path.join(save_dir, 'Transfered_GhostNet.pth')
    result_file_path = os.path.join(save_dir, 'ghostnet_res.csv')
else:
    print("You Must Choose A Model!!!")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_dataset = datasets.ImageFolder(test_dir, data_transform)
    test_num = len(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              drop_last=False, num_workers=workers)
    print("using {} images for testing.".format(test_num))

    if model_name == 'mobilenet':
        model = mobilenet_v3_large(num_classes = num_classes).to(device)
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        print("Mobilenet model loaded!")
    elif model_name == 'ghostnet':
        model = ghostnet(num_classes = num_classes).to(device)
        model.load_state_dict(torch.load(model_weight_path, map_location=device))
        print("Ghostnet model loaded!")
    else:
        print("You Must Chose A Model!!!")

    model.eval()
    with torch.no_grad():
        score_list = []
        predicts = []
        for b, (x, y) in enumerate(test_loader):
            # measure data loading time, which is spent in the `for` statement.
            # Schedule sending to GPU(s)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # compute output
            predict = model(x)
            predict = torch.nn.Softmax(dim=1)(predict)
            for i in predict:
                predicts.append(i)
            if len(predicts) % 200 == 0:
                print("already tested {}/{} images.".format(len(predicts), test_num))

        print("Writing in csv file......")
        # Write rowkey and score in csv file
        with open(result_file_path, "w", newline='') as w:
            csv_writer = csv.writer(w)
            for i in range(test_num):
                res = [test_dataset.imgs[i][0].split('/')[-1]]
                ratio_list = [str(tmp.item()) for tmp in predicts[i]]
                res.extend(ratio_list)
                res.append(test_dataset.imgs[i][1])
                csv_writer.writerow(res)
        print("Testing Finished!.")


if __name__ == '__main__':
    main()

import os
import sys
import math
import json
import random
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

colors = ['r', 'g', 'b', 'c', 'm', 'k']


def draw_dataset_distribution_map(all_class, every_class_num):
    plt.bar(x=range(len(all_class)), height=every_class_num, width=0.6, align="center")
    plt.xticks(ticks=range(len(all_class)), labels=all_class)

    for index, value in enumerate(every_class_num):
        plt.text(x=index, y=value+10, s=str(value), ha='center')

    plt.xlabel("Class Name")
    plt.ylabel("Number")
    plt.show()


def draw_dataset_distribution_map2(all_class, every_class_num, val_every_class_num):
    # fig, ax = plt.subplots(figsize=(10, 8), dpi=60)
    train_every_class_num = []
    for i in range(len(every_class_num)):
        train_every_class_num.append(every_class_num[i]-val_every_class_num[i])
    plt.bar(range(len(all_class)), train_every_class_num, width=0.6, align="center", label="trainset")
    plt.bar(range(len(all_class)), val_every_class_num, width=0.6, bottom=train_every_class_num, align="center", label="testset")
    plt.xticks(ticks=range(len(all_class)), labels=all_class)
    for i in range(len(every_class_num)):
        plt.text(x=i, y=train_every_class_num[i]-30, s=str(train_every_class_num[i]), ha='center')
        plt.text(x=i, y=every_class_num[i]-30, s=str(every_class_num[i]), ha='center')
    plt.xlabel("Class Name")
    plt.ylabel("Number")
    plt.legend(fontsize=8)

    plt.savefig('dataset_distribution.jpg')


def split_dataset(data_root: str, split_rate: float = 0.2):
    random.seed(0)
    all_class = [cla for cla in os.listdir(data_root)]
    all_class.sort()
    class_index_dict = dict((k, v) for v, k in enumerate(all_class))

    train_img_list = []
    val_img_list = []
    train_label_list = []
    val_label_list = []
    every_class_num = []
    val_every_class_num = []
    for the_class in all_class:
        the_dir_path = data_root + "/" + the_class
        images = [the_dir_path + "/" + elem for elem in os.listdir(the_dir_path)]
        images.sort()
        every_class_num.append(len(images))

        image_class = class_index_dict[the_class]
        val_images_path = random.sample(images, int(len(images) * split_rate))
        val_every_class_num.append(len(val_images_path))
        for img_path in images:
            if img_path in val_images_path:
                val_img_list.append(img_path)
                val_label_list.append(image_class)
            else:
                train_img_list.append(img_path)
                train_label_list.append(image_class)

    return train_img_list, train_label_list, val_img_list, val_label_list, all_class, every_class_num, val_every_class_num


def create_step_lr_scheduler(optimizer, step_size: int, gamma: float = 0.1):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)


def create_lambda_lr_scheduler(optimizer, num_step: int, epochs: int, warmup=True, warmup_epochs=1, warmup_factor=1e-3,
                               end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def lambda_lr(x):
        """
        Return a learning rate multiplier factor based on num_step.
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # Record the weight parameters trained by the optimizer.
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    # Record the name of the corresponding weight.
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    return list(parameter_group_vars.values())


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # Count the total loss in one epoch.
    accu_num = torch.zeros(1).to(device)   # Count the number of samples with correct predictions.
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, lr: {:.5f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num,
            optimizer.param_groups[0]["lr"]
        )

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()
        # update lr
        lr_scheduler.step()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # Count the number of samples with correct predictions.
    accu_loss = torch.zeros(1).to(device)  # Count the total loss in one epoch.

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            accu_num.item() / sample_num
        )

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def parse_class_index_file(class_index_file_path):
    assert os.path.exists(class_index_file_path), "file: '{}' dose not exist.".format(class_index_file_path)
    with open(class_index_file_path, 'r') as f:
        class_index_dict = json.load(f)

    return class_index_dict
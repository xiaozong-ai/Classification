import torch
from torchvision import transforms
from utils import (split_dataset, draw_dataset_distribution_map2, get_params_groups, create_lambda_lr_scheduler,
                   train_one_epoch, evaluate)
from mydataset import MyDataset
from convNeXt import ConvNeXt
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim

if __name__ == '__main__':
    model_path = 'weight/convnext_tiny_22k_1k_224.pth'
    max_epoch = 60          # The total epoch the model was trained.
    init_lr = 1e-4          # Initial learning rate.
    weight_decay = 5e-4     # Set the weight decay to prevent overfitting.
    cuda = True             # Set Whether to use GPU.
    fp16 = False            # Set Whether to use mixed-precision training.
    input_shape = 224       # Model Input.
    batch_size = 2          # Batch size for training.
    num_workers = 0         # Set whether to use multithreading to read data.
    freeze_train = True     # Set whether to freeze the backbone during training.

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    train_img_list, train_label_list, val_img_list, val_label_list, all_class, every_class_num, \
        val_every_class_num = split_dataset('dataset/flower_photos/')
    # Plot the distribution of the data set.
    draw_dataset_distribution_map2(all_class, every_class_num, val_every_class_num)

    # Set up the image preprocessing process.
    my_transforms = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(input_shape),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

        "val": transforms.Compose([
            transforms.Resize(int(input_shape * 1.143)),
            transforms.CenterCrop(input_shape),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Instantiate the dataset.
    train_dataset = MyDataset(train_img_list, train_label_list, my_transforms["train"])
    val_dataset = MyDataset(val_img_list, val_label_list, my_transforms["val"])

    # Build the dataset loader.
    train_dataloader = DataLoader(
        train_dataset,
        batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn
    )

    model = ConvNeXt(num_classes=5, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]).to(device)
    if model_path != "":
        # print(model.state_dict())
        pretrained_dict = torch.load(model_path, map_location=device)['model']
        for k in list(pretrained_dict.keys()):
            if "head" in k:
                del pretrained_dict[k]

        model.load_state_dict(pretrained_dict, strict=False)

    if freeze_train:
        for name, param in model.named_parameters():
            if "head" in name:
                param.requires_grad = False

    pg = get_params_groups(model, weight_decay=weight_decay)
    optimizer = optim.AdamW(pg, init_lr, weight_decay=weight_decay)
    lr_scheduler = create_lambda_lr_scheduler(optimizer, len(train_dataloader), max_epoch, warmup=True, warmup_epochs=1)

    best_acc = 0.
    for epoch in range(max_epoch):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_dataloader,
                                                device=device,
                                                epoch=epoch,
                                                lr_scheduler=lr_scheduler)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_dataloader,
                                     device=device,
                                     epoch=epoch)

        if best_acc < val_acc:
            torch.save(model.state_dict(), "logs/best_model.pth")
            best_acc = val_acc
        torch.save(model.state_dict(), "logs/epoch_{}.pth".format(epoch+1))
import os
import math
import torch
import argparse
from itertools import cycle
from ConvLarge import ConvLarge
from dataset import TransCIFAR10
from torchvision.transforms import transforms as T
from torch.utils.data.dataloader import DataLoader
from utils import AddGaussianNoise, WarmupCosineSchedule


def pi_model(args):
    # Setup the hyper parameters
    seed = args.seed
    root = args.data
    start_lr = 0.1
    ref_lr = 0.1
    final_lr = 0.01
    l_batch = 100
    u_batch = 100
    t_batch = 256
    warm_up = 80
    total_epoch = 300
    load_model = True
    filename = f"latest_model.pth.tar.gz"
    path = os.path.join(args.ckpt, filename)

    # device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Reproducibility
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    # Setup the transform
    trans = T.Compose([
        T.RandomCrop(size=28),
        T.RandomHorizontalFlip(p=0.5),
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010)),
        AddGaussianNoise(p=0.5, mean=0., std=0.15)])

    init_trans = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2023, 0.1994, 0.2010))])

    # Setup the train loader
    supervised_set = TransCIFAR10(
        root=root,
        seed=seed,
        keep_file=f'./txt/split_{args.labeled}_{seed}.txt',
        num_labeled=args.labeled,
        training=True,
        transform=trans,
        target_transform=None,
        supervised=True)

    unsupervised_set = TransCIFAR10(
        root=root,
        seed=seed,
        training=True,
        transform=trans,
        target_transform=None,
        supervised=False)

    test_set = TransCIFAR10(
        root=root,
        seed=seed,
        training=False,
        transform=init_trans,
        target_transform=None,
        supervised=False)

    sampler = None
    l_loader = DataLoader(supervised_set,
                          sampler=sampler,
                          batch_size=l_batch,
                          shuffle=True,
                          num_workers=0,
                          drop_last=True,
                          pin_memory=True)

    u_loader = DataLoader(unsupervised_set,
                          batch_size=u_batch,
                          shuffle=True,
                          num_workers=0,
                          drop_last=True,
                          pin_memory=True)

    test_loader = DataLoader(test_set,
                             batch_size=t_batch,
                             shuffle=False,
                             num_workers=0,
                             drop_last=False,
                             pin_memory=True)

    # Setup the model
    model = ConvLarge(num_classes=10).to(device)

    # Iteration per epoch
    ipe = 64 # len(u_loader)##[TODO]64#change

    # Setup optimizer
    optim = torch.optim.SGD(params=model.parameters(), lr=ref_lr, weight_decay=0.0, momentum=0.9)
    sched = WarmupCosineSchedule(optimizer=optim,
                                 warmup_steps=warm_up*ipe,
                                 start_lr=start_lr,
                                 ref_lr=ref_lr,
                                 final_lr=final_lr,
                                 T_max=total_epoch*ipe)
    # Criterion
    cls_fn = torch.nn.CrossEntropyLoss()
    mes_fn = torch.nn.MSELoss()

    start_epoch = 0
    # load last trained model if there it is
    if load_model and os.path.exists(path):
        model, optim, start_epoch = load_checkpoint(
            root=path,
            model=model,
            opt=optim)
        for _ in range(start_epoch):
            for _ in range(ipe):
                sched.step()
        print('last model is loaded...')

    # Training
    for epoch in range(start_epoch, total_epoch):
        model.train()
        for idx, ((l_image, l_label), (u_image, p_image, u_label)) in enumerate(zip(cycle(l_loader), u_loader)):
            l_image, l_label = l_image.to(device), l_label.long().to(device)
            u_image, p_image, u_label = u_image.to(device), p_image.to(device), u_label.long().to(device)

            with torch.cuda.amp.autocast(enabled=True):
                optim.zero_grad()
                inputs = torch.cat([l_image, u_image, p_image], dim=0)
                # z, feat = model(inputs, return_feature=True)
                z = model(inputs)
                with torch.cuda.amp.autocast(enabled=False):
                    # z, feat = z.float(), feat.float()
                    z = z.float()
                    preds = z.argmax(dim=1)
                    cls_loss = cls_fn(z[:l_batch], l_label.detach())
                    mse_loss = mes_fn(z[l_batch+u_batch:], z[l_batch:l_batch+u_batch].detach())
                    coef = math.exp(-5 * (1 - min(epoch / warm_up, 1)) ** 2)
                    loss = cls_loss + coef * mse_loss

                    loss.backward()
                    optim.step()
                    sched.step()

                    l_acc = preds[:l_batch].eq(l_label).float().mean()
                    u_acc = preds[l_batch:l_batch+u_batch].eq(u_label).float().mean()

                    if idx % 30 == 0:
                        lr = optim.param_groups[0]["lr"]
                        log = 'EPOCH:[%03d/%03d], iter:[%04d/%04d], lr: %f, l_loss: %.03f, u_loss: %.03f, l_acc: %.03f, u_acc: %.03f'
                        print(log % (epoch + 1, total_epoch, idx + 1, ipe, lr, cls_loss, mse_loss, l_acc, u_acc))

        # save model per epoch
        save_dict = {'model': model.state_dict(),
                     'opt': optim.state_dict(),
                     'epoch': epoch + 1}
        torch.save(save_dict, path)

        model.eval()
        with torch.no_grad():
            t_acc = []
            for _, (input, label) in enumerate(test_loader):
                input, label = input.to(device), label.long().to(device)
                with torch.cuda.amp.autocast(enabled=True):
                    z = model(input)
                    with torch.cuda.amp.autocast(enabled=False):
                        z.float()
                        preds = z.argmax(dim=1)
                        t_acc.append(preds.eq(label).float().mean())
            test_acc = sum(t_acc) / len(t_acc)
            print('The test accuracy is %.03f' % test_acc)


def load_checkpoint(
    root,
    model,
    opt,
):
    checkpoint = torch.load(root, map_location='cpu')
    epoch = checkpoint['epoch']
    # -- loading encoder
    model.load_state_dict(checkpoint['model'])
    # -- loading optimizer
    opt.load_state_dict(checkpoint['opt'])
    del checkpoint
    return model, opt, epoch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./cifar10", type=str, help="dataset")
    parser.add_argument("--ckpt", default="./pimodel", type=str, help="save directory for model")
    parser.add_argument("--labeled", default=1000, type=int,
                        choices=[500, 1000, 4000], help="The number of labeled data")
    parser.add_argument("--seed", default=123, type=int, help="seed for random variables")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        os.makedirs(args.ckpt)
    if not os.path.exists(args.data):
        os.makedirs(args.data)

    pi_model(args)

import os
import argparse
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import TRCN
from dataset import *
from utils import *
from torch.optim import lr_scheduler

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="TRCN")
parser.add_argument("--batchSize", type=int, default=32, help="Training batch size")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
opt = parser.parse_args()

def main():
    print('Loading dataset ...\n')
    # ATFRSD
    # dataset_train = DatasetFrommatlab("data/ATFRSD/train/TTR.mat", "data/ATFRSD/train/TT.mat", "TTR", "TT")
    # data_mat = scio.loadmat("data/ATFRSD/test/TTR.mat")
    # dataset_val = (np.array(data_mat["TTR"]) -777.11)/(2086.01-777.11)
    # target_mat = scio.loadmat("data/ATFRSD/test/TT.mat")
    # targetset_val = (np.array(target_mat["TT"]) -777.11)/(2086.01-777.11)


    # ATFRSD-W
    dataset_train = DatasetFrommatlab("data/ATFRSD-W/train/TR.mat","data/ATFRSD-W/train/T.mat", "TR", "T")
    data_mat = scio.loadmat("data/ATFRSD-W/test/TR.mat")
    dataset_val = (np.array(data_mat["TR"]) - 640.8) / (1769.4 - 640.8)
    target_mat = scio.loadmat("data/ATFRSD-W/test/T.mat")
    targetset_val = (np.array(target_mat["T"]) - 640.8) / (1769.4 - 640.8)


    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    net = TRCN(channels=1, num_of_layers=opt.num_of_layers)
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(size_average=False)
    criterion_L1 = nn.L1Loss(size_average=False)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()
    criterion_L1.cuda()

    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    writer = SummaryWriter(opt.outf)
    step = 0
    torch.cuda.empty_cache()

    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)

        for i, batch in enumerate(loader_train, 0):
            data, target = batch[0], batch[1]
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            imgn_train = data
            img_train = target
            noise = imgn_train - img_train
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            noise = Variable(noise.cuda())
            out_train = model(imgn_train)
            loss_L2 = criterion(out_train, noise) / (imgn_train.size()[0] * 2)
            loss_L1 = criterion_L1(torch.clamp(imgn_train - model(imgn_train), 0., 1.), img_train) / (img_train.size()[0] * 2)
            loss = 0.8*loss_L1 + 0.2*loss_L2
            loss.backward()
            optimizer.step()

            model.eval()
            out_train = torch.clamp(imgn_train - model(imgn_train), 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            rmse_train = batch_RMSE(out_train, img_train)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f  RMSE_train: %.4f" %
                  (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train , rmse_train))
            if step % 10 == 0:
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
                writer.add_scalar('RMSE on training data', rmse_train, step)
            step += 1
        model.eval()

        imgn_val = dataset_val.astype(np.float32)
        img_val = targetset_val.astype(np.float32)
        with torch.no_grad():
            img_val, imgn_val = Variable(torch.tensor(img_val).cuda(), volatile=True), Variable(
                torch.tensor(imgn_val).cuda(), volatile=True)
            out_val = torch.clamp(imgn_val - model(imgn_val), 0., 1.)
        psnr_val = batch_PSNR(out_val, img_val, 1.)
        rmse_val = batch_RMSE(out_val, img_val)
        rmse_std = batch_RMSE(imgn_val, img_val)
        rmse_std /= len(dataset_val)
        psnr_val /= len(dataset_val)
        rmse_val /= len(dataset_val)
        print("\n RMSE_val: %.4f" % (rmse_std))
        print("\n[epoch %d] PSNR_val: %.4f RMSE_val: %.4f" % (epoch + 1, psnr_val, rmse_val))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        writer.add_scalar('RMSE on validation data', rmse_val, epoch)

        out_train = torch.clamp(imgn_train - model(imgn_train), 0., 1.)
        Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', Img, epoch)
        writer.add_image('noisy image', Imgn, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)
        torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))


if __name__ == "__main__":
    main()

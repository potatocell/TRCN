import os
import argparse
import torch
import scipy.io as scio
from torch.autograd import Variable
from models import *
from utils import *


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="TRCN_Test")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")

opt = parser.parse_args()


def main():
    print('Loading model ...\n')
    net = TRCN(channels=1, num_of_layers=opt.num_of_layers)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth')))
    model.eval()
    print('Loading data info ...\n')

    # ATFRSD
    # data_mat = scio.loadmat("data/ATFRSD/test/TTR.mat")
    # dataset_test = (np.array(data_mat["TTR"])- 777.1) / (2086.0 - 777.1)
    # target_mat = scio.loadmat("data/ATFRSD/test/TT.mat")
    # targetset_test = (np.array(target_mat["TT"]) - 777.1) / (2086.0 - 777.1)

    # ATFRSD-W
    data_mat = scio.loadmat("data/ATFRSD-W/test/TR.mat")
    target_mat = scio.loadmat("data/ATFRSD-W/test/T.mat")
    targetset_test = (np.array(target_mat["T"]) - 640.8) / (1796.4 - 640.8)
    dataset_test = (np.array(data_mat["TR"]) - 640.8) / (1796.4 - 640.8)


    imgn_test = dataset_test.astype(np.float32)
    img_test = targetset_test.astype(np.float32)
    img_test, imgn_test = Variable(torch.tensor(img_test).cuda(), volatile=True), Variable(torch.tensor(imgn_test).cuda(), volatile=True)
    with torch.no_grad():
        Out = torch.clamp(imgn_test - model(imgn_test), 0., 1.)


    # ATFRSD
    # Out_resmp = Out * (2086.0-777.1) + 777.1
    # img_resmp = img_test * (2086.0-777.1) + 777.1
    # imgn_resmp = imgn_test * (2086.0-777.1) + 777.1


    # ATFRSD-W
    Out_resmp = Out * (1796.4 - 640.8) + 640.8
    img_resmp = img_test * (1796.4 - 640.8) + 640.8
    imgn_resmp = imgn_test * (1796.4 - 640.8) + 640.8


    print("global:")
    print("rmse_std: ")
    batch_RMSE_test(imgn_resmp, img_resmp)
    print("rmse_test: ")
    batch_RMSE_test(Out_resmp, img_resmp)

    print("resmp:")
    print("rmse_std: ")
    batch_RMSE_centerandedge(imgn_resmp, img_resmp)
    print("rmse_test: ")
    batch_RMSE_centerandedge(Out_resmp, img_resmp)



if __name__ == "__main__":
    main()

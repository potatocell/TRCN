import math
import torch.nn as nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def batch_RMSE(predicts,targets):
    predicts = predicts.data.cpu().numpy().astype(np.float32)
    targets = targets.data.cpu().numpy().astype(np.float32)

    RMSE = 0
    for i in range(predicts.shape[0]):
        RMSE += 100*(np.sqrt(((predicts[i,:,:,:] - targets[i,:,:,:]) ** 2).mean())/targets[i,:,:,:].mean())
    return (RMSE / predicts.shape[0])

def batch_RMSE_test(predicts,targets):
    predicts = predicts.data.cpu().numpy().astype(np.float32)
    targets = targets.data.cpu().numpy().astype(np.float32)
    RMSE_1 = 0
    for i in range(predicts.shape[0]):
        RMSE_1 += 100*(np.sqrt(((predicts[i,:,:,:] - targets[i,:,:,:]) ** 2).mean())/targets[i,:,:,:].mean())
    RMSE_1=RMSE_1/predicts.shape[0]
    print("rmse: %f " % (RMSE_1))

def batch_RMSE_centerandedge(predicts,targets):
    predicts = predicts.data.cpu().numpy().astype(np.float32)
    targets = targets.data.cpu().numpy().astype(np.float32)
    # center
    predicts_center = predicts[:, :, 13:88, 9:92]  # 17:84, 10:91
    targets_center = targets[:, :, 13:88, 9:92]  # 17:84, 10:91
    print("center:")
    RMSE = 0
    MAX = 0
    for i in range(predicts_center.shape[0]):
        RMSE += 100*(np.sqrt(((predicts_center[i,:,:,:] - targets_center[i,:,:,:]) ** 2).mean())/targets_center[i,:,:,:].mean())
        MAX += np.max(np.abs(predicts_center[i,:,:,:] - targets_center[i,:,:,:]))
    RMSE = RMSE/predicts_center.shape[0]
    MAX = MAX/predicts_center.shape[0]
    print("rmse: %f " % (RMSE))
    print("max: %f " % (MAX))



    #edge
    predicts[:, :, 13:88, 9:92] = 0    # 17:84, 10:91
    targets[:, :,13:88, 9:92] = 0    # 17:84, 10:91
    predicts_edge = predicts
    targets_edge = targets

    print("edge:")
    RMSE_edge = 0
    MAX_edge = 0
    for i in range(predicts_center.shape[0]):
        targets_edge_mean = targets_edge[i, :, :, :].sum() / (101 * 101 - 75 * 83)  # 67*81
        RMSE_edge += 100 * (np.sqrt(((predicts_edge[i, :, :, :] - targets_edge[i, :, :, :]) ** 2).sum()/(101 * 101 - 75 * 83)) / targets_edge_mean)
        MAX_edge += np.max(np.abs(predicts_edge[i, :, :, :] - targets_edge[i, :, :, :]))

    RMSE_edge = RMSE_edge / predicts_center.shape[0]
    MAX_edge = MAX_edge / predicts_center.shape[0]
    print("rmse: %f " % (RMSE_edge))
    print("max: %f " % (MAX_edge))




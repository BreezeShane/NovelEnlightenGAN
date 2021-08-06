from Config import *
import lpips
import numpy as np
import cv2 as cv
from PIL import Image
import torchvision
import torch.nn as nn
from torchvision import transforms
import pandas as pd
import torchvision.models as models
from utils.NIQE import niqe
from skimage.metrics import structural_similarity as ssim


def Load_Test_Results_Paths(Aligned=True, Name='A'):
    result_paths = os.path.join(ROOT_PATH, 'Data/Results/')
    if Aligned:
        img_A_paths = []
        img_B_paths = []
        for root, _, file_names in sorted(os.walk(result_paths, followlinks=True)):
            for file_name in file_names:
                path = os.path.join(root, file_name)
                if '_real_A' in file_name:
                    img_A_paths.append(path)
                if '_fake_B' in file_name:
                    img_B_paths.append(path)
        return img_A_paths, img_B_paths
    else:
        if Name == 'A':
            img_A_paths = []
            for root, _, file_names in sorted(os.walk(result_paths, followlinks=True)):
                for file_name in file_names:
                    path = os.path.join(root, file_name)
                    if '_real_A' in file_name:
                        img_A_paths.append(path)
            return img_A_paths
        else:
            img_B_paths = []
            for root, _, file_names in sorted(os.walk(result_paths, followlinks=True)):
                for file_name in file_names:
                    path = os.path.join(root, file_name)
                    if '_fake_B' in file_name:
                        img_B_paths.append(path)
            return img_B_paths


def Load_Test_Results(Aligned=True, Name='A'):
    result_paths = os.path.join(ROOT_PATH, 'Data/Results/')
    if Aligned:
        img_A_paths = []
        img_B_paths = []
        imgs_A = []
        imgs_B = []
        for root, _, file_names in sorted(os.walk(result_paths, followlinks=True)):
            for file_name in file_names:
                path = os.path.join(root, file_name)
                if '_real_A' in file_name:
                    img_A_paths.append(path)
                if '_fake_B' in file_name:
                    img_B_paths.append(path)
        for a in img_A_paths:
            imgs_A.append(cv.imread(a))
        for b in img_B_paths:
            imgs_B.append(cv.imread(b))
        return imgs_A, imgs_B
    else:
        if Name == 'A':
            img_A_paths = []
            imgs_A = []
            for root, _, file_names in sorted(os.walk(result_paths, followlinks=True)):
                for file_name in file_names:
                    path = os.path.join(root, file_name)
                    if '_real_A' in file_name:
                        img_A_paths.append(path)
            for a in img_A_paths:
                imgs_A.append(cv.imread(a))
            return imgs_A
        else:
            img_B_paths = []
            imgs_B = []
            for root, _, file_names in sorted(os.walk(result_paths, followlinks=True)):
                for file_name in file_names:
                    path = os.path.join(root, file_name)
                    if '_fake_B' in file_name:
                        img_B_paths.append(path)
            for b in img_B_paths:
                imgs_B.append(cv.imread(b))
            return imgs_B


def Compute_MAE():
    MAEs = []
    imgs_A, imgs_B = Load_Test_Results()
    for img_A, img_B in imgs_A, imgs_B:
        Error = abs(img_A - img_B)
        gray = cv.cvtColor(Error, cv.COLOR_BGR2GRAY)
        MAEs.append(np.mean(gray))
        # img_A_gray = cv.cvtColor(img_A, cv.COLOR_BGR2GRAY)
        # img_B_gray = cv.cvtColor(img_B, cv.COLOR_BGR2GRAY)
        # mae = abs(img_A_gray - img_B_gray)
        # MAEs.append(np.mean(mae))
    return np.mean(MAEs)


def Compute_MSE():
    MSEs = []
    imgs_A, imgs_B = Load_Test_Results()
    for img_A, img_B in imgs_A, imgs_B:
        img_A_gray = cv.cvtColor(img_A, cv.COLOR_BGR2GRAY)
        img_B_gray = cv.cvtColor(img_B, cv.COLOR_BGR2GRAY)
        mse = torch.nn.MSELoss()
        MSEs.append(mse(img_A_gray, img_B_gray))
    return np.mean(MSEs)


def Compute_PSNR():
    PSNRs = []
    imgs_A, imgs_B = Load_Test_Results(Aligned=True)
    for img_A, img_B in imgs_A, imgs_B:
        PSNRs.append(cv.PSNR(img_A, img_B))
    return np.mean(PSNRs)


def Compute_SSIM():
    SSIMs = []
    imgs_A, imgs_B = Load_Test_Results(Aligned=True)
    for img_A, img_B in imgs_A, imgs_B:
        img_A_gray = cv.cvtColor(img_A, cv.COLOR_BGR2GRAY)
        img_B_gray = cv.cvtColor(img_B, cv.COLOR_BGR2GRAY)
        SSIMs.append(ssim(img_A_gray, img_B_gray))
    return np.mean(SSIMs)


def Compute_LPIPS():
    loss_fn = lpips.LPIPS(net='alex')
    LPIPSs = []
    if opt.use_gpu:
        loss_fn.cuda()
    img_A_paths, img_B_paths = Load_Test_Results_Paths()
    for img_A_path, img_B_path in img_A_paths, img_B_paths:
        img_A = lpips.im2tensor(lpips.load_image(img_A_path))
        img_B = lpips.im2tensor(lpips.load_image(img_B_path))
        if opt.use_gpu:
            img_A = img_A.cuda()
            img_B = img_B.cuda()
        LPIPSs.append(loss_fn.forward(img_A, img_B))
    return np.mean(LPIPSs)


def Compute_LOE():
    LOEs = []
    imgs_A, imgs_B = Load_Test_Results()
    for img_A, img_B in imgs_A, imgs_B:
        L_A = max(img_A)
        L_B = max(img_B)
        U_A = L_A >= img_A + 0
        U_B = L_B >= img_B + 0
        RD = U_A ^ U_B
        LOEs.append(np.mean(RD))
    return np.mean(LOEs)


def Compute_NIQE():
    NIQEs = []
    imgs_B_paths = Load_Test_Results_Paths(Aligned=False, Name='B')
    for img_B_path in imgs_B_paths:
        img_B = np.array(Image.open(img_B_path).convert('LA'))[:, :, 0]
        NIQEs.append(niqe(img_B))
    return np.mean(NIQEs)


def Compute_SPAQ(size=512, input_size=224):
    SPAQs = []
    imgs_B = Load_Test_Results(Aligned=False, Name='B')
    model = Baseline()
    for img_B in imgs_B:
        w_b, h_b = img_B.size
        if w_b >= size or h_b >= size:
            img_B = transforms.ToTensor()(transforms.Resize(size, Image.BILINEAR)(img_B))
        img_B = np.transpose(img_B, (1, 0, 2))
        img_shape_B = img_B.shape
        if len(img_shape_B) == 2:
            H_B, W_B, = img_shape_B
            num_of_channel_B = 1
        else:
            H_B, W_B, num_of_channel_B = img_shape_B
        if num_of_channel_B == 1:
            img_B = np.asarray([img_B, ] * 3, dtype=img_B.dtype)

        stride = int(input_size / 2)
        hIdxMax_B = H_B - input_size
        wIdxMax_B = W_B - input_size

        hIdx_B = [i * stride for i in range(int(hIdxMax_B / stride) + 1)]
        if H_B - input_size != hIdx_B[-1]:
            hIdx_B.append(H_B - input_size)
        wIdx_B = [i * stride for i in range(int(wIdxMax_B / stride) + 1)]
        if W_B - input_size != wIdx_B[-1]:
            wIdx_B.append(W_B - input_size)
        patches_numpy = [img_B[hId:hId + input_size, wId:wId + input_size, :]
                         for hId in hIdx_B
                         for wId in wIdx_B]
        patches_tensor = [transforms.ToTensor()(p) for p in patches_numpy]
        patches_tensor = torch.stack(patches_tensor, 0).contiguous()
        Image_B = patches_tensor.squeeze(0)

        if opt.use_gpu:
            Image_B = Image_B.cuda()
        score_B = model(Image_B).mean()
        SPAQs.append(score_B.item())
    return np.mean(SPAQs)


def Compute_NIMA():
    NIMAs = []
    imgs_B = Load_Test_Results(Aligned=False, Name='B')
    model_pth = os.path.join(ROOT_PATH, 'Model/epoch-34.pth')
    test_csv = os.path.join(ROOT_PATH, 'lib/test_labels.csv')
    for img_B in imgs_B:
        imgs_B.append(cv.cvtColor(img_B, cv.COLOR_BGR2RGB))
    base_model = models.vgg16(pretrained=True)
    model = NIMA(base_model)
    try:
        model.load_state_dict(torch.load(model_pth))
        print('successfully loaded model')
    except IOError:
        print("Model doesn't exist! ")
        raise
    seed = 42
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    test_df = pd.read_csv(test_csv, header=None)
    for i, img in enumerate(imgs_B):
        gt = test_df[test_df[0] == img].to_numpy()[:, 1:].reshape(10, 1)
        gt_mean = 0.0
        for l, e in enumerate(gt, 1):
            gt_mean += l * e
        NIMAs.append(round(gt_mean, 3))
    return np.mean(NIMAs)


class Baseline(nn.Module):
    def __init__(self):
        super(Baseline, self).__init__()
        self.backbone = torchvision.models.resnet50(pretrained=False)
        fc_feature = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(fc_feature, 1, bias=True)

    def forward(self, x):
        result = self.backbone(x)
        return result


class NIMA(nn.Module):
    """Neural IMage Assessment model by Google"""

    def __init__(self, base_model, num_classes=10):
        super(NIMA, self).__init__()
        self.features = base_model.features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(in_features=25088, out_features=num_classes),
            nn.Softmax())

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

use_gpu = torch.cuda.is_available()


def extract_features(model, dataset, batch_size=128):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)
    model.eval()
    features = []
    for img, _, filename in tqdm(dataloader, total=int(len(dataset) / batch_size)):
        if use_gpu:
            img = Variable(img).cuda(non_blocking=True)
        feature = model.extract_features(img).cpu().detach().numpy()
        features.extend(feature)
    features = np.array(features)
    return features

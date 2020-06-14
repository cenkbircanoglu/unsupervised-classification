import os

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

use_gpu = torch.cuda.is_available()


def extract_features(model, dataset, debug_root=None, epoch=None, batch_size=128):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=False)
    model.eval()
    data = {
        'features': [],
        'filenames': []
    }
    for img, _, filename in tqdm(dataloader, total=int(len(dataset) / batch_size)):
        if use_gpu:
            img = Variable(img).cuda(non_blocking=True)
        feature = model.extract_features(img).cpu().detach().numpy()
        data['features'].extend(feature)
        data['filenames'].extend(filename)
    if debug_root:
        feature_path = os.path.join(debug_root, 'features_%s.npy' % epoch)
        np.save(feature_path, data)
    return data

import os

import torch


def load_latest_checkpoint(model, checkpoint, use_gpu=False):
    checkpoint_dir = os.path.dirname(checkpoint)
    max_epoch = 0
    if os.path.exists(checkpoint_dir):
        filenames = os.listdir(checkpoint_dir)
        latest_model = None
        for filename in filenames:
            epoch = int(filename.replace('.pth', '').split('_')[-1])
            if epoch > max_epoch:
                max_epoch = epoch
                latest_model = filename

        latest_model_path = os.path.join(checkpoint_dir, latest_model)
        print('Latest Model are going to be loaded %s' % latest_model_path)
        if use_gpu:
            model.load_state_dict(torch.load(latest_model_path), strict=True)
        else:
            model.load_state_dict(torch.load(latest_model_path, map_location=torch.device('cpu')), strict=True)

    return model, max_epoch


def save_checkpoint(model, checkpoint, epoch):
    checkpoint = '%s_%s.%s' % (checkpoint.replace('.pth', ''), str(epoch), 'pth')
    try:
        state_dict = model.module.state_dict()
    except:
        state_dict = model.state_dict()
    os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
    torch.save(state_dict, checkpoint)

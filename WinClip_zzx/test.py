
from loguru import logger
from torch.utils.data import DataLoader
from PIL import Image
import torch
from datasets import *
from utils.csv_utils import *
from utils.metrics import *
from utils.training_utils import *
from utils.eval_utils import *


def test(model,
         dataloader: DataLoader,
         device: str,
         is_vis: bool,
         img_dir: str,
         class_name: str,
         cal_pro: bool,
         train_data: DataLoader,
         resolution: int):

    # change the model into eval mode
    model.eval_mode()

    logger.info('begin build text feature gallery...')
    model.build_text_feature_gallery(class_name)
    logger.info('build text feature gallery finished.')

    if train_data is not None:
        logger.info('begin build image feature gallery...')
        for (data, mask, label, name, img_type) in train_data:
            data = [model.transform(Image.fromarray(cv2.cvtColor(f.numpy(), cv2.COLOR_BGR2RGB))) for f in data]
            data = torch.stack(data, dim=0)

            data = data.to(device)
            model.build_image_feature_gallery(data)
        logger.info('build image feature gallery finished.')

    scores = []
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    names = []

    for (data, mask, label, name, img_type) in dataloader:

        data = [model.transform(Image.fromarray(f.numpy())) for f in data]
        data = torch.stack(data, dim=0)

        for d, n, l, m in zip(data, name, label, mask):
            test_imgs += [denormalization(d.cpu().numpy())]
            l = l.numpy()
            m = m.numpy()
            m[m > 0] = 1

            names += [n]
            gt_list += [l]
            gt_mask_list += [m]

        data = data.to(device)
        score = model(data)
        scores += score

    test_imgs, scores, gt_mask_list = specify_resolution(test_imgs, scores, gt_mask_list, resolution=(resolution, resolution))
    result_dict = metric_cal(np.array(scores), gt_list, gt_mask_list, cal_pro=cal_pro)

    if is_vis:
        plot_sample_cv2(names, test_imgs, {'WinClip': scores}, gt_mask_list, save_folder=img_dir)

    return result_dict
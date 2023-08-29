
from loguru import logger
from torch.utils.data import DataLoader
from PIL import Image
import torch
from datasets import *
from utils.csv_utils import *
from utils.metrics import *
from utils.training_utils import *
from utils.eval_utils import *

import clip_zzx

def test(model,
         preprocess,
         dataloader: DataLoader,
         device: str,
         is_vis: bool,
         img_dir: str,
         class_name: str,
         cal_pro: bool,
         train_data: DataLoader,
         resolution: int):

    # change the model into eval mode
    model.eval()

    logger.info('begin build text feature gallery...')
    text_features = clip_zzx.encode_text_with_prompt_ensemble_anomaly(model, class_name, device)
    # model.build_text_feature_gallery(class_name)
    logger.info('build text feature gallery finished.')


    scores = []
    multi_scale_scores = []
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    names = []

    for (data, mask, label, name, img_type) in dataloader:

        # data = preprocess(data).unsqueeze(0).to(device)
        # data = [model.transform(Image.fromarray(f.numpy())) for f in data]
        # data = torch.stack(data, dim=0)

        for d, n, l, m in zip(data, name, label, mask):
            test_imgs += [denormalization(d.cpu().numpy())]
            l = l.numpy()
            m = m.numpy()
            m[m > 0] = 1

            names += [n]
            gt_list += [l]
            gt_mask_list += [m]

        data = data.to(device)
        image_features = model.encode_image(data)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        features = image_features @ text_features.t()
        similarity_map = clip_zzx.get_similarity_map(features[:, 1:, :], data.shape[-2:])
        
            
    # test_imgs, scores, gt_mask_list = specify_resolution(test_imgs, scores, gt_mask_list, resolution=(resolution, resolution))
    
    if is_vis:
        plot_sample_cv2(names, test_imgs, {'WinClip': similarity_map}, gt_mask_list, save_folder=img_dir, multi_scale_scores=resized_multi_scale_scores)

    # return result_dict
    return
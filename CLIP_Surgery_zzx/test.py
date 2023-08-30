
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

def test(model_text,
         model_image,
         preprocess,
         dataloader: DataLoader,
         device: str,
         is_vis: bool,
         img_dir: str,
         class_name: str,
         cal_pro: bool,
         train_data: DataLoader,
         resolution: int):

    logger.info('begin build text feature gallery...')
    text_features = clip_zzx.encode_text_with_prompt_ensemble_anomaly(model_text, class_name, device)
    # model.build_text_feature_gallery(class_name)
    logger.info('build text feature gallery finished.')


    scores = []
    multi_scale_scores = []
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    names = []

    similarity_map_list = []
    # model = torch.nn.DataParallel(model, device_ids=[0, 1])
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
        image_features = model_image(data)
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        features = image_features @ text_features.t()
        similarity_map = clip_zzx.get_similarity_map(features[:, 1:, :], data.shape[-2:])
        
        for i in range(similarity_map.shape[0]):
            similarity_map_list.append((similarity_map[i].detach().cpu().numpy() * 255).astype('uint8'))
        
            
    # test_imgs, scores, gt_mask_list = specify_resolution(test_imgs, scores, gt_mask_list, resolution=(resolution, resolution))
    
    if is_vis:
        plot_sample_cv2(names, test_imgs, {'WinClip': similarity_map_list}, gt_mask_list, save_folder=img_dir)
    return
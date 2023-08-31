
from loguru import logger
from torch.utils.data import DataLoader
from PIL import Image
import torch
from datasets import *
from utils.csv_utils import *
from utils.metrics import *
from utils.training_utils import *
from utils.eval_utils import *
from model_inference import encode_text_with_prompt_ensemble_anomaly

from torchvision.utils import save_image

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
    text_features = encode_text_with_prompt_ensemble_anomaly(model_text, class_name, device)
    # model.build_text_feature_gallery(class_name)
    logger.info('build text feature gallery finished.')


    scores = []
    multi_scale_scores = []
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    names = []

    similarity_map_list = []
    normal_score_list = []
    abnormal_score_list = []
    # model = torch.nn.DataParallel(model, device_ids=[0, 1])
    for (data, mask, label, name, img_type) in dataloader:

        for d, n, l, m in zip(data, name, label, mask):
            test_imgs += [denormalization(d.cpu().numpy())]
            l = l.numpy()
            m = m.numpy()
            m[m > 0] = 1

            names += [n]
            gt_list += [l]
            gt_mask_list += [m]

        save_image(data[0], 'processed_sample_image.png')
        data = data.to(device)
        image_features = model_image(data)
        # image_features = image_features / image_features.norm(dim=1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        features = image_features @ text_features.t()
        # calculate anomaly map
        # normality_and_abnormality_score = features[:, 1:, :].softmax(dim=-1)
        normality_and_abnormality_score = (100*features[:, 1:, :]).softmax(dim=-1)
        normality_and_abnormality_score = normality_and_abnormality_score.reshape(normality_and_abnormality_score.shape[0], int(normality_and_abnormality_score.shape[1] ** 0.5), int(normality_and_abnormality_score.shape[1] ** 0.5), -1)
        
        normality_and_abnormality_score = normality_and_abnormality_score.permute(0, 3, 1, 2)
        normality_and_abnormality_score = torch.nn.functional.interpolate(normality_and_abnormality_score, data.shape[-2], mode='bilinear')
        normality_and_abnormality_score = normality_and_abnormality_score.permute(0, 2, 3, 1)
        normality_score = normality_and_abnormality_score[:, :, :, 0]
        abnormality_score = normality_and_abnormality_score[:, :, :, 1]
        
        similarity_map = clip_zzx.get_similarity_map(features[:, 1:, :], data.shape[-2:])
        similarity_map = similarity_map[:,:,:,1]
        # for i in range(similarity_map.shape[0]):
        #     similarity_map_list.append((similarity_map[i].detach().cpu().numpy() * 255).astype('uint8'))
        for i in range(similarity_map.shape[0]):
            similarity_map_list.append((similarity_map[i].detach().cpu().numpy() * 255).astype('uint8'))
            normal_score_list.append((normality_score[i].detach().cpu().numpy() * 255).astype('uint8'))
            abnormal_score_list.append((abnormality_score[i].detach().cpu().numpy() * 255).astype('uint8'))
        
            
    # test_imgs, scores, gt_mask_list = specify_resolution(test_imgs, scores, gt_mask_list, resolution=(resolution, resolution))
    
    if is_vis:
        figure_dict = {
            'WinClip_simm': similarity_map_list,
            'WinClip_normal:': normal_score_list,
            'WinClip_anomaly:': abnormal_score_list,
        }
        # plot_sample_cv2(names, test_imgs, {'WinClip': similarity_map_list}, gt_mask_list, save_folder=img_dir)
        plot_sample_cv2(names, test_imgs, figure_dict, gt_mask_list, save_folder=img_dir)
    return
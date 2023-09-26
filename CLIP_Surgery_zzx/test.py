
from loguru import logger
from torch.utils.data import DataLoader
from PIL import Image
import torch
from datasets import *
from utils.csv_utils import *
from utils.metrics import *
from utils.training_utils import *
from utils.eval_utils import *
from model_inference import encode_text_with_prompt_ensemble_anomaly, encode_text_with_prompt_ensemble_anomaly_category, encode_text_with_prompt_ensemble_anomaly_single_word, encode_text_with_prompt_ensemble_anomaly_random_word

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
         resolution: int,
         prompt_engineer: str,
         single_word: bool = None):

    logger.info('begin build text feature gallery...')
    if prompt_engineer == 'mean':
        if not single_word:
            text_features = encode_text_with_prompt_ensemble_anomaly(model_text, class_name, device)
        else:
            text_features = encode_text_with_prompt_ensemble_anomaly_single_word(model_text, class_name, single_word, device)
    elif 'cluster' in prompt_engineer:
        text_features, _ = encode_text_with_prompt_ensemble_anomaly_category(model_text, class_name, device, prompt_engineer)
    elif prompt_engineer == 'random':
        text_features = encode_text_with_prompt_ensemble_anomaly_random_word(model_text, class_name, device)
        
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
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        normality_score_list = []
        abnormality_score_list = []
        case_similarity_map_list = []
        for text_feature in text_features:
            features = image_features @ text_feature.t()
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

            normality_score_list.append(normality_score)
            abnormality_score_list.append(abnormality_score)
            case_similarity_map_list.append(similarity_map)
            
        tot_normality_score = torch.stack(normality_score_list)
        tot_normality_score = torch.mean(tot_normality_score,dim=0)
        
        tot_abnormality_score = torch.stack(abnormality_score_list)
        tot_abnormality_score = torch.mean(tot_abnormality_score,dim=0)
        
        tot_similarity_map = torch.stack(case_similarity_map_list)
        tot_similarity_map = torch.mean(tot_similarity_map,dim=0)
            
        for i in range(tot_similarity_map.shape[0]):
            similarity_map_list.append((tot_similarity_map[i].detach().cpu().numpy() * 255).astype('uint8'))
            normal_score_list.append((tot_normality_score[i].detach().cpu().numpy() * 255).astype('uint8'))
            abnormal_score_list.append((abnormality_score[i].detach().cpu().numpy() * 255).astype('uint8'))
        
            
    test_imgs, scores, gt_mask_list = specify_resolution(test_imgs, abnormal_score_list, gt_mask_list, resolution=(data.shape[-1], data.shape[-1]))
    result_dict = metric_cal(np.array(scores), gt_list, gt_mask_list, cal_pro=cal_pro)
    
    
    if is_vis:
        figure_dict = {
            'WinClip_simm': similarity_map_list,
            'WinClip_normal:': normal_score_list,
            'WinClip_anomaly:': abnormal_score_list,
        }
        # plot_sample_cv2(names, test_imgs, {'WinClip': similarity_map_list}, gt_mask_list, save_folder=img_dir)
        plot_sample_cv2(names, test_imgs, figure_dict, gt_mask_list, save_folder=img_dir)
    return result_dict
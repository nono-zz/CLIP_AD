
from loguru import logger
from torch.utils.data import DataLoader
from PIL import Image
import torch
from datasets import *
from utils.csv_utils import *
from utils.metrics import *
from utils.training_utils import *
from utils.eval_utils import *

from torchvision.utils import save_image

def test(model,
         dataloader: DataLoader,
         device: str,
         is_vis: bool,
         img_dir: str,
         class_name: str,
         cal_pro: bool,
         train_data: DataLoader,
         resolution: int,
         sample_only: bool,
         prompt_template_mode: str):

    # change the model into eval mode
    model.eval_mode()

    logger.info('begin build text feature gallery...')
    model.build_text_feature_gallery(class_name, prompt_template_mode=prompt_template_mode)
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
    multi_scale_scores = []
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

        save_image(data[0], 'processed_sample_image.png')
        data = data.to(device)
        score = model(data)
        if isinstance(score, tuple):
            scores += score[0]
            multi_scale_scores += score[1]
        else:
            scores += score
            
    test_imgs, scores, gt_mask_list = specify_resolution(test_imgs, scores, gt_mask_list, resolution=(resolution, resolution))
    
    if not sample_only:
        resized_multi_scale_scores = []
        if not len(multi_scale_scores)==0:
            for multi_scale_score in multi_scale_scores:
                resized_multi_scale_score = [cv2.resize(multi_scale_score[i], (resolution, resolution), interpolation=cv2.INTER_CUBIC) for i in range(multi_scale_score.shape[0])]
                resized_multi_scale_scores.append(resized_multi_scale_score)
        # transpose list to [3, 117]
        array_data = np.array(resized_multi_scale_scores)
        transposed_array = np.transpose(array_data, (1, 0, 2, 3))
    
        
    result_dict = metric_cal(np.array(scores), gt_list, gt_mask_list, cal_pro=cal_pro, sample_only=sample_only)

    if is_vis and not sample_only:
        plot_sample_cv2(names, test_imgs, {'WinClip': scores}, gt_mask_list, save_folder=img_dir, multi_scale_scores=resized_multi_scale_scores)
        
        # if not len(multi_scale_scores)==0:
        #     score_dict = {
        #         'WinClip': scores,
        #         'WinClip_scale_2': transposed_array[0],
        #         'WinClip_scale_3': transposed_array[1],
        #         'WinClip_scale_15': transposed_array[2],
        #     }
        # plot_sample_cv2(names, test_imgs, score_dict, gt_mask_list, save_folder=img_dir, multi_scale_scores=resized_multi_scale_scores)

    return result_dict
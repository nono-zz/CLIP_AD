import os
import cv2
import json
import torch
import random
import logging
import argparse
import numpy as np
from PIL import Image
from skimage import measure
from tabulate import tabulate
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise
import math
import open_clip
from utils import aug
from model import LinearLayer, TextAdapter, Adapter
from dataset import VisaDataset, MVTecDataset, DIYDataset
from prompt_ensemble import encode_text_with_prompt_ensemble, encode_text_with_prompt_ensemble_segmentation
from torchvision import transforms
from scipy.ndimage import gaussian_filter
from torchvision.utils import save_image

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False


def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.8):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    #scoremap[:,:,2]=scoremap[:,:,1]
    #scoremap[:,:,1]=0
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    #cam = np.float32(scoremap)/255 + np.float32(image)/255
    #cam = cam / np.max(cam)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)#, scoremap

def apply_ad_bmap(image, scoremap, alpha=0.7):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    #cam = np.float32(scoremap)/255 + np.float32(image)/255
    #cam = cam / np.max(cam)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc
def _convert_image_to_rgb(image):
    return image.convert("RGB")
def get_data_transforms(size, isize):
    mean_train = (0.48145466, 0.4578275, 0.40821073)#[0.485, 0.456, 0.406]
    std_train = (0.26862954, 0.26130258, 0.27577711)#[0.229, 0.224, 0.225]
    data_transforms = transforms.Compose([
        transforms.Resize(size),
        #transforms.CenterCrop(isize),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        #transforms.CenterCrop(args.input_size),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    gt_transforms = transforms.Compose([
        transforms.Resize(size),
        #transforms.CenterCrop(isize),
        transforms.ToTensor()])
    return data_transforms, gt_transforms

def loss_func(pred):
    soft_loss = -pred[0]*pred[0].log()
    mask = torch.zeros(pred[1:].shape).to(pred[1:].device)
    mask[...,1] = 1
    hard_loss = -mask*pred[1:].log()-(1-mask)*pred[0].log()
    soft_loss = soft_loss.sum(-1).mean()
    hard_loss = hard_loss.sum(-1).mean()
    #alpha=0.9
    return 1*soft_loss+0.5*hard_loss

def weight_reset(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        m.reset_parameters()

def resize_tokens(x):
    B, N, C = x.shape
    x = x.view(B,int(math.sqrt(N)),int(math.sqrt(N)),C)
    return x

def image_tiling(x, scale=0.2):
    #B C H W
    B, C, H, W = x.shape
    H_scale = H*scale
    diff = W-H #263-240=23, 48
    #if diff<H_scale:
    #    return x
    x1 = x[:,:,:,:H]
    x2 = x[:,:,:,diff:]
    x = torch.cat((x1,x2),dim=0)
    return x, diff

def image_recover(x, diff):
    B, C, H, W = x.shape
    x_tmp = torch.zeros(1,C,H,W+diff)
    x_tmp[:,:,:,:diff] = x[:1,:,:,:diff]
    x_tmp[:,:,:,diff:W] = (x[:1,:,:,diff:]+x[1:,:,:,:W-diff].unsqueeze(0))/2
    x_tmp[:,:,:,W:] = x[1:,:,:,W-diff:].unsqueeze(0)
    return x_tmp


#from clip import clip

def test(args):
    img_size = args.image_size
    features_list = args.features_list
    few_shot_features = args.few_shot_features
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    txt_path = os.path.join(save_path, 'log.txt')
    # clip
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, img_size, pretrained=args.pretrained)

    data_transform, gt_transform = get_data_transforms(img_size, img_size)
    model.to(device)
    tokenizer = open_clip.get_tokenizer(args.model)
    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('test')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode='a')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # record parameters
    for arg in vars(args):
        if args.mode == 'zero_shot' and (arg == 'k_shot' or arg == 'few_shot_features'):
            continue
        logger.info(f'{arg}: {getattr(args, arg)}')
    # seg
    with open(args.config_path, 'r') as f:
        model_configs = json.load(f)
    linearlayer = LinearLayer(model_configs['vision_cfg']['width'], model_configs['embed_dim'],
                              len(features_list), args.model, model).to(device)
    # dataset
    transform = transforms.Compose([
            transforms.Resize((img_size)),
            #transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])
    if dataset_name == 'mvtec':
        test_data = MVTecDataset(root=dataset_dir, transform=data_transform, target_transform=transform,
                                 aug_rate=-1, mode='test')
    elif dataset_name == 'visa':
        test_data = VisaDataset(root=dataset_dir, transform=data_transform, target_transform=transform, mode='test')
    elif dataset_name == 'DIY':
        dataset_dir = '/sda/zhaoxiang_sda/outputs/defect_dataset'
        test_data = DIYDataset(root=dataset_dir, transform=data_transform, target_transform=transform, mode='test')
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=True)
    obj_list = test_data.get_cls_names()
    #obj_list = ["candle"]


    # text prompt
    with torch.no_grad():
        text_prompts, text_prompts_list = encode_text_with_prompt_ensemble(model, obj_list, tokenizer, device)
        text_prompts_segmentation = encode_text_with_prompt_ensemble_segmentation(model, obj_list, tokenizer, device)

    #adapters = {n:TextAdapter(text_prompts_list[n]).to(device) for n in obj_list}
    adapters = {n:Adapter(text_prompts_list[n]).to(device) for n in obj_list}
    #adapter = adapter.to(device)
    opts = {n:torch.optim.AdamW(adapters[n].parameters(), 1e-3) for n in obj_list}

    results = {}
    results['cls_names'] = []
    results['imgs_masks'] = []
    results['anomaly_maps'] = []
    results['foreground_map'] = []
    results['background_map'] = []
    results['gt_sp'] = []
    results['pr_sp'] = []
    results['fs_sp'] = []
    for items in test_dataloader:
        image, diff = image_tiling(items['img_defect'])#.to(device)
        #image = image_recover(image,diff)ran
        image = image.to(device)
        cls_name = items['cls_name']
        #if cls_name[0]!="candle":
        #   continue
        results['cls_names'].append(cls_name[0])
        gt_mask = items['img_mask']
        gt_mask[gt_mask > 0.5], gt_mask[gt_mask <= 0.5] = 1, 0
        results['imgs_masks'].append(gt_mask)  # px
        results['gt_sp'].append(items['anomaly'].item())

        #Adapter
        adapters[cls_name[0]].apply(weight_reset)
        with torch.no_grad():
            image_features, patch_tokens = model.encode_image(image, features_list)
        with torch.autograd.set_detect_anomaly(True):           # easier debug
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features = []
            for cls in cls_name:
                text_features.append(text_prompts[cls])
            text_features = torch.stack(text_features, dim=0)

            # sample
            text_probs = (image_features @ text_features[0]).softmax(dim=-1).mean(dim=0)
            results['pr_sp'].append(text_probs[0][1].cpu().item())          # prediction per

            # pixel
            #L, C = text_probs[0,1:].shape
            #H = int(np.sqrt(L))
            #anomaly_map = F.interpolate(text_probs[0,1:].permute(1, 0).view(1, 2, H, H),
            #                                size=img_size, mode='bilinear', align_corners=True)
            #anomaly_map = torch.softmax(anomaly_map, dim=1)[:, 1, :, :].cpu().numpy()
            with torch.no_grad():
                patch_tokens_ln = linearlayer(patch_tokens)
            
            # background & foreground segmentation
            for layer in range(len(patch_tokens_ln)):
                if layer!=6:#(layer%2)!=0:#(layer+1)//2!=0:
                    continue
                patch_token_segmentation = patch_tokens_ln[layer]
                patch_token_segmentation = resize_tokens(patch_token_segmentation)
                patch_token_segmentation = patch_token_segmentation/patch_token_segmentation.norm(dim=-1, keepdim=True)
                segmentation_text_token = text_prompts_segmentation[cls_name[0]]
                segmentation_map = (patch_token_segmentation @ segmentation_text_token.t())
                segmentation_map = F.interpolate(segmentation_map.permute(0, 3,1, 2),
                                                        size=img_size, mode='bilinear', align_corners=True)
                segmentation_map = image_recover(segmentation_map, diff)
                segmentation_map = segmentation_map.permute(0,2,3,1)
                segmentation_map = torch.softmax(segmentation_map*100, dim=-1)
                save_image(segmentation_map[...,0], 'foreground_segmentation.png')
                save_image(segmentation_map[...,1], 'background_segmentation.png')
                
                print(items['img_defect_path'][0])
            results['foreground_map'].append(segmentation_map[...,0])
            results['background_map'].append(segmentation_map[...,1])

            
            # if args.adapter:
            #     for i in range(args.epoch):
            #         anomaly_maps = []
            #         for layer in range(len(patch_tokens_ln)):
            #             if layer!=6:#(layer%2)!=0:#(layer+1)//2!=0:
            #                 continue
            #             tokens = resize_tokens(patch_tokens_ln[layer])
            #             #torch.nn.init.xavier_uniform_(adapter.ad.weight)
            #             tokens = adapters[cls_name[0]](tokens)
            #             tokens = tokens/tokens.norm(dim=-1, keepdim=True)
            #             anomaly_map = (tokens @ text_features)
            #             B, H,W, C = anomaly_map.shape
            #             #H = int(np.sqrt(L))

            #             anomaly_map = F.interpolate(anomaly_map.permute(0, 3,1, 2),
            #                                         size=img_size, mode='bilinear', align_corners=True)
            #             anomaly_map_n = image_recover(anomaly_map[:2], diff)
            #             anomaly_map_a = image_recover(anomaly_map[2:], diff)
            #             anomaly_map = torch.cat((anomaly_map_n,anomaly_map_a),dim=0)
            #             anomaly_map = anomaly_map.permute(0,2,3,1)
            #             anomaly_map = torch.softmax(anomaly_map, dim=-1)
            #             anomaly_maps.append(anomaly_map)
            #         anomaly_map = anomaly_map#torch.sum(torch.stack(anomaly_maps), axis=0)
            #         loss = loss_func(anomaly_map) #B,C,H,W
            #         opts[cls_name[0]].zero_grad()
            #         loss.backward()
            #         opts[cls_name[0]].step()

            # anomaly_maps = []
            # for layer in range(len(patch_tokens_ln)):
            #     if layer!=6:#layer%2!=0:#(layer+1)//2!=0:
            #         continue
            #     tokens = resize_tokens(patch_tokens_ln[layer])
            #     if args.adapter:
            #         with torch.no_grad():
            #             tokens = adapters[cls_name[0]](tokens, is_test=True)
            #     tokens /= tokens.norm(dim=-1, keepdim=True)
            #     anomaly_map = (tokens @ text_features)
            #     #B, L, C = anomaly_map.shape
            #     #H = int(np.sqrt(L))
            #     anomaly_map = torch.softmax(anomaly_map, dim=-1)[:,:,:,1].unsqueeze(1)
            #     anomaly_map = torch.nn.functional.pad(anomaly_map,(1,1,1,1),'replicate')
            #     anomaly_map = torch.nn.functional.avg_pool2d(anomaly_map, 3, stride=1, padding=0,count_include_pad=False)
            #     anomaly_map = F.interpolate(anomaly_map,
            #                                 size=img_size, mode='bilinear', align_corners=True)
            #     anomaly_map = image_recover(anomaly_map, diff)
            #     #anomaly_map = torch.nn.functional.pad(anomaly_map,(2,2,2,2),'replicate')
            #     #anomaly_map = torch.nn.functional.avg_pool2d(anomaly_map, 5, stride=1, padding=0,count_include_pad=False)
            #     anomaly_maps.append(anomaly_map[0])
            #     anomaly_map = anomaly_map[0].cpu().numpy()#torch.sum(torch.stack(anomaly_maps), axis=0)
            #     #anomaly_map = gaussian_filter(anomaly_map,sigma=4)
            #     #anomaly_map = torch.nn.functional.avg_pool2d(anomaly_map, 3, stride=1, padding=1)[0].cpu().numpy()
            # anomaly_map = anomaly_map

            
            # results['anomaly_maps'].append(anomaly_map)
            

            # visualization
            path = items['img_defect_path']
            cls = path[0].split('/')[-2]
            filename = path[0].split('/')[-1]
            # vis = cv2.cvtColor(cv2.resize(cv2.imread(path[0]), (anomaly_map.shape[-1], anomaly_map.shape[-2])), cv2.COLOR_BGR2RGB)  # RGB
            vis = cv2.cvtColor(cv2.imread(path[0]), cv2.COLOR_BGR2RGB)  # RGB
            # vis = vis#[18:-18,18:-18]
            # size = anomaly_map[0].shape
            # mask = anomaly_map[0]#[18:-18,18:-18]
            #mask = cv2.resize(mask,size)
            # mask = normalize(mask)
            # p, r, th = precision_recall_curve(gt_mask.ravel(), mask.ravel())
            # f1_score = (2 * p * r) / (p + r)
            # opt_th = th[np.argmax(f1_score)]
            # binary_mask = np.copy(mask)
            # binary_mask[binary_mask>=opt_th]=1
            # binary_mask[binary_mask<opt_th]=0
            # #mask = cv2.resize(mask,size)
            # #mask = normalize(mask)
            # #mask = cv2.copyMakeBorder(mask,16,16,16,16,cv2.BORDER_CONSTANT, value=(0,0,0))
            # crop_vis = apply_ad_bmap(vis, mask*binary_mask)
            # vis = apply_ad_scoremap(vis, mask)
            vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)  # BGR
            # crop_vis = cv2.cvtColor(crop_vis, cv2.COLOR_RGB2BGR)  # BGR
            save_vis = os.path.join(save_path, 'imgs', cls_name[0], cls)
            if not os.path.exists(save_vis):
                os.makedirs(save_vis)


            # save the visualized results
            if args.visualize:
                cv2.imwrite(os.path.join(save_vis, filename), vis)
                # cv2.imwrite(os.path.join(save_vis, "crop_"+filename), crop_vis)
                # cv2.imwrite(os.path.join(save_vis, "mask_"+filename), binary_mask*255)
                # cv2.imwrite(os.path.join(save_vis, "gt_"+filename), gt_mask[0,0].cpu().numpy()*255)
                cv2.imwrite(os.path.join(save_vis, "fore_"+filename), segmentation_map[0,:,:,0].cpu().numpy()*255)
                cv2.imwrite(os.path.join(save_vis, "back_"+filename), segmentation_map[0,:,:,1].cpu().numpy()*255)
    # # metrics
    # table_ls = []
    # auroc_sp_ls = []
    # auroc_px_ls = []
    # f1_sp_ls = []
    # f1_px_ls = []
    # aupro_ls = []
    # ap_sp_ls = []
    # ap_px_ls = []
    # for obj in obj_list:
    #     table = []
    #     gt_px = []
    #     pr_px = []
    #     gt_sp = []
    #     pr_sp = []
    #     pr_sp_tmp = []
    #     table.append(obj)
    #     for idxes in range(len(results['cls_names'])):
    #         if results['cls_names'][idxes] == obj:
    #             gt_px.append(results['imgs_masks'][idxes].squeeze(1).numpy())
    #             pr_px.append(results['anomaly_maps'][idxes])
    #             gt_sp.append(results['gt_sp'][idxes])
    #             pr_sp.append(results['pr_sp'][idxes])
    #             #print(results['imgs_masks'][idxes].squeeze(1).numpy().shape)
    #     gt_px = np.array(gt_px)
    #     gt_sp = np.array(gt_sp)
    #     pr_px = np.array(pr_px)
    #     pr_sp = np.array(pr_sp)

    #     auroc_px = roc_auc_score(gt_px.ravel(), pr_px.ravel())
    #     auroc_sp = roc_auc_score(gt_sp, pr_sp)
    #     #ap_sp = average_precision_score(gt_sp, pr_sp)
    #     ap_px = average_precision_score(gt_px.ravel(), pr_px.ravel())
    #     # f1_sp
    #     precisions, recalls, thresholds = precision_recall_curve(gt_sp, pr_sp)
    #     f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    #     f1_sp = np.max(f1_scores[np.isfinite(f1_scores)])
    #     # aupr
    #     aupr_sp = auc(recalls, precisions)
    #     # f1_px
    #     precisions, recalls, thresholds = precision_recall_curve(gt_px.ravel(), pr_px.ravel())
    #     f1_scores = (2 * precisions * recalls) / (precisions + recalls)
    #     f1_px = np.max(f1_scores[np.isfinite(f1_scores)])
    #     # aupro
    #     if len(gt_px.shape) == 4:
    #         gt_px = gt_px.squeeze(1)
    #     if len(pr_px.shape) == 4:
    #         pr_px = pr_px.squeeze(1)
    #     aupro = cal_pro_score(gt_px, pr_px)

    #     table.append(str(np.round(auroc_px * 100, decimals=1)))
    #     table.append(str(np.round(f1_px * 100, decimals=1)))
    #     table.append(str(np.round(ap_px * 100, decimals=1)))
    #     table.append(str(np.round(aupro * 100, decimals=1)))
    #     table.append(str(np.round(auroc_sp * 100, decimals=1)))
    #     table.append(str(np.round(f1_sp * 100, decimals=1)))
    #     table.append(str(np.round(aupr_sp * 100, decimals=1)))
        
        

    #     table_ls.append(table)
    #     auroc_sp_ls.append(auroc_sp)
    #     auroc_px_ls.append(auroc_px)
    #     f1_sp_ls.append(f1_sp)
    #     f1_px_ls.append(f1_px)
    #     aupro_ls.append(aupro)
    #     ap_sp_ls.append(aupr_sp)
    #     ap_px_ls.append(ap_px)
    
    
    

    # # logger
    # table_ls.append(['mean', str(np.round(np.mean(auroc_px_ls) * 100, decimals=1)),
    #                  str(np.round(np.mean(f1_px_ls) * 100, decimals=1)), str(np.round(np.mean(ap_px_ls) * 100, decimals=1)),
    #                  str(np.round(np.mean(aupro_ls) * 100, decimals=1)), str(np.round(np.mean(auroc_sp_ls) * 100, decimals=1)),
    #                  str(np.round(np.mean(f1_sp_ls) * 100, decimals=1)), str(np.round(np.mean(ap_sp_ls) * 100, decimals=1))])
    # results = tabulate(table_ls, headers=['objects', 'auroc_px', 'f1_px', 'ap_px', 'aupro', 'auroc_sp',
    #                                       'f1_sp', 'pr_sp'], tablefmt="pipe")
    
    # model_string = args.model
    # dataset_string = args.dataset
    # # feature_list = 
    # if args.adapter:
    #     adapter_string = 'adapter'
    # else:
    #     adapter_string = ''
    
    # experiment_name = model_string + '_' + dataset_string + '_' + adapter_string
    # result_txt_path = os.path.join(save_path, experiment_name + '.txt')
    # with open(result_txt_path, 'w') as file:
    #     file.write(results)

    
    # logger.info("\n%s", results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("VAND Challenge", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="/home/zhaoxiang/dataset/mvtec_anomaly_detection", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='/sda/zhaoxiang_sda/CLIP_AD/ZsadCLIP/results/segmentation', help='path to save results')
    parser.add_argument("--config_path", type=str, default='/home/zhaoxiang/CLIP_AD/ZsadCLIP/open_clip/model_configs/ViT-B-16.json', help="model configs")
    # model
    parser.add_argument("--dataset", type=str, default='DIY', help="test dataset")
    parser.add_argument("--model", type=str, default="ViT-B-16-plus-240", help="model used")
    parser.add_argument("--pretrained", type=str, default="laion400m_e32", help="pretrained weight used")
    parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6, 9, 12], help="features used")
    parser.add_argument("--few_shot_features", type=int, nargs="+", default=[3, 6, 9], help="features used for few shot")
    parser.add_argument("--image_size", type=int, default=240, help="image size")
    parser.add_argument("--mode", type=str, default="zero_shot", help="zero shot or few shot")
    parser.add_argument("--visualize", type=bool, default=True, help="zero shot or few shot")
    parser.add_argument("--adapter", type=bool, default=True, help="adapter")
    parser.add_argument("--epoch", type=int, default=5, help="epoch")
    # few shot
    parser.add_argument("--k_shot", type=int, default=10, help="e.g., 10-shot, 5-shot, 1-shot")
    parser.add_argument("--seed", type=int, default=111, help="random seed")
    args = parser.parse_args()

    setup_seed(args.seed)
    test(args)

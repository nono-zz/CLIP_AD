import argparse
from datasets import *
from utils.csv_utils import *
# from utils.metrics import *
from utils.training_utils import *
# from WinCLIP import *
# from utils.eval_utils import *
from test import test

from model_inference import ImageCLIP, TextCLIP
# specifically for clip surgery
import clip_zzx
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
import torch.nn as nn
BICUBIC = InterpolationMode.BICUBIC
import os

# specifically for clip
import open_clip


def run_winclip(classname, args):
    # prepare the experiment dir
    kwargs = vars(args)
    kwargs['class_name'] = classname
    kwargs['root_dir'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), kwargs['root_dir'])
    model_dir, img_dir, logger_dir, model_name, csv_path = get_dir_from_args(**kwargs)

    # get device
    if kwargs['use_cpu'] == 0:
            # device = f"cuda:1"
        device = torch.device("cuda")
    else:
        device = f"cpu"
    kwargs['device'] = device
    
    # model
    model, _ = clip_zzx.load("ViT-B/16", device=device)
    # model, _ = clip_zzx.load(args.backbone, device=device)

    # model, _, preprocess = open_clip.create_model_and_transforms(args.backbone, pretrained='laion400m_e32')
    # model_2, _, preprocess = open_clip.create_model_and_transforms("ViT-B/16")
    model = model.to(device)
    model_text = TextCLIP(model)
    model_image = ImageCLIP(model)
    model_text = torch.nn.DataParallel(model_text)
    model_image = torch.nn.DataParallel(model_image)
    model_image.eval()
    model_text.eval()

    
    preprocess =  Compose([Resize((224, 224), interpolation=BICUBIC), ToTensor(),
                    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    
    # get test dataloader
    test_dataloader, test_dataset_inst = get_dataloader_from_args(phase='test', preprocess=preprocess, perturbed=False,**kwargs)
    kwargs['out_size_h'] = kwargs['resolution']
    kwargs['out_size_w'] = kwargs['resolution']
    
    train_dataloader, train_dataset_inst = None, None
    
    # evaluation
    with torch.no_grad():
        metrics = test(model_text, model_image, preprocess, test_dataloader, device, is_vis=True, img_dir=img_dir,
                class_name=kwargs['class_name'], cal_pro=kwargs['cal_pro'], train_data=train_dataloader,
                resolution=kwargs['resolution'])
    # for k, v in metrics.items():
    #     logger.info(f"{kwargs['class_name']}======={k}: {v:.2f}")
        
    
    # save_metric(metrics, dataset_classes[kwargs['dataset']], kwargs['class_name'],
    #             kwargs['dataset'], csv_path)
    return
    
def get_args():
    parser = argparse.ArgumentParser(description='Anomaly detection')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'])
    parser.add_argument('--class-name', type=str, default='bottle')

    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=240)  # was 400

    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--vis', type=bool, choices=[True, False], default=True)
    parser.add_argument("--root-dir", type=str, default="result_clipSurgery")
    parser.add_argument("--load-memory", type=bool, default=True)
    parser.add_argument("--cal-pro", type=bool, default=False)
    parser.add_argument("--experiment_indx", type=int, default=1)
    parser.add_argument("--gpu-id", type=str, required=False, default=['0','1'])

    # pure test
    parser.add_argument("--pure-test", type=bool, default=False)

    # method related parameters
    parser.add_argument('--k-shot', type=int, default=0)
    parser.add_argument('--scales', nargs='+', type=tuple, default=(2, 3, 15)) 
    # parser.add_argument('--scales', nargs='+', type=int, default=(15))
    parser.add_argument('--attention_mode', type=str, choices=['vv', 'v', 'qkv'], default='v') 
    parser.add_argument("--backbone", type=str, default="ViT-B-16-plus-240",
                        choices=['ViT-B-16-plus-240'])
    parser.add_argument("--pretrained_dataset", type=str, default="laion400m_e32")

    parser.add_argument("--use-cpu", type=int, default=0)

    args = parser.parse_args()
    
    return args
    
    
if __name__ == '__main__':
    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    # os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    datasets = ['mvtec']
    
    if args.gpu_id is None:
        gpus = "0"
        os.environ["CUDA_VISIBLE_DEVICES"]= gpus
    else:
        gpus = ""
        for i in range(len(args.gpu_id)):
            gpus = gpus + args.gpu_id[i] + ","
        os.environ["CUDA_VISIBLE_DEVICES"]= gpus[:-1]

    torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performa
    
    datasets = ['mvtec']
    
    for dataset in datasets:
        classes = dataset_classes[dataset]
        # classes = ['bottle', 'carpet', 'cable']
        # classes = ['grid', 'leather', 'tile']
        for cls in classes[:]:
            with torch.cuda.device(args.gpu_id):
                run_winclip(classname=cls, args=args)
        
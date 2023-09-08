import argparse
from datasets import *
from utils.csv_utils import *
# from utils.metrics import *
from utils.training_utils import *
from WinCLIP import *
# from utils.eval_utils import *
from test import test


import os

def run_winclip(classname, args):
    # prepare the experiment dir
    kwargs = vars(args)
    kwargs['class_name'] = classname
    kwargs['root_dir'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), kwargs['root_dir'])
    model_dir, img_dir, logger_dir, model_name, csv_path = get_dir_from_args(**kwargs)

    # get device
    if kwargs['use_cpu'] == 0:
            device = f"cuda:0"
    else:
        device = f"cpu"
    kwargs['device'] = device
    
    # get test dataloader
    test_dataloader, test_dataset_inst = get_dataloader_from_args(phase='test', perturbed=False,**kwargs)
    kwargs['out_size_h'] = kwargs['resolution']
    kwargs['out_size_w'] = kwargs['resolution']
    
    train_dataloader, train_dataset_inst = None, None
    
    # model
    model = WinClipAD(**kwargs)
    model = model.to(device)

    # evaluation
    metrics = test(model, test_dataloader, device, is_vis=True, img_dir=img_dir,
            class_name=kwargs['class_name'], cal_pro=kwargs['cal_pro'], train_data=train_dataloader,
            resolution=kwargs['resolution'],
            sample_only=kwargs['sample_only'],
            prompt_template_mode = kwargs['prompt_mode'])
    for k, v in metrics.items():
        logger.info(f"{kwargs['class_name']}======={k}: {v:.2f}")
        
    
    save_metric(metrics, dataset_classes[kwargs['dataset']], kwargs['class_name'],
                kwargs['dataset'], csv_path)
    
    
def get_args():
    parser = argparse.ArgumentParser(description='Anomaly detection')
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa'])
    parser.add_argument('--class-name', type=str, default='bottle')

    # parser.add_argument('--img-resize', type=int, default=224)
    # parser.add_argument('--img-cropsize', type=int, default=224)
    # parser.add_argument('--resolution', type=int, default=224)  # was 400
    
    parser.add_argument('--img-resize', type=int, default=240)
    parser.add_argument('--img-cropsize', type=int, default=240)
    parser.add_argument('--resolution', type=int, default=240)  # wa

    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--vis', type=bool, choices=[True, False], default=True)
    parser.add_argument("--root-dir", type=str, default="result_winclip")
    parser.add_argument("--load-memory", type=bool, default=True)
    parser.add_argument("--cal-pro", type=bool, default=False)
    parser.add_argument("--experiment_indx", type=int, default=1)
    parser.add_argument("--gpu-id", type=int, default=0)

    # pure test
    parser.add_argument("--pure-test", type=bool, default=False)

    # method related parameters
    parser.add_argument('--k-shot', type=int, default=0)
    # parser.add_argument('--scales', nargs='+', type=tuple, default=(2, 3, 15)) 
    parser.add_argument('--scales', nargs='+', type=int, default=(15))
    parser.add_argument('--attention_mode', type=str, choices=['vv', 'v', 'qkv'], default='qkv') 
    parser.add_argument("--backbone", type=str, default="ViT-B-16-plus-240",
                        choices=['ViT-B-16-plus-240', 'ViT-B-16'])
    parser.add_argument("--pretrained_dataset", type=str, default="laion400m_e32", choices=['laion400m_e32', 'openai'])
    parser.add_argument("--sample_only", type=bool, default=True)
    parser.add_argument("--prompt_mode", type=str, default='hanqiu', choices=['hanqiu, winclip'])
    

    parser.add_argument("--use-cpu", type=int, default=0)

    args = parser.parse_args()
    
    return args
    
    
if __name__ == '__main__':
    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    datasets = ['mvtec']
    
    for dataset in datasets:
        classes = dataset_classes[dataset]
        for cls in classes[:]:
            run_winclip(classname=cls, args=args)
        
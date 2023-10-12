"""make variations of input image"""

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

import statistics

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from optimizer import build_optimizer
from lr_scheduler import build_lr_scheduler
import yaml

from torch.utils.data import Dataset, DataLoader
from dataset import CustomDataset

from yacs.config import CfgNode as CN
from cfg_defaults import _C as cfg_default


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    # print(f"loaded input image of size ({w}, {h}) from {path}")
    # w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    w, h = (256, 256)
    # w, h = (512, 512)
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def main(CATEGORY = None):
    parser = argparse.ArgumentParser()
    # CATEGORY = 'tile'
    CATEGORY = 'juice_bottle'
    # CATEGORY = 'pushpins'

    # parser.add_argument(
    #     "--prompt",
    #     type=str,
    #     nargs="?",
    #     # default="a painting of a virus monster playing guitar",
    #     # default=f"a photo of {CATEGORY} with texture or color defect region",
    #     # default=f"a photo of {CATEGORY} with broken or defect part for industrial anomaly detection",
    #     default= f"{CATEGORY} with defect",
    #     # default=f"a photo of {CATEGORY} with color defect region",
    #     help="the prompt to render"
    # )
    
    
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        # default=f"a photo of {CATEGORY} with logical anomalies",
        default=f"logical anomaly",
        # default=f"a photo of {CATEGORY} with more than 1 pins in one cell",
        # default=f"a photo of {CATEGORY} with logical anomaly on the bottom left part",
        # default=f"a photo of {CATEGORY} with color defect region",
        help="the prompt to render"
    )
    
    parser.add_argument(
        "--strength",
        type=float,
        # default=0.75,
        # default=0.05,
        # default=0.2,
        default=0.99,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )

    parser.add_argument(
        "--init-img",
        type=str,
        nargs="?",
        # default=f"/home/zhaoxiang/dataset/mvtec_anomaly_detection/{CATEGORY}/train/good/024.png",
        default=f"/home/zhaoxiang/dataset/mvtec_loco_anomaly_detection/{CATEGORY}/train/good/005.png",
        help="path to the input image"
    )
    

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )
    
    parser.add_argument(
        "--cls_name",
        type=str,
        nargs="?",
        help="dir to write results to",
        default=f"{CATEGORY}"
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/sda/zhaoxiang/CLIP_AD/stable-diffusion/configs/stable-diffusion/v1-prompt_learner_train.yaml",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/sda/zhaoxiang/CLIP_AD/stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        raise NotImplementedError("PLMS sampler not (yet) supported")
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, 'training', f"strength_{opt.strength}")
    os.makedirs(sample_path, exist_ok=True)
    # base_count = len(os.listdir(sample_path))
    base_count = 0
    grid_count = len(os.listdir(outpath)) - 1
    
    config_path = '/sda/zhaoxiang/CLIP_AD/stable-diffusion/scripts/train_config.yaml'
    # with open(config_path, 'r') as file:
    #     cfg = yaml.safe_load(file)
    # for key, value in cfg.items():
    #     setattr(cfg, key, value)
    cfg = cfg_default.clone()
    cfg.merge_from_file(config_path)

    optim = build_optimizer(model.cond_stage_model.prompt_learner, cfg.OPTIM)
    sched = build_lr_scheduler(optim, cfg.OPTIM)
    mse_loss = torch.nn.MSELoss()
    
    data_folder = '/home/zhaoxiang/dataset/mvtec_loco_anomaly_detection/breakfast_box/train/good/'
    custom_dataset = CustomDataset(data_folder)
    
    batch_size = 10
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(50):
        loss = []
        for img_paths in data_loader:
            imgs = []
            for img_path in img_paths:
                img = load_img(img_path).to(device)
                imgs.append(img)
            # init_image = torch.cat(imgs, 'b ... -> b ...', b=len(img_paths))
            init_image = torch.cat(imgs, dim=0)
            init_image.requires_grad_()
            # init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
            init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

            sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

            assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
            t_enc = int(opt.strength * opt.ddim_steps)
            print(f"target t_enc is {t_enc} steps")

            precision_scope = autocast if opt.precision == "autocast" else nullcontext
            with precision_scope("cuda"):
                with model.ema_scope():
                    tic = time.time()
                    all_samples = list()
                    for n in trange(opt.n_iter, desc="Sampling"):
                        for prompts in tqdm(data, desc="data"):
                            uc = None
                            if opt.scale != 1.0:
                                uc = model.get_learned_conditioning(batch_size * [""])
                            if isinstance(prompts, tuple):
                                prompts = list(prompts)
                            c = model.get_learned_conditioning(prompts)

                            # encode (scaled latent)
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                            # decode it
                            samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=uc,)

                            x_samples = model.decode_first_stage(samples)
                            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                            
                            rec_loss = mse_loss(x_samples, init_image)
                            loss.append(rec_loss.item())
                            
                            # step and update
                            optim.zero_grad()
                            rec_loss.backward()
                            optim.step()
                            print('loss is: {}'.format(statistics.mean(loss)))

                            if not opt.skip_save:
                                for x_sample in x_samples:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    Image.fromarray(x_sample.astype(np.uint8)).save(
                                        os.path.join(sample_path, f"{base_count:05}.png"))
                                    base_count += 1
                            all_samples.append(x_samples)

                    if not opt.skip_grid:
                        # additionally, save as grid
                        grid = torch.stack(all_samples, 0)
                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        grid = make_grid(grid, nrow=n_rows)

                        # to image
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
                        grid_count += 1
        print('loss for epoch[{}/50] is: {}'.format(epoch, statistics.mean(loss)))


if __name__ == "__main__":
    main()

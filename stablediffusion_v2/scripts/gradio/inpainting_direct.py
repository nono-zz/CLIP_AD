import sys
import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat
from imwatermark import WatermarkEncoder
from pathlib import Path

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
import os

from skimage.metrics import mean_squared_error

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from scripts.gradio.pipeline_utils import superpixel_segmentation, pad_and_resize, image_diference_check


torch.set_grad_enabled(False)


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)

    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    return sampler


def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
        "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
        "txt": num_samples * [txt],
        "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
    }
    return batch


def inpaint(sampler, image, mask, prompt, seed, scale, ddim_steps, num_samples=1, w=512, h=512):
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "SDV2"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h // 8, w // 8)
    start_code = torch.from_numpy(start_code).to(
        device=device, dtype=torch.float32)

    with torch.no_grad(), \
            torch.autocast("cuda"):
        batch = make_batch_sd(image, mask, txt=prompt,
                              device=device, num_samples=num_samples)

        c = model.cond_stage_model.encode(batch["txt"])

        c_cat = list()
        for ck in model.concat_keys:     #('mask', 'masked_image')
            cc = batch[ck].float()          # 'masked_image'
            if ck != model.masked_image_key:
                bchw = [num_samples, 4, h // 8, w // 8]
                cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
            else:
                cc = model.get_first_stage_encoding(
                    model.encode_first_stage(cc))
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)

        # cond
        cond = {"c_concat": [c_cat], "c_crossattn": [c]}

        # uncond cond
        uc_cross = model.get_unconditional_conditioning(num_samples, "")
        uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

        shape = [model.channels, h // 8, w // 8]
        samples_cfg, intermediates = sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=1.0,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_full,
            x_T=start_code,
        )
        x_samples_ddim = model.decode_first_stage(samples_cfg)

        result = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                             min=0.0, max=1.0)

        result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
    return [put_watermark(Image.fromarray(img.astype(np.uint8)), wm_encoder) for img in result]

def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    # im_resized = im_padded.resize((256, 256))
    im_resized = im_padded.resize((512, 512))
    # return im_padded
    return im_resized

def predict(input_image, prompt, ddim_steps, num_samples, scale, seed):
    init_image = input_image["image"].convert("RGB")
    init_mask = input_image["mask"].convert("RGB")
    image = pad_image(init_image) # resize to integer multiple of 32
    mask = pad_image(init_mask) # resize to integer multiple of 32
    width, height = image.size
    print("Inpainting...", width, height)

    result = inpaint(
        sampler=sampler,
        image=image,
        mask=mask,
        prompt=prompt,
        seed=seed,
        scale=scale,
        ddim_steps=ddim_steps,
        num_samples=num_samples,
        h=height, w=width
    )

    return result

# config = '/sda/zhaoxiang/CLIP_AD/stablediffusion_v2/configs/stable-diffusion/v2-inpainting-inference.yaml'
# ckpt = '/sda/zhaoxiang/CLIP_AD/stablediffusion_v2/checkpoints/512-inpainting-ema.ckpt'
ckpt = '/sda/zhaoxiang_sda/CLIP_AD/stablediffusion_v2/checkpoints/sd-v1-5-inpainting.ckpt'
config = '/sda/zhaoxiang_sda/CLIP_AD/stablediffusion_v2/configs/stable-diffusion/v1-inpainting-inference.yaml'

sampler = initialize_model(config, ckpt)

# block = gr.Blocks().queue()
# with block:
#     with gr.Row():
#         gr.Markdown("## Stable Diffusion Inpainting")

    # with gr.Row():
    #     with gr.Column():
    #         input_image = gr.Image(source='upload', tool='sketch', type="pil")
    #         prompt = gr.Textbox(label="Prompt")
    #         run_button = gr.Button(label="Run")
    #         with gr.Accordion("Advanced options", open=False):
    #             num_samples = gr.Slider(
    #                 label="Images", minimum=1, maximum=4, value=4, step=1)
    #             ddim_steps = gr.Slider(label="Steps", minimum=1,
    #                                    maximum=50, value=45, step=1)
    #             scale = gr.Slider(
    #                 label="Guidance Scale", minimum=0.1, maximum=30.0, value=10, step=0.1
    #             )
    #             seed = gr.Slider(
    #                 label="Seed",
    #                 minimum=0,
    #                 maximum=2147483647,
    #                 step=1,
    #                 randomize=True,
    #             )
    #     with gr.Column():
    #         gallery = gr.Gallery(label="Generated images", show_label=False).style(
    #             grid=[2], height="auto")

    # run_button.click(fn=predict, inputs=[
    #                  input_image, prompt, ddim_steps, num_samples, scale, seed], outputs=[gallery])

for j in range(100):

    CATEGORY = 'bottle'

    # set the prompt    
    # prompt = f"a photo of a {CATEGORY}, defect, logical mistake, damaged parts, broken, (hyper details), (realistic), 8k"
    prompt = f"defect, scratches, dents, colored spots, cracks, misplacement, missing parts, damaged, flaw, blemished, broken"

    # Example usage:
    image_path = f"/sda/zhaoxiang_sda/outputs/defect_dataset/images_raw/{CATEGORY}/00007.png"
    num_superpixels = 100
    target_superpixel = None
    target_size = (512, 512)

    image, superpixel_mask = superpixel_segmentation(image_path, num_superpixels, target_superpixel)
    resized_image, resized_mask = pad_and_resize(image, superpixel_mask, target_size)

    # resized_image.save('/sda/zhaoxiang_sda/outputs/inpainting/resized_image.jpg')
    mask_dir = '/sda/zhaoxiang_sda/outputs/inpainting/mask'
    mask_count = len(os.listdir(mask_dir))
    resized_mask.save(os.path.join(mask_dir, f"{mask_count:05}.png"))
    print('done')

    # set hyperparameters
    ddim_steps = 45
    num_samples = 4
    scale = 10
    seed = 1

    width, height = resized_image.size
    print("Inpainting...", width, height)
    save_dir = '/sda/zhaoxiang_sda/outputs/inpainting/results'
    base_count = len(os.listdir(save_dir))

    result_images = inpaint(
        sampler=sampler,
        image=resized_image,
        mask=resized_mask,
        prompt=prompt,
        seed=seed,
        scale=scale,
        ddim_steps=ddim_steps,
        num_samples=num_samples,
        h=height, w=width
    )

    for index, result_image in enumerate(result_images):
        # check image difference
        init_image_array = np.array(resized_image)
        result_image_array = np.array(result_image)
        flag = image_diference_check(init_image_array, result_image_array)
        
        if flag:
            result_image.save(os.path.join(save_dir, f"{mask_count}_{index:05}.png"))
        else:
            print("difference check failed")
    
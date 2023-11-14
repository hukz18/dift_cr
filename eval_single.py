import os
import re
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import logging

def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)
            
def get_cor_cfg(method):
    cor_cfg = {}
    if method == 'dift':
        cor_cfg['img_size'] = 768
        cor_cfg['ensemble_size'] = 8
    elif method == 'ldm_sc':
        cor_cfg['img_size'] = 512
    elif method == 'sd_dino':
        cor_cfg['model_type'] = 'dinov2_vitb14'
    return cor_cfg

def get_cor_pairs(method, model, src_image, trg_image, src_points, src_prompt, trg_prompt, cfg, device='cuda'):
    if method == 'dift':
        from get_cor import get_cor_pairs as get_cor_pairs_dift
        return get_cor_pairs_dift(model, src_image, trg_image, src_points, src_prompt, trg_prompt, cfg['img_size'], cfg['ensemble_size'], return_cos_maps=cfg['visualize'])
    elif method == 'ldm_sc':
        from baselines.ldm_sc.get_cor import get_cor_pairs as get_cor_paris_ldm_sc
        return get_cor_paris_ldm_sc(model, src_image, trg_image, src_points, cfg['img_size'], device), None
    elif method == 'sd_dino':
        from baselines.sd_dino.get_cor import get_cor_pairs as get_cor_pairs_sd_dino
        model, aug, extractor = model
        return get_cor_pairs_sd_dino(model, aug, extractor, src_image, trg_image, src_points, src_prompt, trg_prompt, device=device), None
    else:
        raise NotImplementedError

def get_model(method, cor_cfg, device='cuda'):
    if method == 'dift':
        from src.models.dift_sd import SDFeaturizer
        return SDFeaturizer(device)
    elif method == 'ldm_sc':
        from baselines.ldm_sc.optimize import load_ldm
        return load_ldm(device, 'CompVis/stable-diffusion-v1-4')
    elif method == 'sd_dino':
        from baselines.sd_dino.extractor_sd import load_model
        from baselines.sd_dino.extractor_dino import ViTExtractor
        model_type = cor_cfg['model_type']
        stride = 14 if 'v2' in model_type else 8
        extractor = ViTExtractor(model_type, stride, device=device)
        model, aug = load_model(diffusion_ver='v1-5', image_size=960, num_timesteps=100, block_indices=(2,5,8,11))
        return model, aug, extractor

def plot_img_pairs(imglist, src_points, trg_points, trg_mask=None, cos_maps=None, save_name='corr.png', fig_size=3, alpha=0.45, scatter_size=30):
    has_trg_mask = trg_mask is not None
    num_imgs = len(cos_maps) + 1 if cos_maps is not None else 2
    fig, axes = plt.subplots(1, num_imgs + has_trg_mask, figsize=(fig_size*(num_imgs + has_trg_mask), fig_size))
    plt.tight_layout()

    axes[0].imshow(imglist[0])
    axes[0].axis('off')
    axes[0].set_title('source image')
    for x, y in src_points:
        x, y = int(np.round(x)), int(np.round(y))
        axes[0].scatter(x, y, s=scatter_size)

    for i in range(1, num_imgs):
        axes[i].imshow(imglist[1])
        if cos_maps is not None:
            heatmap = cos_maps[i - 1][0]
            heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # Normalize to [0, 1]
            axes[i].imshow(255 * heatmap, alpha=alpha, cmap='viridis')
        axes[i].axis('off')
        axes[i].scatter(trg_points[i - 1][0], trg_points[i - 1][1], c='C%d' % (i - 1), s=scatter_size)
        axes[i].set_title('target image')
    
    if has_trg_mask:
        axes[-1].imshow(trg_mask, cmap='gray')
        axes[-1].axis('off')
        axes[-1].set_title('target mask')
    plt.plot()
    plt.savefig(save_name)
    plt.close()


def nearest_distance_to_mask_contour(mask, x, y):
    # Convert the boolean mask to an 8-bit image
    mask_8bit = (mask.astype(np.uint8) * 255)
    
    # Find the contours in the mask
    contours, _ = cv2.findContours(mask_8bit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if point is inside any contour
    num = 0
    for contour in contours:
        if cv2.pointPolygonTest(contour, (x, y), False) == 1:  # Inside contour
            num += 1
    if num % 2 == 1:
        return 0
    
    # If point is outside all contours, find the minimum distance between the point and each contour
    min_distance = float('inf')
    for contour in contours:
        distance = cv2.pointPolygonTest(contour, (x, y), True)  # Measure distance
        if distance < min_distance:
            min_distance = distance

    # normalize the distance with the diagonal length of the mask
    diag_len = np.sqrt(mask.shape[0]**2 + mask.shape[1]**2)
    return abs(min_distance) / diag_len


def analyze_dists(total_dists, dump_name=None):
    all_dists, lines = [], []
    for action in total_dists:
        action_dists = []
        for trg_object in total_dists[action]:
            all_dists += total_dists[action][trg_object]
            action_dists += total_dists[action][trg_object]
            lines.append(f'{trg_object.split("_")[0]:12s}: dist mean:{np.array(total_dists[action][trg_object]).mean():.3f}, success rate: {(np.array(total_dists[action][trg_object]) == 0).sum() / len(total_dists[action][trg_object]):.3f} ({(np.array(total_dists[action][trg_object]) == 0).sum()}/{len(total_dists[action][trg_object])})')
        lines.append(f'==={action.split("_")[0].upper().center(6)}===: dist mean:{np.mean(action_dists):.3f}, success rate: {(np.array(action_dists)==0).sum() / len(action_dists):.3f} ({(np.array(action_dists)==0).sum()}/{len(action_dists)})')
    lines.append(f'=== ALL  ===: dist mean:{np.mean(all_dists):.3f}, success rate: {(np.array(all_dists)==0).sum() / len(all_dists):.3f} ({(np.array(all_dists)==0).sum()}/{len(all_dists)})')
    if dump_name is not None:
        with open(dump_name, 'w') as f:
            f.writelines([line + '\n' for line in lines])
    for line in lines:
        print(line)

if __name__ == '__main__':
    method = 'dift'
    exp_name = 'secr_avg_pts'
    average_pts, visualize = True, True
    
    cor_cfg = get_cor_cfg(method)
    cor_cfg['visualize'] = True
    # set_global_logging_level(logging.ERROR)
    model = get_model(method, cor_cfg, device='cuda')
    # with open(f'os.path.join(res_dir, 'total_dists.pkl'), 'rb') as f:
    #     total_dists = pickle.load(f)
    src_image = 'datasets/eval_once/source.png'
    trg_image = 'datasets/eval_once/target.png'
    src_points = 'datasets/eval_once/source.txt'
    with open(src_points) as f:
        lines = f.readlines()
        src_points = [list(map(float, line.rstrip().split(','))) for line in lines if re.match(r'^\d+.\d+,.*\d+.\d+$', line.rstrip())]
        if average_pts:
            src_points = [np.mean(np.array(src_points), axis=0).astype(np.int32)]
    src_prompt = 'a photo of a spoon'
    trg_prompt = 'a photo of a flower'
    trg_points, cor_maps = get_cor_pairs(method, model, src_image, trg_image, src_points, src_prompt, trg_prompt, cor_cfg)
    print(f'{trg_points[0][0]}, {trg_points[0][1]}')
    imglist = [Image.open(file).convert('RGB') for file in [src_image, trg_image]]
    plot_img_pairs(imglist, src_points, trg_points, None, cor_maps, os.path.join('datasets/eval_once', f'result.png'))
    # total_dists = dataset_walkthrough(base_dir, method, model, exp_name, cor_cfg, average_pts, visualize)
    

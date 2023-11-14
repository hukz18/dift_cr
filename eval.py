import os
import re
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import logging
import argparse
import shutil

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
    elif method == 'dino_vit':
        cor_cfg['img_size'] = 224
        cor_cfg['model_type'] = 'dino_vits8'
        cor_cfg['stride'] = 4
    return cor_cfg

def get_cor_pairs(method, model, src_image, trg_image, src_points, src_prompt, trg_prompt, cfg, device='cuda'):
    if method == 'dift':
        from get_cor import get_cor_pairs
        return get_cor_pairs(model, src_image, trg_image, src_points, src_prompt, trg_prompt, cfg['img_size'], cfg['ensemble_size'], return_cos_maps=cfg['visualize'])
    elif method == 'ldm_sc':
        from baselines.ldm_sc.get_cor import get_cor_pairs
        return get_cor_pairs(model, src_image, trg_image, src_points, cfg['img_size'], device), None
    elif method == 'sd_dino':
        from baselines.sd_dino.get_cor import get_cor_pairs
        model, aug, extractor = model
        return get_cor_pairs(model, aug, extractor, src_image, trg_image, src_points, src_prompt, trg_prompt, device=device), None
    elif method == 'dino_vit':
        from baselines.dino_vit.get_cor import get_cor_pairs
        return get_cor_pairs(model, src_image, trg_image, src_points, cfg['img_size'], device=device)
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
    elif method == 'dino_vit':
        from baselines.dino_vit.extractor import ViTExtractor
        model_type = cor_cfg['model_type']
        stride = cor_cfg['stride']
        return ViTExtractor(model_type, stride, device=device)

def plot_img_pairs(imglist, src_points, trg_points, trg_mask, cos_maps=None, save_name='corr.png', fig_size=3, alpha=0.45, scatter_size=30):
    num_imgs = len(cos_maps) + 1 if cos_maps is not None else 2
    fig, axes = plt.subplots(1, num_imgs + 1, figsize=(fig_size*(num_imgs + 1), fig_size))
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
    
    axes[-1].imshow(trg_mask, cmap='gray')
    axes[-1].axis('off')
    axes[-1].set_title('target mask')
    plt.plot()
    plt.savefig(save_name)
    plt.close()


def nearest_distance_to_mask_contour(mask, x, y, mask_threshold):
    # Convert the boolean mask to an 8-bit image
     
    mask_8bit = ((mask > mask_threshold).astype(np.uint8) * 255)
    
    # Find the contours in the mask
    contours, _ = cv2.findContours(mask_8bit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Check if point is inside any contour
    num = 0
    for contour in contours:
        if cv2.pointPolygonTest(contour, (x, y), False) == 1:  # Inside contour
            num += 1
    if num % 2 == 1:
        return 0, np.array(mask)[int(y), int(x)]
    
    # If point is outside all contours, find the minimum distance between the point and each contour
    min_distance = float('inf')
    for contour in contours:
        distance = cv2.pointPolygonTest(contour, (x, y), True)  # Measure distance
        if distance < min_distance:
            min_distance = distance

    # normalize the distance with the diagonal length of the mask
    diag_len = np.sqrt(mask.shape[0]**2 + mask.shape[1]**2)
    return abs(min_distance) / diag_len, np.array(mask)[int(y), int(x)]


def dataset_walkthrough(base_dir, method, model, exp_name, cor_cfg={}, average_pts=True, visualize=False, mask_threshold=120):
    eval_pairs = 0
    total_dists, nss_values = {}, {}
    gt_dir = os.path.join(base_dir, 'GT')
    base_dir = os.path.join(base_dir, 'egocentric')
    
    for trg_object in os.listdir(base_dir):
        eval_pairs += len(os.listdir(os.path.join(base_dir, trg_object)))
    print(f'Start evaluating {eval_pairs} correspondance pairs...')
    
    cor_cfg['device'] = 'cuda'
    cor_cfg['visualize'] = visualize

    pbar = tqdm(total=eval_pairs)
    
    for trg_object in os.listdir(base_dir):
        object_path = os.path.join(base_dir, trg_object)
        total_dists[trg_object], nss_values[trg_object] = [], []
        for instance in os.listdir(object_path):
            instance_path = os.path.join(object_path, instance)
            for file in os.listdir(instance_path):
                if file.endswith('.jpg'):
                    trg_image = os.path.join(instance_path, file)
                    mask_file = os.path.join(gt_dir, trg_object, file.replace('jpg', 'png'))
                    with Image.open(mask_file) as img:
                        try:
                            trg_mask = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
                        except:
                            trg_mask = np.array(img)
                elif file.endswith('.txt'):
                    src_image = os.path.join(instance_path, file).replace('txt', 'png')
                    src_object = file.split('_')[0]
                    with open(os.path.join(instance_path, file), 'r') as f:
                        lines = f.readlines()
                        src_points = [list(map(float, line.rstrip().split(','))) for line in lines if re.match(r'^\d+.\d+,.*\d+.\d+$', line.rstrip())]
                        if average_pts:
                            src_points = [np.mean(np.array(src_points), axis=0).astype(np.int32)]
            pbar.set_description(f'{trg_object}-{instance}')
            prompts = [f'a photo of a {src_object}', f'a photo of {trg_object}']
            trg_points, cor_maps = get_cor_pairs(method, model, src_image, trg_image, src_points, prompts[0], prompts[1], cor_cfg)
            trg_point = np.mean(trg_points, axis=0)
            trg_dist, nss_value = nearest_distance_to_mask_contour(trg_mask, trg_point[0], trg_point[1], mask_threshold)
            total_dists[trg_object].append(trg_dist)
            nss_values[trg_object].append(nss_value)
            # print(trg_point, trg_dist)
            if visualize:
                res_dir =  f'results/{method}/{exp_name}/{trg_object}'
                shutil.rmtree(res_dir, ignore_errors=True)
                imglist = [Image.open(file).convert('RGB') for file in [src_image, trg_image]]
                os.makedirs(res_dir, exist_ok=True)
                plot_img_pairs(imglist, src_points, trg_points, trg_mask, cor_maps, os.path.join(res_dir, f'{instance}_{trg_dist:.2f}_{nss_value}.png'))
            pbar.update(1)
    pbar.close()
    return total_dists, nss_values


def analyze_dists(total_dists, nss_values, dump_name=None):
    all_dists, all_nss, lines = [], [], []
    for trg_object in total_dists.keys():
        all_dists += total_dists[trg_object]
        all_nss += nss_values[trg_object]
        lines.append(f'{trg_object.split("_")[0]:12s}: dist mean:{np.array(total_dists[trg_object]).mean():.3f}, nss mean: {np.array(nss_values[trg_object]).mean():.1f}, success rate: {(np.array(total_dists[trg_object]) == 0).sum() / len(total_dists[trg_object]):.3f} ({(np.array(total_dists[trg_object]) == 0).sum()}/{len(total_dists[trg_object])})')
    lines.append(f'=== ALL  ===: dist mean:{np.mean(all_dists):.3f}, nss mean: {np.array(all_nss).mean():.1f}, success rate: {(np.array(all_dists)==0).sum() / len(all_dists):.3f} ({(np.array(all_dists)==0).sum()}/{len(all_dists)})')
    if dump_name is not None:
        with open(dump_name, 'w') as f:
            f.writelines([line + '\n' for line in lines])
    for line in lines:
        print(line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', '-m', type=str, default='dift')
    parser.add_argument('--dataset', '-d', type=str, default='clip_b32_crop')
    parser.add_argument('--exp_name', '-e', type=str, default='')
    parser.add_argument('--mask_threshold', '-t', type=int, default=120)
    parser.add_argument('--visualize', '-v', action='store_true')
    parser.add_argument('--avg_pts', '-a', action='store_true')
    args = parser.parse_args()
    average_pts, visualize = args.avg_pts, args.visualize
    exp_name = args.dataset + '_' + str(args.mask_threshold) if len(args.exp_name) == 0 else args.exp_name  + '_' + str(args.mask_threshold)
    cor_cfg = get_cor_cfg(args.method)
    # set_global_logging_level(logging.ERROR)
    model = get_model(args.method, cor_cfg, device='cuda')
    base_dir = f'datasets/{args.dataset}'
    res_dir = f'results/{args.method}/{exp_name}'
    # with open(f'os.path.join(res_dir, 'total_dists.pkl'), 'rb') as f:
    #     total_dists = pickle.load(f)
    
    total_dists, nss_values = dataset_walkthrough(base_dir, args.method, model, exp_name, cor_cfg, average_pts, visualize, args.mask_threshold)
    with open(os.path.join(res_dir, 'results.pkl'), 'wb') as f:
        pickle.dump({'total_dists': total_dists, 'nss_values': nss_values}, f)

    analyze_dists(total_dists, nss_values, os.path.join(res_dir, 'total_dists.txt'))

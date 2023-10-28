import os
import re
import cv2
import torch
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm
from get_corr import get_corr_pairs
from extractor_sd import load_model
import matplotlib.pyplot as plt
from extractor_dino import ViTExtractor

def nearest_distance_to_mask_contour(mask, x, y):
    # Convert the boolean mask to an 8-bit image
    mask_8bit = (mask.astype(np.uint8) * 255)
    
    # Find the contours in the mask
    contours, _ = cv2.findContours(mask_8bit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    
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

def analyze_dists(total_dists):
    all_dists = []
    for action in total_dists:
        action_dists = []
        for trg_object in total_dists[action]:
            all_dists += total_dists[action][trg_object]
            action_dists += total_dists[action][trg_object]
            print(f'{trg_object.split("_")[0]:12s}: dist mean:{np.array(total_dists[action][trg_object]).mean():.3f}, success rate: {(np.array(total_dists[action][trg_object]) == 0).sum() / len(total_dists[action][trg_object]):.3f} ({(np.array(total_dists[action][trg_object]) == 0).sum()}/{len(total_dists[action][trg_object])})')
        print(f'==={action.split("_")[0].upper().center(6)}===: dist mean:{np.mean(action_dists):.3f}, success rate: {(np.array(action_dists)==0).sum() / len(action_dists):.3f} ({(np.array(action_dists)==0).sum()}/{len(action_dists)})')
    print(f'=== ALL  ===: dist mean:{np.mean(all_dists):.3f}, success rate: {(np.array(all_dists)==0).sum() / len(all_dists):.3f} ({(np.array(all_dists)==0).sum()}/{len(all_dists)})')


def plot_img_pairs(imglist, points, trg_mask, save_name='corr.png', fig_size=3, alpha=0.45, scatter_size=30):
    fig, axes = plt.subplots(1, 3, figsize=(fig_size * 3, fig_size))
    plt.tight_layout()

    for i in range(2):
        axes[i].imshow(imglist[i])
        axes[i].axis('off')
        axes[i].set_title('source image' if i == 0 else 'target image')
        for x, y in points[i]:
            x, y = int(np.round(x)), int(np.round(y))
            axes[i].scatter(x, y, s=scatter_size)
    
    axes[-1].imshow(trg_mask, cmap='gray')
    axes[-1].axis('off')
    axes[-1].set_title('target mask')
    plt.plot()
    plt.savefig(save_name)
    plt.close()
    
def dataset_walkthrough(model, aug, extractor, exp_name, visualize=False, average_pts=True):
    base_dir, gt_dir = 'eval_all/egocentric', 'eval_all/GT'
    total_dists = {}
    for action in (pbar_a := tqdm(os.listdir(base_dir))):
        pbar_a.set_description(action)
        action_path = os.path.join(base_dir, action)
        total_dists[action] = {}
        for trg_object in (pbar_o := tqdm(os.listdir(action_path), leave=False)):
            pbar_o.set_description(trg_object)
            object_path = os.path.join(action_path, trg_object)
            total_dists[action][trg_object] = []
            for instance in (pbar_i := tqdm(os.listdir(object_path), leave=False)):
                pbar_i.set_description(instance)
                instance_path = os.path.join(object_path, instance)
                for file in os.listdir(instance_path):
                    if file.endswith('.jpg'):
                        trg_image = os.path.join(instance_path, file)
                        mask_file = os.path.join(gt_dir, action, trg_object, file.replace('jpg', 'png'))
                        with Image.open(mask_file) as img:
                            trg_mask = np.array(img) > 122
                    elif file.endswith('.txt'):
                        src_image = os.path.join(instance_path, file).replace('txt', 'png')
                        src_object = file.split('_')[0]
                        with open(os.path.join(instance_path, file), 'r') as f:
                            lines = f.readlines()
                            src_points = [list(map(float, line.rstrip().split(','))) for line in lines if re.match(r'^\d+.\d+,\d+.\d+$', line.rstrip())]
                            if average_pts:
                                src_points = [np.mean(np.array(src_points), axis=0).astype(np.int32)]
                src_prompt, trg_prompt = [f'a photo of a {src_object}', f'a photo of {trg_object}']
                trg_points = get_corr_pairs(model, aug, extractor, src_image, trg_image, src_points, src_prompt, trg_prompt)
                trg_point = np.mean(trg_points, axis=0)
                trg_dist = nearest_distance_to_mask_contour(trg_mask, trg_point[0], trg_point[1])
                total_dists[action][trg_object].append(trg_dist)
                if visualize:
                    imglist = [Image.open(file).convert('RGB') for file in [src_image, trg_image]]
                    os.makedirs(f'results/baselines/sd_dino/{exp_name}/{action}/{trg_object}', exist_ok=True)
                    plot_img_pairs(imglist, [src_points, trg_points], trg_mask, f'results/baselines/sd_dino/{exp_name}/{action}/{trg_object}/{instance}_{trg_dist:.2f}.png')
    return total_dists

if __name__ == '__main__':
    visualize = True
    exp_name, average_pts = 'no_avg_pts', False
    model_type = 'dinov2_vitb14'
    stride = 14 if 'v2' in model_type else 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(model_type, stride, device=device)
    model, aug = load_model(diffusion_ver='v1-5', image_size=960, num_timesteps=100, block_indices=(2,5,8,11))
    total_dists = dataset_walkthrough(model, aug, extractor, exp_name, visualize, average_pts)
    analyze_dists(total_dists)
    with open(f'results/baselines/sd_dino/{exp_name}/total_dists.pkl', 'wb') as f:
        pickle.dump(total_dists, f)
    
    
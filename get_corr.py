import gc
import os
import cv2
import torch
import pickle
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision.transforms import PILToTensor
from src.models.dift_sd import SDFeaturizer
import matplotlib.pyplot as plt
from tqdm import tqdm
import re

def dataset_walkthrough(dift, img_size, ensemble_size, exp_name, average_pts=True, visualize=False):
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
                prompts = [f'a photo of a {src_object}', f'a photo of {trg_object}']
                trg_points, cor_maps = get_cor_pairs(dift, src_image, trg_image, src_points, prompts[0], prompts[1], img_size, ensemble_size, average_pts, return_cos_maps=visualize)
                trg_point = np.mean(trg_points, axis=0)
                trg_dist = nearest_distance_to_mask_contour(trg_mask, trg_point[0], trg_point[1])
                total_dists[action][trg_object].append(trg_dist)
                print(trg_point, trg_dist)
                if visualize:
                    imglist = [Image.open(file).convert('RGB') for file in [src_image, trg_image]]
                    os.makedirs(f'results/{exp_name}/{action}/{trg_object}', exist_ok=True)
                    plot_img_pairs(imglist, src_points, trg_points, cor_maps, trg_mask, f'results/{exp_name}/{action}/{trg_object}/{instance}_{trg_dist:.2f}.png')
    return total_dists


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


def plot_img_pairs(imglist, src_points, trg_points, cos_maps, trg_mask, save_name='corr.png', fig_size=3, alpha=0.45, scatter_size=30):
    num_imgs = len(cos_maps) + 1
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

def get_cor_pairs(dift, src_image, trg_image, src_points, src_prompt, trg_prompt, img_size, ensemble_size, average_pts=True, return_cos_maps=False):
    """
    src_image, trg_image: relative path of src and trg images
    src_points: resized affordance points in src_image
    average_pts: average before correspondance or not
    -----
    return: correspondance maps of each src_point and each target_point
    """
    trg_points = []
    
    with Image.open(src_image) as img:
        src_w, src_h = img.size
        src_image = img.resize((img_size, img_size)).convert('RGB')
        src_x_scale, src_y_scale = img_size / src_w, img_size / src_h
    with Image.open(trg_image) as img:
        trg_w, trg_h = img.size
        trg_image = img.resize((img_size, img_size)).convert('RGB')
        trg_x_scale, trg_y_scale = img_size / trg_w, img_size / trg_h
    
    src_points = [[int(np.round(x * src_x_scale)), int(np.round(y * src_y_scale))] for (x, y) in src_points]
    
    src_tensor = (PILToTensor()(src_image) / 255.0 - 0.5) * 2
    trg_tensor = (PILToTensor()(trg_image) / 255.0 - 0.5) * 2
    src_ft = dift.forward(src_tensor, prompt=src_prompt, ensemble_size=ensemble_size)
    trg_ft = dift.forward(trg_tensor, prompt=trg_prompt, ensemble_size=ensemble_size)
    num_channel = src_ft.size(1)
    cos = nn.CosineSimilarity(dim=1)
    
    if average_pts:
        src_points = [np.mean(np.array(src_points), axis=0).astype(np.int32)]
    
    src_ft = nn.Upsample(size=(img_size, img_size), mode='bilinear')(src_ft)
    src_vectors = [src_ft[0, :, y, x].view(1, num_channel, 1, 1) for (x, y) in src_points]
    del src_ft
    gc.collect()
    torch.cuda.empty_cache()
    
    trg_ft = nn.Upsample(size=(img_size, img_size), mode='bilinear')(trg_ft)
    cos_maps = [cos(src_vec, trg_ft).cpu().numpy() for src_vec in src_vectors]
    del trg_ft
    gc.collect()
    torch.cuda.empty_cache()
    
    for cos_map in cos_maps:
        max_yx = np.unravel_index(cos_map.argmax(), cos_map.shape)[1:]
        trg_points.append([max_yx[1], max_yx[0]])
    
    trg_points = [[int(np.round(x / trg_x_scale)), int(np.round(y / trg_y_scale))] for (x, y) in trg_points]
    cos_maps = [nn.Upsample(size=(trg_h, trg_w), mode='bilinear')(torch.tensor(cos_map).view(1, 1, img_size, img_size)).numpy()[0] for cos_map in cos_maps] if return_cos_maps else None
    
    return trg_points, cos_maps


if __name__ == '__main__':
    dift = SDFeaturizer()
    ft, imglist = [], []

    img_size = 768
    ensemble_size = 8
    exp_name = 'avg_pts'
    average_pts, visualize = True, True
    # with open('results/{exp_name}/total_dists.pkl', 'rb') as f:
    #     total_dists = pickle.load(f)
    total_dists = dataset_walkthrough(dift, img_size, ensemble_size, exp_name, average_pts, visualize)
    with open(f'results/{exp_name}/total_dists.pkl', 'wb') as f:
        pickle.dump(total_dists, f)
    analyze_dists(total_dists)

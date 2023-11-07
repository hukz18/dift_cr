import os
import re
import cv2
import torch
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from get_corr import get_cor_pairs
from src.models.dift_sd import SDFeaturizer
import multiprocessing
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool



def init_worker():
    """This will be called each time a worker is created."""
    torch.set_grad_enabled(False)
    
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


# one list to store action, trg_object, one list for rets, if list is full pop all and get return
# add a "eval one" option
def dataset_walkthrough(dift_list, img_size, ensemble_size, exp_name, average_pts=True, visualize=False, num_workers=1):
    eval_pairs, attached_pairs = 0, 0
    total_dists = {}
    base_dir, gt_dir = 'eval_secr/egocentric', 'eval_secr/GT'
    
    for action in os.listdir(base_dir):
        for trg_object in os.listdir(os.path.join(base_dir, action)):
            eval_pairs += len(os.listdir(os.path.join(base_dir, action, trg_object)))
    print(f'Start evaluating {eval_pairs} correspondance pairs...')
    
    ret_list, param_buffer = [], []
    free_gpus = [i for i in range(num_workers)]
    
    pbar = tqdm(total=eval_pairs)
    workers_pool = Pool(num_workers, initializer=init_worker)
    for action in os.listdir(base_dir):
        action_path = os.path.join(base_dir, action)
        total_dists[action] = {}
        for trg_object in os.listdir(action_path):
            object_path = os.path.join(action_path, trg_object)
            total_dists[action][trg_object] = []
            for instance in os.listdir(object_path):
                pbar.set_description(f'{action}-{trg_object}-{instance}')
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
                            src_points = [list(map(float, line.rstrip().split(','))) for line in lines if re.match(r'^\d+.\d+,.*\d+.\d+$', line.rstrip())]
                prompts = [f'a photo of a {src_object}', f'a photo of {trg_object}']
                gpu_id = free_gpus.pop()
                ret_list.append(workers_pool.apply_async(get_cor_pairs, args=(dift_list[gpu_id], src_image, trg_image, src_points, prompts[0], prompts[1], img_size, ensemble_size, average_pts, visualize)))
                param_buffer.append((src_points, trg_mask, trg_object, instance, action, src_image, trg_image))
                attached_pairs += 1
                if len(free_gpus) == 0 or attached_pairs == eval_pairs:
                    while len(ret_list) > 0:
                        trg_points, cor_maps = ret_list.pop(0).get()
                        trg_point = np.mean(trg_points, axis=0)
                        src_points, trg_mask, trg_object, instance, action, src_image, trg_image = param_buffer.pop(0)
                        trg_dist = nearest_distance_to_mask_contour(trg_mask, trg_point[0], trg_point[1])
                        total_dists[action][trg_object].append(trg_dist)
                        if visualize:
                            imglist = [Image.open(file).convert('RGB') for file in [src_image, trg_image]]
                            os.makedirs(f'results/{exp_name}/{action}/{trg_object}', exist_ok=True)
                            plot_img_pairs(imglist, src_points, trg_points, trg_mask, cor_maps, f'results/{exp_name}/{action}/{trg_object}/{instance}_{trg_dist:.2f}.png')
                        pbar.update(1)
                    free_gpus = [i for i in range(num_workers)]
    pbar.close()
    workers_pool.terminate()
    return total_dists


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
    num_workers = 8
    multiprocessing.set_start_method("spawn")
    # with Pool(num_workers, initializer=init_worker) as p:
    #     models = p.map(SDFeaturizer, ['cuda:%d' % i for i in range(num_workers)])
    thread_pool = ThreadPool(num_workers)
    models = thread_pool.map(SDFeaturizer, ['cuda:%d' % i for i in range(num_workers)])
    thread_pool.terminate()
    
    print(f'Done initializing models on {num_workers} GPUs')
    
    ft, imglist = [], []

    img_size = 768
    ensemble_size = 8
    exp_name = 'avg_pts_secr'
    average_pts, visualize = True, True
    # with open(f'results/{exp_name}/total_dists.pkl', 'rb') as f:
    #     total_dists = pickle.load(f)
    total_dists = dataset_walkthrough(models, img_size, ensemble_size, exp_name, average_pts, visualize, num_workers)
    with open(f'results/{exp_name}/total_dists.pkl', 'wb') as f:
        pickle.dump(total_dists, f)
    analyze_dists(total_dists, f'results/{exp_name}/total_dists.txt')

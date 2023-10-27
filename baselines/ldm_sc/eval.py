import re
import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from optimize import optimize_prompt, run_image_with_tokens_cropped, find_max_pixel_value, load_ldm


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


def get_cor_pairs(ldm, src_image, trg_image, src_points, img_size):
    """
    src_image, trg_image: relative path of src and trg images
    src_points: resized affordance points in src_image
    average_pts: average before correspondance or not
    -----
    return: correspondance maps of each src_point and each target_point
    """
    trg_points = []
    layers = [5,6,7,8]
    with Image.open(src_image) as img:
        src_w, src_h = img.size
        src_image = img.resize((img_size, img_size), Image.BILINEAR).convert('RGB')
        src_tensor = torch.Tensor(np.array(src_image).transpose(2, 0, 1)) / 255.0
        src_x_scale, src_y_scale = img_size / src_w, img_size / src_h
    with Image.open(trg_image) as img:
        trg_w, trg_h = img.size
        trg_image = img.resize((img_size, img_size), Image.BILINEAR).convert('RGB')
        trg_tensor = torch.Tensor(np.array(trg_image).transpose(2, 0, 1)) / 255.0
        trg_x_scale, trg_y_scale = img_size / trg_w, img_size / trg_h
    
    src_points = [torch.Tensor([int(np.round(x * src_x_scale)), int(np.round(y * src_y_scale))]) for (x, y) in src_points]
    all_contexts = []
    for src_point in src_points:
        contexts = []
        for _ in range(5):
            context = optimize_prompt(ldm, src_tensor, src_point/img_size, num_steps=129, device='cuda:0', layers=layers, lr = 0.0023755632081200314, upsample_res=img_size, noise_level=-8, sigma = 27.97853316316864, flip_prob=0.0, crop_percent=93.16549294381423)
            contexts.append(context)
        all_contexts.append(torch.stack(contexts))
        
        all_maps = []
        for context in contexts:
            maps = []
            attn_maps, _ = run_image_with_tokens_cropped(ldm, trg_tensor, context, index=0, upsample_res = img_size, noise_level=-8, layers=layers, device='cuda:0', crop_percent=93.16549294381423, num_iterations=20, image_mask = None)
            for k in range(attn_maps.shape[0]):
                avg = torch.mean(attn_maps[k], dim=0, keepdim=True)
                maps.append(avg)
            maps = torch.stack(maps, dim=0)
            all_maps.append(maps)
        all_maps = torch.stack(all_maps, dim=0)
        all_maps = torch.mean(all_maps, dim=0)
        all_maps = torch.nn.Softmax(dim=-1)(all_maps.reshape(len(layers), img_size*img_size))
        all_maps = all_maps.reshape(len(layers), img_size, img_size)
        
        all_maps = torch.mean(all_maps, dim=0)
        trg_points.append(find_max_pixel_value(all_maps, img_size = img_size).cpu().numpy())
    
    
    trg_points = [[int(np.round(x / trg_x_scale)), int(np.round(y / trg_y_scale))] for (x, y) in trg_points]
    
    return trg_points

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
    
    
def dataset_walkthrough(ldm, img_size, exp_name, visualize=False, average_pts=True):
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
                        with open(os.path.join(instance_path, file), 'r') as f:
                            lines = f.readlines()
                            src_points = [list(map(float, line.rstrip().split(','))) for line in lines if re.match(r'^\d+.\d+,\d+.\d+$', line.rstrip())]
                            if average_pts:
                                src_points = [np.mean(np.array(src_points), axis=0).astype(np.int32)]
                trg_points = get_cor_pairs(ldm, src_image, trg_image, src_points, img_size)
                trg_point = np.mean(trg_points, axis=0)
                trg_dist = nearest_distance_to_mask_contour(trg_mask, trg_point[0], trg_point[1])
                total_dists[action][trg_object].append(trg_dist)
                if visualize:
                    imglist = [Image.open(file).convert('RGB') for file in [src_image, trg_image]]
                    os.makedirs(f'results/baselines/ldm_sc/{exp_name}/{action}/{trg_object}_{trg_dist:.2f}', exist_ok=True)
                    plot_img_pairs(imglist, [src_points, trg_points], trg_mask, f'results/baselines/ldm_sc/{exp_name}/{action}/{trg_object}/{instance}.png')
    return total_dists


if __name__ == '__main__':
    img_size = 512
    visualize, average_pts = False, True
    exp_name = 'no_avg_pts'
    ldm = load_ldm('cuda:0', 'CompVis/stable-diffusion-v1-4')
    # src_image = 'eval_all/egocentric/drag/suitcase/suitcase_000529/cabinet_01.png'
    # trg_image = 'eval_all/egocentric/drag/suitcase/suitcase_000529/suitcase_000529.jpg'
    # with open('eval_all/egocentric/drag/suitcase/suitcase_000529/cabinet_01.txt', 'r') as f:
    #     lines = f.readlines()
    #     src_points = [list(map(float, line.rstrip().split(','))) for line in lines if re.match(r'^\d+.\d+,\d+.\d+$', line.rstrip())]
    # src_points = [np.mean(src_points, axis=0).astype(np.int32)]
    dataset_walkthrough(ldm, img_size, exp_name, visualize, average_pts)
    
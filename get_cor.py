import gc
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from torchvision.transforms import PILToTensor


def get_cor_pairs(dift, src_image, trg_image, src_points, src_prompt, trg_prompt, img_size, ensemble_size=8, return_cos_maps=False):
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


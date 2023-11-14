import torch
from PIL import Image


def get_cor_pairs(extractor, src_image: str, trg_image: str, src_points: list, load_size: int = 224, layer: int = 9, 
                  facet: str = 'key', bin: bool = True, device='cuda:0'):

    # extracting descriptors for each image
    src_image_batch, src_image_pil = extractor.preprocess(src_image, load_size)

    src_image = Image.open(src_image)
    src_image_width, src_image_height = src_image.size

    descriptors_src = extractor.extract_descriptors(src_image_batch.to(device), layer, facet, bin)
    num_patches_src, _ = extractor.num_patches, extractor.load_size

    trg_image_batch, _ = extractor.preprocess(trg_image, load_size)
    descriptors_trg = extractor.extract_descriptors(trg_image_batch.to(device), layer, facet, bin)
    num_patches_trg, _ = extractor.num_patches, extractor.load_size

    # calculate similarity between src_image and trg_image descriptors
    similarities = chunk_cosine_sim(descriptors_src, descriptors_trg)

    # calculate best buddies
    image_idxs = torch.arange(num_patches_src[0] * num_patches_src[1], device=device)
    sim_src, nn_src = torch.max(similarities, dim=-1)  # nn_1 - indices of block2 closest to block1
    sim_trg, nn_trg = torch.max(similarities, dim=-2)  # nn_2 - indices of block1 closest to block2
    sim_src, nn_src = sim_src[0, 0], nn_src[0, 0]
    sim_trg, nn_trg = sim_trg[0, 0], nn_trg[0, 0]

    indices_to_show = []
    for i in range(len(src_points)):
        transferred_x1 = (src_points[i][0]*src_image_pil.size[0]/src_image_width - extractor.stride[1] - extractor.p // 2)/extractor.stride[1] + 1
        transferred_y1 = (src_points[i][1]*src_image_pil.size[1]/src_image_height - extractor.stride[0] - extractor.p // 2)/extractor.stride[0] + 1
        indices_to_show.append(int(transferred_y1) * num_patches_src[1] + int(transferred_x1))
    
    src_img_indices_to_show = torch.arange(num_patches_src[0] * num_patches_src[1], device=device)[indices_to_show]
    trg_img_indices_to_show = nn_src[indices_to_show]
    # coordinates in descriptor map's dimensions
    src_img_y_to_show = (src_img_indices_to_show / num_patches_src[1]).cpu().numpy()
    src_img_x_to_show = (src_img_indices_to_show % num_patches_src[1]).cpu().numpy()
    trg_img_y_to_show = (trg_img_indices_to_show / num_patches_trg[1]).cpu().numpy()
    trg_img_x_to_show = (trg_img_indices_to_show % num_patches_trg[1]).cpu().numpy()
    points_src, points_trg = [], []
    for y1, x1, y2, x2 in zip(src_img_y_to_show, src_img_x_to_show, trg_img_y_to_show, trg_img_x_to_show):
        x_src_show = (int(x1) - 1) * extractor.stride[1] + extractor.stride[1] + extractor.p // 2
        y_src_show = (int(y1) - 1) * extractor.stride[0] + extractor.stride[0] + extractor.p // 2
        x_trg_show = (int(x2) - 1) * extractor.stride[1] + extractor.stride[1] + extractor.p // 2
        y_trg_show = (int(y2) - 1) * extractor.stride[0] + extractor.stride[0] + extractor.p // 2
        points_src.append([y_src_show, x_src_show])
        points_trg.append([y_trg_show, x_trg_show])
    return  points_trg

def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y) """
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)

import torch
from torch.optim import Adam, AdamW

from .optimizer import MultiOptimizer

def generate_recon_samples(image_id, width, height):
    """
    generate model input samples for image reconstruction
    """
    grid_y = torch.linspace(0, 1, height)
    grid_x = torch.linspace(0, 1, width)
    axis_y, axis_x = torch.meshgrid(grid_y, grid_x, indexing='ij')
    # C H W
    uv = torch.stack((axis_x, axis_y))
    samples = torch.cat([torch.ones(1, height, width)*float(image_id), uv]).view(3, -1)
    samples = samples.permute(1, 0)
    return samples

def reconstruct_image(model, image_id, width, height):
    """
    reconstruct image using model
    """
    samples = generate_recon_samples(image_id, width, height)
    samples = samples.to(next(model.parameters()).device)
    out = model(samples)
    out = out.permute(1, 0)
    return out.view(3, height, width)

def compute_psnr(img_1, img_2):
    """
    compute psnr score for two images, operate on tensors
    """
    if img_1.dtype == torch.uint8:
        img_1 = img_1.float() / 255.

    if img_2.dtype == torch.uint8:
        img_2 = img_2.float() / 255.

    squared_error = torch.square(img_1 - img_2)
    mse = torch.mean(squared_error)
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr

def compute_dataset_psnr(model, dataset):
    """
    compute psnr score for all model reconstructed images with original images in dataset.
    returns a list of psnr scores.
    """
    psnr = []
    for image_id in dataset.images.keys():
        data = dataset.images[image_id]["data"]
        shape = dataset.images[image_id]["shape"]
        rec_img = reconstruct_image(model, image_id, shape[2], shape[1])
        score = compute_psnr(data, rec_img.cpu()).item()
        psnr.append(score)
    return psnr

def create_optimizer(model, adam_params, adamw_params):
    """
    create customized optimizer for the model.
    for embeddings, use Adam, for mlp, use AdamW.
    """
    adam_optim = Adam(model.embeddings.parameters(), **adam_params)
    adamw_optim = AdamW(model.mlp.parameters(), **adamw_params)
    return MultiOptimizer({"embeddings": adam_optim, "mlp": adamw_optim})

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from sklearn.decomposition import PCA

folder = ""#"trained_models/dog/v1/pnp/v5_3200s_sks_dog_10f"

def visualize_resnet(hidden_states, visualization_dict):
    #hs = svd_compression(hidden_states)
    for i in range(hidden_states.shape[0]):
        hs = hidden_states[i]

        hs, sum = svd_compression(hs)

        hs = hs.reshape(hs.shape[0], -1)
        hs = hs.cpu().numpy()
        hs = hs.T

        pca = PCA(n_components=3)
        pca.fit(hs)
        pca_img = pca.transform(hs)

        h = w = int(np.sqrt(pca_img.shape[0]))
        pca_img = pca_img.reshape(h, w, 3)
        pca_img_min = pca_img.min(axis=(0, 1))
        pca_img_max = pca_img.max(axis=(0, 1))
        pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)
        pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
        pca_img = T.Resize(512, interpolation=T.InterpolationMode.NEAREST)(pca_img)

        im_name = f"resnet_timestep_{visualization_dict['timestep']}_{visualization_dict['blocktype']}_subblock{visualization_dict['subblock']}_im{i+1}_{sum}.png"
        
        pca_img.save(folder + "/" + im_name)

def visualize_attention(query, key, visualization_dict):
    for i in range(query.shape[0]):
        q = query[i]
        k = key[i]
        a = torch.matmul(q, k.T)
        a = torch.nn.functional.softmax(a, dim=1)

        a, sum = svd_compression_attn(a)

        a = a.cpu().numpy()

        pca = PCA(n_components=3)
        pca.fit(a)
        pca_img = pca.transform(a)

        h = w = int(np.sqrt(pca_img.shape[0]))
        pca_img = pca_img.reshape(h, w, 3)
        pca_img_min = pca_img.min(axis=(0, 1))
        pca_img_max = pca_img.max(axis=(0, 1))
        pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)
        pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
        pca_img = T.Resize(512, interpolation=T.InterpolationMode.NEAREST)(pca_img)

        im_name = f"attn_timestep_{visualization_dict['timestep']}_{visualization_dict['blocktype']}_subblock{visualization_dict['subblock']}_im{i+1}_{sum}.png"
        
        pca_img.save(folder + "/" + im_name)

def svd_compression(matrix):
    dtype = matrix.dtype
    matrix = matrix.to(matrix.device, dtype=torch.float32)

    c, h, w = matrix.shape
    matrix = matrix.reshape(matrix.shape[0], -1)

    u, s, v = torch.linalg.svd(matrix, full_matrices=True)

    if matrix.shape[0] >= matrix.shape[1]:
        padding = torch.zeros(u.shape[1] - s.shape[0], s.shape[0])
        dim = 0
    else:
        padding = torch.zeros(s.shape[0], v.shape[1] - s.shape[0])
        dim = 1
    padding = padding.to(matrix.device)

    #S[:-1] = 0.0
    comp_factor = 10
    comp_factor = int(torch.min(torch.Tensor([comp_factor, s.shape[0]])).item())
    replace_mat = torch.zeros(comp_factor).to(matrix.device)
    #s[-comp_factor:] = replace_mat
    s[:comp_factor] = replace_mat

 
    s_diag = torch.diag(s)
    # s_diag = s_diag[None, :, :]
    # for i in range(s.shape[0] - 1):
    #     s_temp = torch.diag(s[i + 1])
    #     s_temp = s_temp[None, :, :]
    #     s_diag = torch.cat((s_diag, s_temp), dim=0)
    s_diag = torch.cat((s_diag, padding), dim=dim)

    L = torch.matmul(u, s_diag)
    matrix = torch.matmul(L, v)
    matrix = matrix.reshape(c, h, w)
    matrix = matrix.to(matrix.device, dtype=dtype)

    return matrix, torch.sum(matrix).item()

def svd_compression_attn(matrix):
    dtype = matrix.dtype
    matrix = matrix.to(matrix.device, dtype=torch.float32)

    u, s, v = torch.linalg.svd(matrix, full_matrices=True)

    if matrix.shape[0] >= matrix.shape[1]:
        padding = torch.zeros(u.shape[1] - s.shape[0], s.shape[0])
        dim = 0
    else:
        padding = torch.zeros(s.shape[0], v.shape[1] - s.shape[0])
        dim = 1
    padding = padding.to(matrix.device)

    #S[:-1] = 0.0
    comp_factor = 10
    comp_factor = int(torch.min(torch.Tensor([comp_factor, s.shape[0]])).item())
    replace_mat = torch.zeros(comp_factor).to(matrix.device)
    #s[-comp_factor:] = replace_mat
    s[:comp_factor] = replace_mat

 
    s_diag = torch.diag(s)
    # s_diag = s_diag[None, :, :]
    # for i in range(s.shape[0] - 1):
    #     s_temp = torch.diag(s[i + 1])
    #     s_temp = s_temp[None, :, :]
    #     s_diag = torch.cat((s_diag, s_temp), dim=0)
    s_diag = torch.cat((s_diag, padding), dim=dim)

    L = torch.matmul(u, s_diag)
    matrix = torch.matmul(L, v)

    matrix = matrix.to(matrix.device, dtype=dtype)

    return matrix, torch.sum(matrix).item()
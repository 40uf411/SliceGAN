import os
from torch import nn
import torch
from torch import autograd
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import sys
import datetime
## Training Utils

def mkdr(proj,proj_dir,Training):
    """
    When training, creates a new project directory or overwrites an existing directory according to user input. When testing, returns the full project path
    :param proj: project name
    :param proj_dir: project directory
    :param Training: whether new training run or testing image
    :return: full project path
    """
    pth = proj_dir + '/' + proj
    if Training:
        if not os.path.exists(pth):
            os.makedirs(pth)
            return pth
        else:
            dt = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            new_pth = f"{pth}_{dt}"
            os.makedirs(new_pth)
            return new_pth
    else:
        return pth


def weights_init(m):
    """
    Initialises training weights
    :param m: Convolution to be intialised
    :return:
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def calc_gradient_penalty(netD, real_data, fake_data, batch_size, l, device, gp_lambda,nc):
    """
    calculate gradient penalty for a batch of real and fake data
    :param netD: Discriminator network
    :param real_data:
    :param fake_data:
    :param batch_size:
    :param l: image size
    :param device:
    :param gp_lambda: learning parameter for GP
    :param nc: channels
    :return: gradient penalty
    """
    #sample and reshape random numbers
    alpha = torch.rand(batch_size, 1, device = device)
    alpha = alpha.expand(batch_size, int(real_data.nelement() / batch_size)).contiguous()
    alpha = alpha.view(batch_size, nc, l, l)

    # create interpolate dataset
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())
    interpolates.requires_grad_(True)

    #pass interpolates through netD
    disc_interpolates = netD(interpolates)
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size(), device = device),
                              create_graph=True, only_inputs=True)[0]
    # extract the grads and calculate gp
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty


def calc_eta(steps, time, start, i, epoch, num_epochs):
    """
    Estimates the time remaining based on the elapsed time and epochs
    :param steps:
    :param time: current time
    :param start: start time
    :param i: iteration through this epoch
    :param epoch:
    :param num_epochs: totale no. of epochs
    """
    elap = time - start
    progress = epoch * steps + i + 1
    rem = num_epochs * steps - progress
    ETA = rem / progress * elap
    hrs = int(ETA / 3600)
    mins = int((ETA / 3600 % 1) * 60)
    print('[%d/%d][%d/%d]\tETA: %d hrs %d mins'
          % (epoch, num_epochs, i, steps,
             hrs, mins))

## Plotting Utils
def post_proc(img,imtype):
    """
    turns one hot image back into grayscale
    :param img: input image
    :param imtype: image type
    :return: plottable image in the same form as the training data
    """
    try:
        #make sure it's one the cpu and detached from grads for plotting purposes
        img = img.detach().cpu()
    except:
        pass
    if imtype == 'colour':
        return np.int_(255 * (np.swapaxes(img[0], 0, -1)))
    if imtype == 'grayscale':
        return 255*img[0][0]
    else:
        nphase = img.shape[1]
        return 255*torch.argmax(img, 1)/(nphase-1)
        
def test_plotter(img,slcs,imtype,pth):
    """
    creates a fig with 3*slc subplots showing example slices along the three axes
    :param img: raw input image
    :param slcs: number of slices to take in each dir
    :param imtype: image type
    :param pth: where to save plot
    """
    img = post_proc(img,imtype)
    fig, axs = plt.subplots(slcs, 3)
    if imtype == 'colour':
        for j in range(slcs):
            axs[j, 0].imshow(img[j, :, :, :], vmin = 0, vmax = 255)
            axs[j, 1].imshow(img[:, j, :, :],  vmin = 0, vmax = 255)
            axs[j, 2].imshow(img[:, :, j, :],  vmin = 0, vmax = 255)
    elif imtype == 'grayscale':
        for j in range(slcs):
            axs[j, 0].imshow(img[j, :, :], cmap = 'gray')
            axs[j, 1].imshow(img[:, j, :], cmap = 'gray')
            axs[j, 2].imshow(img[:, :, j], cmap = 'gray')
    else:
        for j in range(slcs):
            axs[j, 0].imshow(img[j, :, :])
            axs[j, 1].imshow(img[:, j, :])
            axs[j, 2].imshow(img[:, :, j])
    plt.savefig(pth + '_slices.png')
    plt.close()

def graph_plot(data,labels,pth,name):
    """
    simple plotter for all the different graphs
    :param data: a list of data arrays
    :param labels: a list of plot labels
    :param pth: where to save plots
    :param name: name of the plot figure
    :return:
    """

    for datum,lbl in zip(data,labels):
        plt.plot(datum, label = lbl)
    plt.legend()
    plt.savefig(pth + '_' + name)
    plt.close()


def test_img(pth, imtype, netG, nz = 64, lf = 4, periodic=False):
    """
    saves a test volume for a trained or in progress of training generator
    :param pth: where to save image and also where to find the generator
    :param imtype: image type
    :param netG: Loaded generator class
    :param nz: latent z dimension
    :param lf: length factor
    :param show:
    :param periodic: list of periodicity in axis 1 through n
    :return:
    """
    netG.load_state_dict(torch.load(pth + '_Gen.pt'))
    netG.eval()
    netG.cuda()
    noise = torch.randn(1, nz, lf, lf, lf).cuda()
    if periodic:
        if periodic[0]:
            noise[:, :, :2] = noise[:, :, -2:]
        if periodic[1]:
            noise[:, :, :, :2] = noise[:, :, :, -2:]
        if periodic[2]:
            noise[:, :, :, :, :2] = noise[:, :, :, :, -2:]
    with torch.no_grad():
        raw = netG(noise)
    print('Postprocessing')
    gb = post_proc(raw,imtype)[0]
    if periodic:
        if periodic[0]:
            gb = gb[:-1]
        if periodic[1]:
            gb = gb[:,:-1]
        if periodic[2]:
            gb = gb[:,:,:-1]
    tif = np.int_(gb)
    tifffile.imwrite(pth + '.tif', tif)

    return tif, raw, netG






import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from slicegan import model, networks, util
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import tifffile
import time
# from slicegan import preprocessing, util # Assuming these are available
# from slicegan.util import post_proc # No longer needed if we use GPU version


def post_proc_gpu(raw_tensor, imtype):
    """
    Optimized GPU-based post-processing for a generated tensor.
    Assumes grayscale output and scales to 0-255 integers.
    """
    processed = raw_tensor.detach()

    # Generic Grayscale Optimization: Scale and convert to integer
    if imtype == 'grayscale':
        # Assuming Tanh (-1 to 1) or Sigmoid (0 to 1) output
        if processed.min() >= -1.0 and processed.max() <= 1.0:
            processed = (processed + 1.0) / 2.0 * 255.0
        elif processed.min() >= 0.0 and processed.max() <= 1.0:
            processed = processed * 255.0

        processed = torch.clamp(processed, 0, 255).to(torch.int16)

    # Remove the channel dimension (C)
    processed = processed.squeeze(1)

    return processed




def sample_and_analyze_optimized(pth, imtype, netG_class, nz, lf, N_samples=16, micro_batch_size=4):
    """
    Optimized function for generating, analyzing, and saving samples.
    Performs generation, post-processing, and concatenation entirely on the GPU.
    Includes verbose print statements to track execution flow.
    """
    start_total = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Starting Optimized Sampling and Analysis ---")
    print(f"Target Device: {device}")
    print(f"Total Samples: {N_samples}, Micro-Batch Size: {micro_batch_size}, Volume Size: {lf}x{lf}x{lf}")

    # 1. Setup Generator
    try:
        start_load = time.time()
        generator = netG_class().to(device)
        generator.load_state_dict(torch.load(pth + '_Gen.pt', map_location=device))
        generator.eval()
        print(f"STATUS: Generator loaded in {time.time() - start_load:.4f}s.")
    except Exception as e:
        print(f"ERROR: Could not load generator: {e}")
        return None

    all_raw_tensors = [] # Stores tensors on the GPU

    # 2. Sample N Images using Micro-Batching (All on GPU)
    print(f"STATUS: Entering generation loop...")
    with torch.no_grad():
        for i in range(0, N_samples, micro_batch_size):
            start_batch = time.time()
            current_batch_size = min(micro_batch_size, N_samples - i)
            batch_start_index = i

            # Forward Pass
            print(f"  BATCH {batch_start_index}/{N_samples}: Generating noise on GPU...")
            noise = torch.randn(current_batch_size, nz, lf, lf, lf, device=device)

            print(f"  BATCH {batch_start_index}/{N_samples}: Starting forward pass on GPU...")
            if device.type == 'cuda': torch.cuda.synchronize()
            start_forward = time.time()
            raw_batch = generator(noise)
            if device.type == 'cuda': torch.cuda.synchronize()
            print(f"  BATCH {batch_start_index}/{N_samples}: Forward pass finished in {time.time() - start_forward:.4f}s. Output shape: {raw_batch.shape}")

            # Post-Processing
            print(f"  BATCH {batch_start_index}/{N_samples}: Starting post-processing on GPU...")
            start_postproc = time.time()
            processed_batch = post_proc_gpu(raw_batch, imtype) # Output: (B, D, H, W) on GPU
            if device.type == 'cuda': torch.cuda.synchronize()
            print(f"  BATCH {batch_start_index}/{N_samples}: Post-processing finished in {time.time() - start_postproc:.4f}s.")

            # Store and Print Status
            all_raw_tensors.append(processed_batch)
            print(f"  BATCH {batch_start_index}/{N_samples}: Batch finished in {time.time() - start_batch:.4f}s. Stored {len(all_raw_tensors) * current_batch_size}/{N_samples} total.")

    # 3. Concatenate and Transfer (The point of max data movement)
    print("\nSTATUS: Concatenating all tensors on GPU...")
    start_concat = time.time()
    concatenated_gpu_tensor = torch.cat(all_raw_tensors, dim=0)
    if device.type == 'cuda': torch.cuda.synchronize()
    print(f"STATUS: Concatenation finished in {time.time() - start_concat:.4f}s. Starting GPU -> CPU transfer...")

    start_transfer = time.time()
    concatenated_samples = concatenated_gpu_tensor.cpu().numpy()
    end_transfer = time.time()
    print(f"STATUS: GPU-to-CPU transfer complete in {end_transfer - start_transfer:.4f}s. Final NumPy shape: {concatenated_samples.shape}")

    # 4. Save the concatenated images as a single numpy array
    start_save_npy = time.time()
    np.save(pth + f'_N{N_samples}_samples_LF{lf}.npy', concatenated_samples)
    print(f"STATUS: NumPy array saved in {time.time() - start_save_npy:.4f}s to {pth}_N{N_samples}_samples_LF{lf}.npy")

    # 5. Visualization and Analysis (CPU-Bound Operations)

    if N_samples < 2:
        print("STATUS: Need at least 2 samples for visualization, skipping analysis plots.")
        return concatenated_samples

    rand_indices = random.sample(range(N_samples), 2)
    sample_a = concatenated_samples[rand_indices[0]]
    sample_b = concatenated_samples[rand_indices[1]]

    # --- Histogram Plot ---
    start_hist = time.time()
    plt.figure(figsize=(8, 6))
    hist_a, bins = np.histogram(sample_a.flatten(), bins=256, range=(0, 256))
    hist_b, _ = np.histogram(sample_b.flatten(), bins=256, range=(0, 256))
    avg_hist = (hist_a + hist_b) / 2
    bin_centers = (bins[:-1] + bins[1:]) / 2
    plt.plot(bin_centers, avg_hist)
    plt.title(f'Average Histogram (Indices: {rand_indices[0]}, {rand_indices[1]})')
    plt.xlabel('Voxel Value'); plt.ylabel('Frequency')
    plt.savefig(pth + '_Avg_Histogram.png'); plt.close()
    print(f"STATUS: Histogram plot saved in {time.time() - start_hist:.4f}s.")

    # --- 6x6 Slice Grid Plot ---
    start_slice_plot = time.time()
    n_slices_to_plot = 6

    def plot_slices(sample, fig_title, save_name):
        L = sample.shape[0]
        indices = np.linspace(0, L - 1, n_slices_to_plot, dtype=int)
        fig, axes = plt.subplots(3, n_slices_to_plot, figsize=(15, 7.5))
        fig.suptitle(fig_title, fontsize=16)
        directions = ['X-Normal (YZ-plane)', 'Y-Normal (XZ-plane)', 'Z-Normal (XY-plane)']

        for d, direction in enumerate(directions):
            for i, index in enumerate(indices):
                ax = axes[d, i]
                if d == 0: slice_data = sample[index, :, :]
                elif d == 1: slice_data = sample[:, index, :]
                else: slice_data = sample[:, :, index]

                ax.imshow(slice_data, cmap='gray'); ax.axis('off')
                if i == 0: ax.set_title(direction, fontsize=10, loc='left')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(pth + save_name)
        plt.close(fig)

    plot_slices(sample_a, f'Sample A (Index {rand_indices[0]})', '_SampleA_Slices.png')
    plot_slices(sample_b, f'Sample B (Index {rand_indices[1]})', '_SampleB_Slices.png')
    print(f"STATUS: Slice grid plots saved in {time.time() - start_slice_plot:.4f}s.")

    print(f"\n--- Total Execution Time: {time.time() - start_total:.4f}s ---")
    return concatenated_samples

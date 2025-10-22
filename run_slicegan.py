### Welcome to SliceGAN ###
####### Steve Kench #######
'''
Use this file to define your settings for a training run, or
to generate a synthetic image using a trained generator.
'''

from slicegan import model, networks, util

# Define project name
Project_name = 'NMC'
# Specify project folder.
Project_dir = 'Trained_Generators'

# Simple configuration: replace argparse with a small Config class.
# Set `training` to 1 to run training, or 0 to run testing.
class Config:
    training = False

config = Config()
Training = config.training

Project_path = util.mkdr(Project_name, Project_dir, Training)

## Data Processing
# Define image  type (colour, grayscale, three-phase or two-phase.
# n-phase materials must be segmented)
image_type = 'grayscale'
# img_channels should be number of phases for nphase, 3 for colour, or 1 for grayscale
img_channels = 1
# define data type (for colour/grayscale images, must be 'colour' / '
# greyscale. nphase can be, 'tif2D', 'png', 'jpg', tif3D, 'array')
data_type = 'tif2D'
# Path to your data. One string for isotrpic, 3 for anisotropic
data_path = ['/home/ucl/elen/aaouf/lemmens_slices/slice_x_center+0.tiff',
             '/home/ucl/elen/aaouf/lemmens_slices/slice_y_center+0.tiff',
             '/home/ucl/elen/aaouf/lemmens_slices/slice_z_center+0.tiff']

## Network Architectures
# Training image size, no. channels and scale factor vs raw data
img_size, scale_factor = 64,  1
# z vector depth
z_channels = 32
# Layers in G and D
lays = 5
laysd = 6
dk, gk = [4]*laysd, [4]*lays                                    # kernal sizes
# gk[0]=8
ds, gs = [2]*laysd, [2]*lays                                    # strides
# gs[0] = 4
df, gf = [img_channels, 64, 128, 256, 512, 1], [
    z_channels, 1024, 512, 128, 32, img_channels]  # filter sizes for hidden layers

dp, gp = [1, 1, 1, 1, 0], [2, 2, 2, 2, 3]

## Create Networks
netD, netG = networks.slicegan_rc_nets(Project_path, Training, image_type, dk, ds, df,dp, gk ,gs, gf, gp)

# Train
if Training:
    model.train(Project_path, image_type, data_type, data_path, netD, netG, img_channels, img_size, z_channels, scale_factor)
else:
    # Configuration for the new sampling function
    N_SAMPLES = 2  # Total number of samples you want to generate
    MICRO_BATCH = 1 # Generate 4 samples at a time to save memory
    LENGTH_FACTOR = 4 # Use lf=64 or smaller, based on previous memory fix

    # The netG passed here is the *class* (networks.Generator), which the
    # sample_and_analyze function will instantiate and load weights for.
    final_samples = util.sample_and_analyze_optimized(
        Project_path, 
        image_type, 
        netG, # Note: Pass the class, not netG()
        z_channels, 
        LENGTH_FACTOR,  
        N_samples=N_SAMPLES,
        micro_batch_size=MICRO_BATCH
    )

    # Optional: Add any logic to handle the final_samples numpy array here if needed.
    if final_samples is not None:
         print(f"Sampling and analysis complete. Final array shape: {final_samples.shape}")

import numpy as np
import torch
import matplotlib.pyplot as plt
import tifffile
def batch(data,type,l, sf):
    """
    Generate a batch of images randomly sampled from a training microstructure
    :param data: data path
    :param type: data type
    :param l: image size
    :param sf: scale factor
    :return:
    """
    Testing = True
    if type in ['png', 'jpg', 'tif2D']:
        datasetxyz = []
        for img in data:
            img = plt.imread(img) if type != 'tif2D' else tifffile.imread(img)
            if img.ndim > 2:
                img = img[..., 0]
            img = img.astype(np.float32)
            minv, maxv = float(img.min()), float(img.max())
            if maxv > minv:
                img = (img - minv) / (maxv - minv)
            else:
                img = np.zeros_like(img, dtype=np.float32) 
            if len(img.shape)>2:
                print('converting to grayscale')
                img = img[:,:,0]
                print("new: ", img.shape)
            img = img[::sf,::sf]
            x_max, y_max= img.shape[:]
            # use torch (and cuda if available)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            n_samples = 32 * 900

            # initialize data array with single channel
            data = np.empty([n_samples, 1, l, l])

            # vectorise random index generation
            xs = np.random.randint(1, x_max - l - 1, size=n_samples)
            ys = np.random.randint(1, y_max - l - 1, size=n_samples)

            for i, (x, y) in enumerate(zip(xs, ys)):
                patch = img[x:x + l, y:y + l]
                data[i, 0] = patch

            if Testing:
                # Create a 6x6 subplot grid of random images
                fig, axes = plt.subplots(6, 6, figsize=(12, 12))
                random_indices = np.random.choice(data.shape[0], size=36, replace=False)
                for idx, ax in enumerate(axes.flat):
                    ax.imshow(data[random_indices[idx], 0, :, :], cmap='gray')
                    ax.axis('off')
                plt.tight_layout()
                plt.savefig('training_samples_grid.png', dpi=150, bbox_inches='tight')
                plt.close()
                # save the average histogram of the first 10 samples
                plt.figure()
                tmp_data = data[:10, 0, :, :].reshape(-1)
                plt.hist(tmp_data, bins=50, density=True)
                plt.title('Histogram of first 10 samples')
                plt.xlabel('Pixel Intensity')
                plt.ylabel('Density')
                plt.savefig('training_samples_histogram.png', dpi=150, bbox_inches='tight')
                plt.close()

            data = torch.FloatTensor(data)
            dataset = torch.utils.data.TensorDataset(data)
            datasetxyz.append(dataset)

    elif type=='tif3D':
        datasetxyz=[]
        img = np.array(tifffile.imread(data[0]))
        img = img[::sf,::sf,::sf]
        ## Create a data store and add random samples from the full image
        x_max, y_max, z_max = img.shape[:]
        print('training image shape: ', img.shape)
        vals = np.unique(img)
        for dim in range(3):
            data = np.empty([32 * 900, len(vals), l, l])
            print('dataset ', dim)
            for i in range(32*900):
                x = np.random.randint(0, x_max - l)
                y = np.random.randint(0, y_max - l)
                z = np.random.randint(0, z_max - l)
                # create one channel per phase for one hot encoding
                lay = np.random.randint(img.shape[dim]-1)
                for cnt,phs in enumerate(list(vals)):
                    img1 = np.zeros([l,l])
                    if dim==0:
                        img1[img[lay, y:y + l, z:z + l] == phs] = 1
                    elif dim==1:
                        img1[img[x:x + l,lay, z:z + l] == phs] = 1
                    else:
                        img1[img[x:x + l, y:y + l,lay] == phs] = 1
                    data[i, cnt, :, :] = img1[:,:]
                    # data[i, (cnt+1)%3, :, :] = img1[:,:]

            if Testing:
                for j in range(2):
                    plt.imshow(data[j, 0, :, :] + 2 * data[j, 1, :, :])
                    plt.pause(1)
                    plt.show()
                    plt.clf()
                plt.close()
            data = torch.FloatTensor(data)
            dataset = torch.utils.data.TensorDataset(data)
            datasetxyz.append(dataset)

    elif type=='colour':
        ## Create a data store and add random samples from the full image
        datasetxyz = []
        for img in data:
            img = plt.imread(img)
            img = img[::sf,::sf,:]
            ep_sz = 32 * 900
            data = np.empty([ep_sz, 3, l, l])
            x_max, y_max = img.shape[:2]
            for i in range(ep_sz):
                x = np.random.randint(0, x_max - l)
                y = np.random.randint(0, y_max - l)
                # create one channel per phase for one hot encoding
                data[i, 0, :, :] = img[x:x + l, y:y + l,0]
                data[i, 1, :, :] = img[x:x + l, y:y + l,1]
                data[i, 2, :, :] = img[x:x + l, y:y + l,2]
            print('converting')
            if Testing:
                datatest = np.swapaxes(data,1,3)
                datatest = np.swapaxes(datatest,1,2)
                for j in range(5):
                    plt.imshow(datatest[j, :, :, :])
                    plt.pause(0.5)
                    plt.show()
                    plt.clf()
                plt.close()
            data = torch.FloatTensor(data)
            dataset = torch.utils.data.TensorDataset(data)
            datasetxyz.append(dataset)

    elif type=='grayscale':
        datasetxyz = []
        for img in data:
            img = plt.imread(img)
            if len(img.shape) > 2:
                img = img[:, :, 0]
            img = img/img.max()
            img = img[::sf, ::sf]
            x_max, y_max = img.shape[:]
            data = np.empty([32 * 900, 1, l, l])
            for i in range(32 * 900):
                x = np.random.randint(1, x_max - l - 1)
                y = np.random.randint(1, y_max - l - 1)
                subim = img[x:x + l, y:y + l]
                data[i, 0, :, :] = subim
            if Testing:
                for j in range(7):
                    plt.imshow(data[j, 0, :, :])
                    plt.pause(0.3)
                    plt.show()
                    plt.clf()
                plt.close()
            data = torch.FloatTensor(data)
            dataset = torch.utils.data.TensorDataset(data)
            datasetxyz.append(dataset)
            
    return datasetxyz



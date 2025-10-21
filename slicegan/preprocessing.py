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
    Testing = False
    if type in ['png', 'jpg', 'tif2D']:
        datasetxyz = []
        batch_size = 1000  # Reduced batch size
        for img_path in data:
            img = plt.imread(img_path) if type != 'tif2D' else tifffile.imread(img_path)
            if len(img.shape)>2:
                img = img[:,:,0]
            img = img[::sf,::sf]
            x_max, y_max = img.shape[:]
            phases = np.unique(img)
            total_samples = 32 * 900
            
            # Process in smaller batches
            for batch_start in range(0, total_samples, batch_size):
                batch_end = min(batch_start + batch_size, total_samples)
                current_batch_size = batch_end - batch_start
                batch_data = np.zeros([current_batch_size, len(phases), l, l], dtype=np.float32)
                
                for i in range(current_batch_size):
            if Testing and batch_start == 0:
                for j in range(min(7, current_batch_size)):
                    plt.imshow(batch_data[j, 0, :, :]+2*batch_data[j, 1, :, :])
                    plt.pause(0.3)
                    plt.show()
                    plt.clf()
                plt.close()
            batch_data = torch.FloatTensor(batch_data)
            dataset = torch.utils.data.TensorDataset(batch_data)
            datasetxyz.append(dataset)
            del batch_data  # Explicitly free memory
                    plt.show()
                    plt.clf()
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



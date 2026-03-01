import os
import numpy as np
import SimpleITK as sitk


base_path = "Processed_data_nii/"  # Path to the prostate dataset
save_path = "Preprocessed_data/"  # Path to save the preprocessed data
train_ratio = 0.6

sites = ['BIDMC', 'HK', 'I2CVB', 'BMC', 'RUNMC', 'UCL']

def convert_from_nii_to_png(img):
    high = np.quantile(img, 0.99)
    low = np.min(img)
    img = np.where(img > high, high, img)
    lungwin = np.array([low * 1., high * 1.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg = (newimg * 255).astype(np.uint8)
    return newimg

for site in sites:
    
    if not os.path.exists(f"{save_path}/{site}_data"):
        os.makedirs(f"{save_path}/{site}_data")
    images, labels = [], []
    sitedir = os.path.join(base_path, site)

    sample_list = sorted(os.listdir(sitedir))
    sample_list = [x for x in sample_list if 'segmentation.nii.gz' in x.lower()]

    for sample in sample_list:
        sampledir = os.path.join(sitedir, sample)
        if os.path.getsize(sampledir) < 1024 * 1024 and sampledir.endswith("nii.gz"):
            imgdir = os.path.join(sitedir, sample[:6] + ".nii.gz")
            label_v = sitk.ReadImage(sampledir)
            image_v = sitk.ReadImage(imgdir)
            label_v = sitk.GetArrayFromImage(label_v)
            label_v[label_v > 1] = 1
            image_v = sitk.GetArrayFromImage(image_v)
            image_v = convert_from_nii_to_png(image_v)
                
            for i in range(1, label_v.shape[0] - 1):
                label = np.array(label_v[i, :, :])
                if np.all(label == 0):
                    continue
                image = np.array(image_v[i - 1:i + 2, :, :])
                image = np.transpose(image, (1, 2, 0))

                labels.append(label)
                images.append(image)
        else:
            print(f"Skipping {sampledir} due to size")
    labels = np.array(labels).astype(int).astype(np.int_).squeeze()
    images = np.array(images)

    trainlen = int(max(train_ratio * len(labels), 32))
    vallen = int(0.2 * len(labels))
    testlen = int(0.2 * len(labels))
    
    save_dir = {
        "train": {'images': images[:trainlen], 'labels': labels[:trainlen]},
        "valid": {'images': images[trainlen:trainlen + vallen], 'labels': labels[trainlen:trainlen + vallen]},
        "test": {'images': images[-testlen:], 'labels': labels[-testlen:]}
    }
    np.save(f"{save_path}/{site}_data/train.npy", save_dir["train"], allow_pickle=True)
    np.save(f"{save_path}/{site}_data/valid.npy", save_dir["valid"], allow_pickle=True)
    np.save(f"{save_path}/{site}_data/test.npy", save_dir["test"], allow_pickle=True)
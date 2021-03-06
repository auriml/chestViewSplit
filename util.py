import torch
import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from skimage import transform, io, img_as_float, exposure
from torch.autograd import Variable

currentroot = os.getcwd()
os.chdir("../")
root = os.getcwd()
os.chdir(currentroot)

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)



def make_dataset(dir):
    images = []
    dir = os.path.expanduser(dir)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:

        img = Image.open(f)
        rgb = img.convert('RGB')
        return rgb

def SJ_loader(path):
    with open(path, 'rb') as f:
        img = img_as_float(io.imread(path))
        img = transform.resize(img, (578,560))
        img = exposure.equalize_hist(img)
        img = np.uint8(img * 255)
        rgb = np.dstack((img,img,img))
        rgb = Image.fromarray(rgb)
        return rgb

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class MyFolder(data.Dataset):
    def __init__(self, root=None, imgs=None,  transform=None, target_transform=None,
                 loader=default_loader):

        imgs = imgs if imgs is not None else make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path = self.imgs[index]
        
        img = None
        try:
            img = self.loader(path)
        except:
            print("failed %s..." %path)
            path = root + '/SJ' + '/image_dir_processed/20536686640136348236148679891455886468_k6ga29.png'
            img = self.loader(path)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            img = self.target_transform(img)
        

        return img, path

    def __len__(self):
        return len(self.imgs)
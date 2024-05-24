from torchvision import transforms
import numpy as np 
import cv2
np.random.seed(0)

class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


color_jitter = transforms.ColorJitter(0.4,0.4,0.4,0.1)
train_img_trans = transforms.Compose([transforms.RandomResizedCrop(size = 224,scale=(0.5, 1.0)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomApply([color_jitter], p = 0.7),
                                    transforms.RandomGrayscale(p  = 0.2),
                                    GaussianBlur(3),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
                                    ])
val_img_trans = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])                                         
])
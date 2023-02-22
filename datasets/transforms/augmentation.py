import numpy as np
import torch
import cv2
from scipy.signal import resample

try:
    from utils import openpose2coco
except Exception as e:
    pass

class ConvertToCoco(object):
    def __call__(self, sample):
        image = sample['image']
        image[:, openpose2coco[:, 0]] = image[:, openpose2coco[:, 1]]
        sample.update({'image': image})
        return sample


class DropOutFrames(object):
    def __init__(self, probability=0.1):
        self.probability = probability

    def __call__(self, sample):
        image = sample['image']

        mask = np.random.random(size = image.shape[0]) < self.probability
        mask_indices = np.argwhere(mask).ravel()
        image = np.take(image, mask_indices, axis = 0)

        sample.update({'image': image})
        return sample

class DropOutJoints(object):
    def __init__(self, prob=1, dropout_rate_range=0.1):
        self.dropout_rate_range = dropout_rate_range
        self.prob = prob

    def __call__(self, sample):
        if np.random.binomial(1, self.prob, 1) != 1:
            return sample

        data = sample['image']

        dropout_rate = np.random.uniform(0, self.dropout_rate_range, size = data.shape)
        zero_indices = 1 - np.random.binomial(1, dropout_rate, size = data.shape)
        data = data * zero_indices

        sample.update({'image': data})
        return sample

class RandomPace(object):
    def __init__(self, paces, period_length = 72):
        self.paces = paces
        self.unique_paces = np.unique(self.paces)
        self.period_length = period_length

    def __call__(self, sample):
        image = sample['image']
        h, w = image.shape[:2]

        pace_idx = np.random.choice(np.arange(len(self.paces)))
        pace = self.paces[pace_idx]

        if pace < 1:
            image = np.repeat(image, repeats = int(1 / pace), axis = 0)

            if h - self.period_length <= 0:
                # print(h, self.period_length)
                pass

            top = np.random.randint(0, h + 1 - self.period_length)
            image = image[top: top + self.period_length, :]

        if pace >= 1:
            pace = int(pace)
            if h - self.period_length <= 0:
                # print(h, self.period_length)
                pass
            top = np.random.randint(0, h + 1 - pace * self.period_length)
            image = image[top: top + pace * self.period_length, :]
            image = image[::pace]

        sample.update({'image': image})
        # sample['pace'] = np.array(pace_idx)
        sample['pace'] = np.argwhere(self.unique_paces == pace).ravel()

        return sample

class FlipSequence(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, sample):
        if np.random.random() <= self.probability:
            return sample

        image = sample['image']

        image = np.flip(image, axis=0).copy()
        sample.update({'image': image})
        return sample


class SqueezeAndFlip(object):
    def __init__(self, amount = None, flip_prob = 0.25):
        """
            amount should be either None to simply randomly flip the image
            or some value between 0 and 0.5
        """
        self.amount = 0 if amount is None else amount
        self.flip_prob = flip_prob

    def __call__(self, sample):
        image = sample['image']

        squeeze_amount = 1 - self.amount * np.random.random()

        chance = np.random.random()
        if chance < self.flip_prob:
            squeeze_amount *= -1

        image[:, 0] = squeeze_amount * image[:, 0]
        sample.update({'image': image})
        return sample

class PointNoise(object):
    def __init__(self, std=0.15):
        self.std = std

    def __call__(self, sample):
        image = sample['image']
        noise = np.random.uniform(-self.std, self.std, (image.shape[0], image.shape[1], 2)).astype(np.float32)
        image[:, :, :2] = image[:, :, :2] + noise
        # image[:, :, 2] = np.clip(image[:, :, 2], min=0, max=1)
        sample.update({'image': image})
        return sample

class JointNoise(object):
    def __init__(self, std=0.5):
        self.std = std

    def __call__(self, sample):
        data = sample['image']

        noise = torch.zeros((data.shape[1], 3))
        noise[:, :2].uniform_(-self.std, self.std)

        data =  data + noise.repeat((data.shape[0], 1, 1)).numpy()
        sample.update({'image': data})
        return sample

class Permutation(object):
    def __init__(self, do_apply = True, permutation_size = 12, prob = 0.2):
        self.permutation_size = permutation_size
        self.prob = prob
        self.apply = do_apply

    def __call__(self, sample):
        if not self.apply:
            return sample

        chance = np.random.random()
        if chance > self.prob:
            sample['permutation'] = np.array([0])
            return sample

        image = sample['image']
        h, w = image.shape[:2]

        top = np.random.randint(self.permutation_size + 1, h - self.permutation_size)
        bottom = np.random.randint(0, top - self.permutation_size)

        permutation1 = image[top: top + self.permutation_size].copy()
        permutation2 = image[bottom: bottom + self.permutation_size].copy()

        image[bottom: bottom + self.permutation_size] = permutation1
        image[top: top + self.permutation_size] = permutation2

        sample['permutation'] = np.array([1])
        sample.update({'image': image})

        return sample

class RandomPaces(object):
    def __init__(self, paces, minimum_length = 72):
        self.paces = paces
        self.minimum_length = minimum_length

    def transform_to_pace(self, sequence, pace):
        if pace == 1:
            return sequence
        elif pace < 1:
            for _ in range(int(0.5 // pace)):
                sequence = interpolate(sequence)

            return sequence
        else:
            return sequence[::pace]

    def __call__(self, image):
        h, w = image.shape[:2]

        number_of_paces = random.randint(1, 4)

        new_image = None
        slice_size = h // number_of_paces

        for count in range(number_of_paces):
            pace = random.choice(self.paces)
            slice = image[count * slice_size: (count + 1) * slice_size]

            new_slice = self.transform_to_pace(slice, pace)
            if new_image is None:
                new_image = new_slice
            else:
                new_image = np.concatenate((new_image, new_slice), axis=0)

        while new_image.shape[0] < self.minimum_length:
            new_image = interpolate(new_image)

        return new_image

class ConvertToTSSI(object):

    def __init__(self, do_apply = False):
        self.do_apply = do_apply

    def __call__(self, sample):
        if not self.do_apply:
            return sample

        image = sample['image']
        image[:, openpose2coco[:, 0]] = image[:, openpose2coco[:, 1]]

        mid_shoulder = (image[:, 5] + image[:, 6]) / 2
        mid_shoulder = np.expand_dims(mid_shoulder, axis=1)
        
        mid_hip = (image[:, 11] + image[:, 12]) / 2
        mid_hip = np.expand_dims(mid_hip, axis=1)
        
        root_joint = (mid_shoulder + mid_hip) / 2

        image = np.concatenate((image, mid_hip, root_joint), axis=1)
        # image = np.concatenate((image, mid_shoulder, mid_hip, root_joint), axis=1)
    
        graph_dfs_traversal = np.array([
            19, 18, 11, 13, 15,
            13, 11, 18, 12, 14,
            16, 14, 12, 18, 19,
            17, 5, 7, 9, 7, 5,
            17, 6, 8, 10, 8, 6,
            17, 0, 1, 3, 1, 0,
            2, 4, 2, 0, 17, 19
        ])

        image = image[:, graph_dfs_traversal, :]
        sample.update({'image': image})

        return sample

class CustomResize(object):
    def __init__(self, new_size):
        self.new_size = (new_size, new_size)
    
    def __call__(self, sample):
        image = sample['image']

        image = cv2.resize(image, self.new_size)

        sample.update({'image': image})

        return sample

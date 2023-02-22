import math
import torch
import numpy as np


class MultiInput:
    def __init__(self, connect_joint, center=0, enabled=False, concat=False):
        self.connect_joint = connect_joint
        self.center = center
        self.enabled = enabled
        self.concat = concat

    def __call__(self, sample):
        image = sample['image']

        if not self.enabled:
            image = np.expand_dims(image, axis = 2)
            sample.update({'image': image})
            return sample

        # T, V, C -> T, V, I=3, C + 2
        T, V, C = image.shape
        x_new = np.zeros((T, V, 3, C + 2))
        # print(x_new.shape, image.shape)

        # Joints
        x = image
        x_new[:, :, 0, :C] = x

        for i in range(V):
            x_new[:, i, 0, C:] = x[:, i, :2] - x[:, self.center, :2]

        # Velocity
        for i in range(T - 2):
            x_new[i, :, 1, :2] = x[i + 1, :, :2] - x[i, :, :2]
            x_new[i, :, 1, 3:] = x[i + 2, :, :2] - x[i, :, :2]
        x_new[:, :, 1, 3] = x[:, :, 2]

        # Bones
        for i in range(V):
            x_new[:, i, 2, :2] = x[:, i, :2] - x[:, self.connect_joint[i], :2]
        bone_length = 0

        for i in range(C - 1):
            bone_length += np.power(x_new[:, :, 2, i], 2)

        bone_length = np.sqrt(bone_length) + 0.0001

        for i in range(C - 1):
            x_new[:, :, 2, C+i] = np.arccos(x_new[:, :, 2, i] / bone_length)
        x_new[:, :, 2, 3] = x[:, :, 2]

        if self.concat:
            x_new = np.concatenate([x_new[:, :, i] for i in range(3)], axis = 2)

        image = x_new
        sample.update({'image': image})

        return sample

class NormalizeEmpty:
    def __call__(self, sample):
        image = sample['image']

        # Fix empty detections
        frames, joints = np.where(image[:, :, 0] == 0)
        for frame, joint in zip(frames, joints):
            center_of_gravity = np.mean(image[frame], axis = 1)
            image[frame, joint, 0] = center_of_gravity[0]
            image[frame, joint, 1] = center_of_gravity[1]
            image[frame, joint, 2] = 0

        sample.update({'image': image})
        return sample


class RandomFlipLeftRight:
    def __init__(self, p=0.5, flip_idx=None):
        self.p = p
        self.flip_idx = flip_idx

    def __call__(self, sample):
        if np.random.random() > self.p:
            return sample

        image = sample['image']
        image = image[:, self.flip_idx]
        sample.update({'image': image})
        return sample


class RandomFlipSequence:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        image = sample['image']
        if np.random.random() > self.p:
            return sample

        image = image.flip(0)
        sample.update({'image': image})

        return sample


class PadSequence:
    def __init__(self, sequence_length=25):
        self.sequence_length = sequence_length

    def __call__(self, sample):
        image = sample['image']

        input_length = image.shape[0]
        if input_length > self.sequence_length:
            return sample

        diff = self.sequence_length + 1 - input_length
        len_pre = int(math.ceil(diff / 2))
        len_pos = int(diff / 2) + 1

        while len_pre > image.shape[0] or len_pos > image.shape[0]:
            image = np.concatenate([np.flip(image, 0), image, np.flip(image, 0)], axis = 0)

        pre = np.flip(image[1:len_pre], 0)
        pos = np.flip(image[-1 - len_pos:-1], 0)
        image = np.concatenate([pre, image, pos], axis = 0)[:self.sequence_length]

        sample.update({'image': image})

        return sample

class MirrorPoses(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, data):
        if np.random.random() <= self.probability:
            center = np.mean(data[:, :, 0], axis=1, keepdims=True)
            data[:, :, 0] = center - data[:, :, 0] + center

        return data

class RandomCropSequence:
    def __init__(self, min_sequence_length=20, p=0.25):
        self.min_sequence_length = min_sequence_length
        self.p = p

    def __call__(self, sample):
        image = sample['image']

        length = image.shape[0]
        if length <= self.min_sequence_length or np.random.random() > self.p:
            return sample

        sequence_length = int(np.random.randint(self.min_sequence_length, length))
        start = np.random.randint(0, length - sequence_length)
        end = start + sequence_length
        image = image[start:end]

        sample.update({'image': image})
        return sample


class RandomSelectSequence:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length

    def __call__(self, sample):
        image = sample['image']

        length = image.shape[0]
        if length <= self.sequence_length:
            return sample

        start = np.random.randint(0, length - self.sequence_length)
        end = start + self.sequence_length
        image = image[start:end]

        sample.update({'image': image})
        return sample


class SelectSequenceCenter:
    def __init__(self, sequence_length=10):
        self.sequence_length = sequence_length

    def __call__(self, sample):
        image = sample['image']

        start = int((image.shape[0] / 2) - (self.sequence_length / 2))
        end = start + self.sequence_length
        image = image[start:end]

        sample.update({'image': image})
        return sample

class RandomMove:
    def __init__(self, noise=(3, 1)):
        self.noise = noise

    def __call__(self, sample):
        image = sample['image']

        noise = torch.zeros(3)
        noise[0].uniform_(-self.noise[0], self.noise[0])
        noise[1].uniform_(-self.noise[1], self.noise[1])

        image = image + noise.repeat((image.shape[0], image.shape[1], 1)).numpy()

        sample.update({'image': image})
        return sample

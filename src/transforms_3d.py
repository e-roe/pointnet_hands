import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x


class SelectPoints:
    def __init__(self, selection):
        if isinstance(selection, str):
            selection_parts = {
                'lhand': range(0, 21),
                'pose': range(21, 54),
                'rhand': range(54, 75),
            }
            selection = selection_parts[selection]

        self.selection = selection

    def __call__(self, x):
        return x[:, self.selection, :]


class SelectCoords:
    def __init__(self, selection):
        coord_names = {
            'x': 0,
            'y': 1,
            'z': 2,
            'c': 3,
        }
        self.selection = [coord_names[coord] for coord in selection if isinstance(coord, str)]

    def __call__(self, x):
        return x[:, :, self.selection]


class GaussianNoise:
    def __init__(self, amount=0.001):
        self.amount = amount

    def __call__(self, x):
        noise = self.amount * np.random.normal(size=x.shape)
        return x + noise


class ToTensor:
    def __call__(self, x):
        return torch.tensor(x, dtype=torch.float)


class TimeScale:
    def __init__(self, num_frames):
        self.num_frames = num_frames

    def __call__(self, x):
        num_frames, num_points, num_coords = x.shape
        new_data = np.zeros((self.num_frames, num_points, num_coords))
        f = np.arange(0, num_frames)
        f_vals = np.linspace(0, num_frames, self.num_frames)

        for p in range(num_points):
            for c in range(num_coords):
                new_data[:, p, c] = np.interp(f_vals, f, x[:, p, c])

        return new_data


class Rotate:
    def __init__(self, axis='y', angle = 0.0, prob=1.0):
        if isinstance(angle, list) or isinstance(angle, tuple):
            angle = np.random.uniform(angle[0], angle[1])
        self.angle_in_degree = angle
        self.axis = axis
        self.rotation_matrix = R.from_euler(self.axis, angle, degrees=True).as_matrix()
        self.prob = prob

    def __call__(self, x):
        if np.random.rand() < self.prob:
            rot_points = []
            for i in range(2):
                x[:,  i] = x[:,  i] - 0.5
            for point in x:
                coord = np.dot(self.rotation_matrix, point)
                rot_points.append(coord)
            rot_points = np.array(rot_points)
            for i in range(2):
                rot_points[:, i] = rot_points[:, i] + 0.5

            return rot_points

        return x


class Flip:
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, x):
        if np.random.rand() < self.prob:
            x[:, 0] = 1.0 - x[:, 0]

        return x


class Scale:
    def __init__(self, prob=1.0, axis='x', factor=1.0):
        self.prob = prob
        self.axis = axis
        self.axes = {'x':0, 'y':1, 'z':2}
        self.scale_matrix = np.array([[1, 0.0, 0.0],
                                      [0.0, 1, 0.0],
                                      [0.0, 0.0, 1]])
        self.scale_matrix[self.axes[axis], self.axes[axis]] = factor
        self.delta = (1 - factor) / 2

    def __call__(self, x):
        if np.random.rand() <= self.prob:
            scale_points = []
            for point in x:
                point[self.axes[self.axis]] = point[self.axes[self.axis]] - 0.5
                scaled = np.dot(self.scale_matrix, point)
                scaled[self.axes[self.axis]] = scaled[self.axes[self.axis]] + 0.5
                scale_points.append(scaled)
            return np.array(scale_points)

        return x


class Zoom:
    def __init__(self, prob=1.0, factor=1.0):
        self.prob = prob
        self.axes = {'x':0, 'y':1, 'z':2}
        self.scale_matrix = np.array([[factor, 0.0, 0.0],
                                      [0.0, factor, 0.0],
                                      [0.0, 0.0, 1]])

    def __call__(self, x):
        if np.random.rand() <= self.prob:
            for i in range(2):
                x[:, :, i] = x[:, :, i] - 0.5

            zoom_points = []
            for frame in x:
                frame_points = []
                for point in frame:
                    zoomed = np.dot(self.scale_matrix, point)
                    frame_points.append(zoomed)
                zoom_points.append(frame_points)
            zoom_points = np.array(zoom_points)
            for i in range(2):
                zoom_points[:, :, i] = zoom_points[:, :, i] + 0.5

            return zoom_points

        return x
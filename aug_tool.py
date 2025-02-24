import numpy as np

from batchgenerators.augmentations.utils import rotate_coords_3d, rotate_coords_2d, scale_coords, \
    elastic_deform_coordinates, create_zero_centered_coordinate_mesh, interpolate_img


class MirrorTransform(object):
    def augment_mirroring(self, data, code=(1, 1)):
        if code[0] == 1:
            data[:] = data[::-1]
        if code[1] == 1:
            data[:, :] = data[:, ::-1]
        # if code[2] == 1:
        #     data[:, :, :] = data[:, :, ::-1]
        return data

    def rand_code(self):
        code = []
        for i in range(3):
            if np.random.uniform() < 0.5:
                code.append(1)
            else:
                code.append(0)
        return code


class SpatialTransform(object):
    def __init__(self, patch_center_dist_from_border=30,
                 do_elastic_deform=True, alpha=(0., 1000.), sigma=(10., 13.),
                 do_rotation=True, angle_x=(0, 2 * np.pi), angle_y=(0, 2 * np.pi), angle_z=(0, 2 * np.pi),
                 do_scale=True, scale=(0.75, 1.25), border_mode_data='constant', border_cval_data=0, order_data=3,
                 border_mode_seg='constant', border_cval_seg=0, order_seg=1, random_crop=True, data_key="data",
                 label_key="seg", p_el_per_sample=1, p_scale_per_sample=1, p_rot_per_sample=1,
                 independent_scale_for_each_axis=False, p_rot_per_axis:float=1):
        self.independent_scale_for_each_axis = independent_scale_for_each_axis
        self.p_rot_per_sample = p_rot_per_sample
        self.p_scale_per_sample = p_scale_per_sample
        self.p_el_per_sample = p_el_per_sample
        self.data_key = data_key
        self.label_key = label_key
        self.patch_center_dist_from_border = patch_center_dist_from_border
        self.do_elastic_deform = do_elastic_deform
        self.alpha = alpha
        self.sigma = sigma
        self.do_rotation = do_rotation
        self.angle_x = angle_x
        self.angle_y = angle_y
        self.angle_z = angle_z
        self.do_scale = do_scale
        self.scale = scale
        self.border_mode_data = border_mode_data
        self.border_cval_data = border_cval_data
        self.order_data = order_data
        self.border_mode_seg = border_mode_seg
        self.border_cval_seg = border_cval_seg
        self.order_seg = order_seg
        self.p_rot_per_axis = p_rot_per_axis

    def augment_spatial(self, data, coords, is_label=False):
        if is_label:
            order = self.order_seg
            border_mode = self.border_mode_seg
            border_cval = self.border_cval_seg
        else:
            order= self.order_data
            border_mode = self.border_mode_data
            border_cval = self.border_cval_data
        data = interpolate_img(data, coords, order, border_mode, cval=border_cval)
        return data


    def rand_coords(self, patch_size):
        dim = len(patch_size)

        coords = create_zero_centered_coordinate_mesh(patch_size)

        if self.do_elastic_deform and np.random.uniform() < self.p_el_per_sample:
            a = np.random.uniform(self.alpha[0], self.alpha[1])
            s = np.random.uniform(self.sigma[0], self.sigma[1])
            coords = elastic_deform_coordinates(coords, a, s)

        if self.do_rotation and np.random.uniform() < self.p_rot_per_sample:

            if np.random.uniform() <= self.p_rot_per_axis:
                a_x = np.random.uniform(self.angle_x[0], self.angle_x[1])
            else:
                a_x = 0

            if dim == 3:
                if np.random.uniform() <= self.p_rot_per_axis:
                    a_y = np.random.uniform(self.angle_y[0], self.angle_y[1])
                else:
                    a_y = 0

                if np.random.uniform() <= self.p_rot_per_axis:
                    a_z = np.random.uniform(self.angle_z[0], self.angle_z[1])
                else:
                    a_z = 0

                coords = rotate_coords_3d(coords, a_x, a_y, a_z)
            else:
                coords = rotate_coords_2d(coords, a_x)

        if self.do_scale and np.random.uniform() < self.p_scale_per_sample:
            if not self.independent_scale_for_each_axis:
                if np.random.random() < 0.5 and self.scale[0] < 1:
                    sc = np.random.uniform(self.scale[0], 1)
                else:
                    sc = np.random.uniform(max(self.scale[0], 1), self.scale[1])
            else:
                sc = []
                for _ in range(dim):
                    if np.random.random() < 0.5 and self.scale[0] < 1:
                        sc.append(np.random.uniform(self.scale[0], 1))
                    else:
                        sc.append(np.random.uniform(max(self.scale[0], 1), self.scale[1]))
            coords = scale_coords(coords, sc)
        ctr = np.asarray([patch_size[0]//2, patch_size[1]//2])
        aa = ctr[:, np.newaxis, np.newaxis, np.newaxis]
        coords += ctr[:, np.newaxis, np.newaxis]

        return coords

class Crop(object):
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def augment_crop(self, data, code=(1, 1, 1)):
        data = data[code[0]:code[0] + self.patch_size[0], code[1]:code[1] + self.patch_size[1], code[2]:code[2] + self.patch_size[2]]

        return data

    def rand_code(self, shape):
        n_x = np.random.randint(0, shape[2] - self.patch_size[2], 1)[0]
        n_y = np.random.randint(0, shape[1] - self.patch_size[1], 1)[0]
        n_z = np.random.randint(0, shape[0] - self.patch_size[0], 1)[0]
        code = [n_z, n_y, n_x]
        return code

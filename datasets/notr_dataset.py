import os
import skimage.transform
import numpy as np
import PIL.Image as pil
import pickle, json, cv2

from .base_dataset import BaseDataset

class NOTRDataset(BaseDataset):
    """Superclass for different types of Waymo dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(NOTRDataset, self).__init__(*args, **kwargs)
        self.num_cams = 0 # 只使用第一个相机
        
        self.K = dict()
        self.get_all_intrinsic()

        self.full_res_shape = (1920, 1280)

        self.categories = {
            0: 'undefined',             # Anything that does not fit the other classes or is too ambiguous tolabel.
            1: 'ego_vehicle',           # The Waymo vehicle.
            2 : 'car',                  # Small vehicle such as a sedan, SUV, pickup truck, minivan or golf cart.
            3 : 'truck',                # Large vehicle that carries cargo.
            4 : 'bus',                  # Large vehicle that carries more than 8 passengers.
            5 : 'other_vehicle',        # Large vehicle that is not a truck or a bus.
            6 : 'bicycle',              # Bicycle with no rider.
            7 : 'motorcycle',           # Motorcycle with no rider.
            8 : 'trailer',              # Trailer attached to another vehicle or horse.
            9 : 'pedestrian',           # Pedestrian. Does not include objects associated with the pedestrian, such as suitcases, strollers or cars.
            10 : 'bicyclist',           # Bicycle with rider.
            11 : 'motorcyclist',        # Motorcycle with rider.
            12 : 'bird',                # Birds, including ones on the ground.
            13 : 'ground_animal',       # Animal on the ground such as a dog, cat, cow, etc.
            14 : 'const_cone_pole',     # Cone or short pole related to construction.
            15 : 'pole',                # Permanent horizontal and vertical lamp pole, traffic sign pole, etc.
            16 : 'pedestrian_stuff',    # Large object carried/pushed/dragged by a pedestrian.
            17 : 'sign',                # Sign related to traffic, including front and back facing signs.
            18 : 'traffix_light',       # The box that contains traffic lights regardless of front or back facing.
            19 : 'building',            # Permanent building and walls, including solid fences.
            20 : 'road',                # Drivable road with proper markings, including parking lots and gas stations.
            21 : 'lane_marker',         # Marking on the road that is parallel to the ego vehicle and defineslanes.
            22 : 'road_marker',         # All markings on the road other than lane markers.
            23 : 'sidewalk',            # Paved walkable surface for pedestrians, including curbs.
            24 : 'vegetation',          # Vegetation including tree trunks, tree branches, bushes, tall grasses,flowers and so on.
            25 : 'sky',                 # The sky, including clouds.
            26 : 'ground',              # Other horizontal surfaces that are drivable or walkable.
            27 : 'dynamic',             # Object that is not permanent in its current position and does not belongto any of the above classes.
            28 : 'static',              # Object that is permanent in its current position and does not belong toany of the above classes.
        }
        
    def get_timestep(self, folder, frame_index, offset):
        return 1    # consistent timesteps in this dataset

    def get_all_intrinsic(self):
        for file in self.filenames:
            folder = file.split()[0]
            if folder not in self.K:
                self.K[folder] = np.eye(4, dtype=np.float32)
                intrinsic = np.loadtxt(
                    os.path.join(self.data_path, folder, 'intrinsics', '0.txt')
                )
                fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
                self.K[folder][:3,:3] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    def get_intrinsic(self, folder):
        self.K[folder][0] = self.K[folder][0] / self.full_res_shape[0]
        self.K[folder][1] = self.K[folder][1] / self.full_res_shape[1]
        return self.K[folder]
    
    def get_gt_dim(self, folder, frame_index, side):
        return self.full_res_shape[1], self.full_res_shape[0]
    
    def get_img_path(self, folder, frame_index, side):
        f_str = "{:03d}_0{}".format(frame_index, self.img_ext)
        # return os.path.join(self.data_path, folder, 'images', self.img_type, f_str)
        return os.path.join(self.data_path, folder, 'images', f_str)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_img_path(folder, frame_index, side))
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color
    
    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:06d}{}".format(frame_index, '.npy')
        depth_path = os.path.join(self.data_path, folder, 'depth', f_str)

        depth = np.load(depth_path)

        if do_flip:
            depth[:,0] = self.full_res_shape[0] - depth[:,0]
        
        depth = np.concatenate((depth[:,1:2], depth[:,0:1], depth[:,2:3]), axis=1)    # (N, 3) -> [h_i, w_i, z_i]

        return depth

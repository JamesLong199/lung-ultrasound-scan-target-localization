# inspired by teddylew12/race_on_cv, made my own customization
# https://github.com/teddylew12/race_on_cv/blob/master/tag.py

# class that contains relevant information for all tags of interest
# assume uniform tag size for all tags used
import math
import numpy as np

class TagList():

    def __init__(self,tag_size):
        # inputs
        # -- tag_size: tag size in meter, must not be None
        assert tag_size > 0
        self.tag_size = tag_size
        self.tag_dict = {}

    def add_tag(self,family,id,pos,angles,radian=False):
        # inputs
        # -- family: tag family
        # -- id: tag id of its family
        # -- pos: tuple/list (x,y,z) coordinate of tag center
        # -- angles: tuple (theta_x,theta_y,theta_z) euler angle of tag
        # -- radian: if the angles provided are in radians
        assert isinstance(family,str)
        assert isinstance(id,int) and id>=0
        assert len(pos) == 3 and len(angles) == 3

        if (family,id) in self.tag_dict.keys():
            print('Same (family,id) already exists!\nPrevious tag info will be overwritten!')

        pos = (np.array(pos)).reshape(3,)
        angles = (np.array(angles)).reshape(3,)
        # convert degree to radian
        if radian == False:
            angles = angles/180 * math.pi

        self.tag_dict[(family,id)] = (pos,angles)

    def get_tag_pose(self,family,id):
        # return corresponding tag pose from tag family and id

        if (family,id) in self.tag_dict.keys():
            return self.tag_dict[(family,id)]
        else:
            print('Tag does not exist!')

    def get_tag_families(self):
        # return all tag families in a string separated by space
        ret_str = ''
        tag_fam_list = []
        for tag_fam,_ in self.tag_dict.keys():
            if tag_fam not in tag_fam_list:
                tag_fam_list.append(tag_fam)
                ret_str  = ret_str + tag_fam

        if len(ret_str) == 0:
            print('Found no tags!')

        return ret_str

    def get_tag_size(self):
        # return tag size
        return self.tag_size

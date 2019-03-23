import os
import sys
import numpy as np

sys.path.append('./motion/')
import BVH as BVH
import Animation as Animation
from InverseKinematics import JacobianInverseKinematics
from Quaternions import Quaternions

rest, rest_names, _ = BVH.load('./model/nao_heirarchy6.bvh')
rest_targets = Animation.positions_global(rest)
rest_height = rest_targets[0,:,1].max() - rest_targets[0,:,1].min() 

rest_luarm_qt = Quaternions(np.array([ 0.70710678,  0.        ,  0.70710678,  0.        ])) 
rest_ruarm_qt = Quaternions(np.array([ 0.70710678,  0.        ,  -0.70710678,  0.        ])) 

filename = "./data/wave.bvh"
mocap, mocap_names, mocap_ftime = BVH.load(filename)
mocap_targets = Animation.positions_global(mocap)
mocap_height = mocap_targets[0,:,1].max() - mocap_targets[0,:,1].min() 

targets = (rest_height / mocap_height) * mocap_targets

anim = rest.copy()
anim.positions = anim.positions.repeat(len(targets), axis=0)
anim.rotations.qs = anim.rotations.qs.repeat(len(targets), axis=0)
anim.positions[:,0] = targets[:,0]

rest_map = {}
for i, name in enumerate(rest_names):
    rest_map[name] = i
mocap_map = {}
for i, name in enumerate(mocap_names):
    mocap_map[name] = i

joint_map = {
    "Hips" : "Torso", 
    "LeftArm" :"LUpperarm",
    "LeftForeArm" :"LLowerarm",
    "LeftHand" :"LHand",

    "RightArm" :"RUpperarm",
    "RightForeArm" :"RLowerarm",
    "RightHand" :"RHand",

    "Neck1" :"Neck",
    "Head" :"Head",

    "LHipJoint" :"LHip", 
    "LeftUpLeg" :"LThigh",
    "LeftLeg" :"LLeg",
    "LeftFoot" :"LFoot",
    "LeftToeBase" :"LToes",

    "RHipJoint" :"RHip", 
    "RightUpLeg" :"RThigh",
    "RightLeg" :"RLeg",
    "RightFoot" :"RFoot",
    "RightToeBase" :"RToes",
}

# joint_map = {
#     "hip" : "Torso", 
#     "lhumerus" :"LUpperarm",
#     "lradius" :"LLowerarm",
#     "lwrist" :"LHand",

#     "rhumerus" :"RUpperarm",
#     "rradius" :"RLowerarm",
#     "rwrist" :"RHand",

#     "thorax" :"Neck",
#     "head" :"Head",

#     "lhipjoint" :"LHip", 
#     "lfemur" :"LThigh",
#     "ltibia" :"LLeg",
#     "lfoot" :"LFoot",
#     "ltoes" :"LToes",

#     "rhipjoint" :"RHip", 
#     "rfemur" :"RThigh",
#     "rtibia" :"RLeg",
#     "rfoot" :"RFoot",
#     "rtoes" :"RToes",
# }

targetmap = {} 
for mocap_joint, rest_joint in joint_map.items():
    if(rest_joint in ["LUpperarm", "RUpperarm"]):
        anim.rotations[:, rest_map[rest_joint]] = mocap.rotations[:,mocap_map[mocap_joint]]
    else:
        anim.rotations[:, rest_map[rest_joint]] = mocap.rotations[:,mocap_map[mocap_joint]]
    targetmap[rest_map[rest_joint]] = targets[:,mocap_map[mocap_joint]]

anim.rotations[:, rest_map["LUpperarm"]] += rest_luarm_qt 
anim.rotations[:, rest_map["RUpperarm"]] += rest_ruarm_qt 

ik = JacobianInverseKinematics(anim, targetmap, iterations=5000, damping=7, silent=False)
ik()
BVH.save('./processed/wave.bvh', anim, rest_names, 1.0/25)
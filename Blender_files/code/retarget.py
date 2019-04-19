import sys
sys.path.append('./motion/')
import os
import numpy as np
import BVH as BVH
import Animation as Animation
from InverseKinematics import JacobianInverseKinematics
from Quaternions import Quaternions

# HYPER PARAMS (CONSTANTS) for Processing
modelpath = "./model/nao_heirarchy2.bvh"

# clippath = "./data/test_wip.bvh"
clippath = "./data/required/required_wave.bvh"
# clippath = "./data/required/required_wip.bvh"

# retargetpath = './processed/actual_retarget/retarget_test_wip.bvh'
retargetpath = './processed/actual_retarget/retarget_req_wave.bvh'
# retargetpath = './processed/actual_retarget/retarget_req_wip.bvh'

constraintpath = "./model/constraints_2.txt"

# c_retargetpath = "./processed/constrained_retarget/test_wip.bvh"
c_retargetpath = "./processed/constrained_retarget/wave.bvh"
# c_retargetpath = "./processed/constrained_retarget/wip.bvh"

############################################################################################

rest, rest_names, _ = BVH.load(modelpath)
rest_targets = Animation.positions_global(rest)
rest_height = rest_targets[0,:,1].max() - rest_targets[0,:,1].min() 

mocap, mocap_names, mocap_ftime = BVH.load(clippath)
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

# joint_map = {
#     "Hips" : "Torso", 
#     "LeftArm" :"LUpperarm",
#     "LeftForeArm" :"LLowerarm",
#     "LeftHand" :"LHand",

#     "RightArm" :"RUpperarm",
#     "RightForeArm" :"RLowerarm",
#     "RightHand" :"RHand",

#     "Neck1" :"Neck",
#     "Head" :"Head",

#     "LHipJoint" :"LHip", 
#     "LeftUpLeg" :"LThigh",
#     "LeftLeg" :"LLeg",
#     "LeftFoot" :"LFoot",
#     "LeftToeBase" :"LToes",

#     "RHipJoint" :"RHip", 
#     "RightUpLeg" :"RThigh",
#     "RightLeg" :"RLeg",
#     "RightFoot" :"RFoot",
#     "RightToeBase" :"RToes",
# }

joint_map = {
    "hip" : "Torso", 
    "lhumerus" :"LUpperarm",
    "lradius" :"LLowerarm",
    "lwrist" :"LHand",

    "rhumerus" :"RUpperarm",
    "rradius" :"RLowerarm",
    "rwrist" :"RHand",

    "thorax" :"Neck",
    "head" :"Head",

    "lhipjoint" :"LHip", 
    "lfemur" :"LThigh",
    "ltibia" :"LLeg",
    "lfoot" :"LFoot",
    "ltoes" :"LToes",

    "rhipjoint" :"RHip", 
    "rfemur" :"RThigh",
    "rtibia" :"RLeg",
    "rfoot" :"RFoot",
    "rtoes" :"RToes",
}

targetmap = {} 
for mocap_joint, rest_joint in joint_map.items():
    # if rest_joint not in ["RHand", "LHand"]:
    anim.rotations[:, rest_map[rest_joint]] += mocap.rotations[:,mocap_map[mocap_joint]]
    targetmap[rest_map[rest_joint]] = targets[:,mocap_map[mocap_joint]]

# ik = JacobianInverseKinematics(anim, targetmap, iterations=5000, damping=7, silent=False)
# ik()

# BVH.save(retargetpath, anim, rest_names, 1.0/25, order='xyz')

############################################################################################

with open(constraintpath, 'r') as f:
    content = f.readlines()

content = [x.strip() for x in content]
joints = [list(filter(None, c.split('\t'))) for c in content]
from bvh import Bvh
with open(retargetpath) as f:
    new_mocap = Bvh(f.read())

channel = ["Xrotation", "Yrotation", "Zrotation"]

with open(c_retargetpath, 'w') as f:
    f.write("Frames: " + str(new_mocap.nframes) + "\n")    
    f.write("Frame Time: " + str(new_mocap.frame_time) + "\n")    
    
    for fr in range(new_mocap.nframes):
        out_list = [0,0,0]
        for j in joints:
            for x, ang in enumerate(j[1:]):
                if ang != "0":
                    out_list.append(new_mocap.frame_joint_channel(fr, j[0], channel[x]))
                else:
                    out_list.append(0)
        frame = [str(a) for a in out_list]
        f.write(" ".join(frame) + "\n")


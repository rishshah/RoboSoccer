import os
import sys
import numpy as np

sys.path.append('./motion/')
import BVH as BVH
import Animation as Animation
from InverseKinematics import JacobianInverseKinematics

rest, rest_names, _ = BVH.load('./model/nao_heirarchy5.bvh')
# BVH.save('./data/base.bvh', rest, rest_names)

# with open('tp.txt') as f:
#   content = f.readlines()

# content = [x.strip() for x in content] 
# content = [x.split() for x in content] 
# channel_mapping = ['Xrotation', 'Yrotation', 'Zrotation']

# mapping = {}
# for joint in content:
#   mapping[joint[0]] = []
#   for i in range(1,4):
#       if (joint[i] != '0'):
#           mapping[joint[0]].append((channel_mapping[i-1], joint[i]))


filename = "./data/wave.bvh"
hdm05anim, names, ftime = BVH.load(filename)

targets = Animation.positions_global(hdm05anim)
 
anim = rest.copy()
anim.positions = anim.positions.repeat(len(targets), axis=0)
anim.rotations.qs = anim.rotations.qs.repeat(len(targets), axis=0)

targetmap = {}
# constraintsmap = {
#     # 'lfemur'    : 'LHip3', 
#     'ltibia'    : ('LLeg', -1),
#     'lfoot'     : ('LFoot', 1),
    
#     # 'rfemur'    : 'RHip3',
#     'rtibia'    : ('RLeg', -1),
#     'rfoot'     : ('RFoot', 1),
    
#     # 'lhumerus'  : 'LUpperarm',
#     # 'lradius'   : 'LLowerarm',
    
#     # 'rhumerus'  : 'RUpperarm',
#     # 'rradius'   : 'RLowerarm',

# }

constraintsmap = {
    # 'LeftUpLeg'    : ('LHip3', 1.0), 
    # 'LeftLeg'      : ('LLeg', -1.0),
    # 'LeftFoot'     : ('LFoot', 1.0),
    
    # 'RightUpLeg'    : ('RHip3', 1.0),
    # 'RightLeg'      : ('RLeg', -1.0),
    # 'RightFoot'     : ('RFoot', 1.0),
    
    'LeftArm'     : ('LUpperarm', 1.0),
    'LeftForeArm' : ('LLowerarm', 1.0),
    
    'RightArm'    : ('RUpperarm', 1.0),
    'RightForeArm': ('RLowerarm', 1.0),

}


for name in constraintsmap.keys():
    for i in range(0,len(rest_names)):
        for j in range(0,len(names)):
            if rest_names[i] == constraintsmap[name][0] and name == names[j]:
                anim.rotations[:,i] = hdm05anim.rotations[:,j] * constraintsmap[name][1] 


for ti in range(targets.shape[1]):
    for j in range(0, len(rest_names)):
        if names[ti] in constraintsmap.keys() and constraintsmap[names[ti]] == rest_names[j]:
            targetmap[j] = targets[:,ti]

# ik = JacobianInverseKinematics(anim, targetmap, iterations=10, damping=2.0, silent=True)
# ik()

BVH.save('./processed/wave1.bvh', anim, names=rest_names)

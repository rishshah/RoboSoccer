from bvh import Bvh
import json, sys, math
import numpy as np

sys.path.append('../motion/')
import BVH as BVH
import Animation as Animation

# with open('../data/walk_in_place.bvh') as f:
with open('../data/wave.bvh') as f:
  performer = Bvh(f.read())

with open('../model/nao_heirarchy5.bvh') as f:
  end_user = Bvh(f.read())

# with open('../model/mapping.json') as f:
with open('../model/mapping1.json') as f:
  data = json.load(f)

# hdm05anim, names, ftime = BVH.load("../data/walk_in_place.bvh")
hdm05anim, names, ftime = BVH.load("../data/wave.bvh")
targets = Animation.positions_global(hdm05anim)


a = targets[:,3] - targets[:,1]
b = targets[:,3] - targets[:,2]

lhip2 = np.rad2deg(np.arctan(b[:,1]/b[:,0]))
for i in range(0,performer.nframes):
  if lhip2[i] < 0:
    lhip2[i] = -90 - lhip2[i]
  else:
    lhip2[i] = 90 - lhip2[i]
lhip3 = np.rad2deg(np.arctan(a[:,2]/a[:,1])) 


c = targets[:,8] - targets[:,6]
d = targets[:,8] - targets[:,7]

rhip2 = np.rad2deg(np.arctan(c[:,1]/c[:,0]))
for i in range(0,performer.nframes):
  if rhip2[i] < 0:
    rhip2[i] = -90 - rhip2[i]
  else:
    rhip2[i] = 90 - rhip2[i]
rhip3 = np.rad2deg(np.arctan(d[:,2]/d[:,1])) 


# e = targets[:,8] - targets[:,6]
# f = targets[:,8] - targets[:,7]


end_user_joint_names = end_user.get_joints_names()
transformed_frames = np.zeros((performer.nframes, len(end_user_joint_names),3))
channels = ['Zrotation', 'Yrotation', 'Xrotation']


# def hip_value(frame, type):
#   if type == 0:
#     return 0
#   elif type == 'lhipjoint': #hip2
#     return -lhip2[frame]
#     return 0
#   elif type == 'lfemur': #hip3
#     return lhip3[frame]
#   elif type == 'rhipjoint': #hip2
#     return -rhip2[frame]
#     return 0
#   elif type == 'rfemur': #hip3
#     return rhip3[frame]
#   else:
#     print("PANIC")
#     return None
def hip_value(frame, type):
  if type == 0:
    return 0
  elif type == 'LHipJoint': #hip2
    return -lhip2[frame]
    return 0
  elif type == 'LeftUpLeg': #hip3
    return lhip3[frame]
  elif type == 'RHipJoint': #hip2
    return -rhip2[frame]
    return 0
  elif type == 'RightUpLeg': #hip3
    return rhip3[frame]

  # elif type == 'RightArm': #lae1
  #   return rhip3[frame]
  # elif type == 'RightArm': #rae1
  #   return rhip3[frame]
  else:
    print("PANIC")
    return None


for frame in range(500):
  for i, joint in enumerate(end_user_joint_names):
    for k, channel in enumerate(channels): 
      if channel not in data[joint]:
        if "const" in data[joint]:
          transformed_frames[frame, i, k] = data[joint]["const"][channel] 
        elif "custom" in data[joint]:
          transformed_frames[frame, i, k] = hip_value(frame, data[joint]["custom"][channel])
        else:
          transformed_frames[frame, i, k] = 0
      else:
        ans = 0
        for p_joint, p_obj in data[joint][channel].items():
          for p_channel, p_val in p_obj.items():
            ans += p_val * performer.frame_joint_channel(frame, p_joint, p_channel)
        transformed_frames[frame, i, k] = ans


with open('../model/nao_heirarchy5.bvh') as f:
  model = f.readlines()

with open('../processed/wave3.bvh', 'w') as f:
  f.write(''.join(model[:-3]))
  f.write("Frames: " + str(performer.nframes) + "\n")
  f.write("Frame Time: " + str(performer.frame_time) + "\n")
  for fr in range(performer.nframes):
    f.write("0 0 0 ")
    for i in range(len(end_user_joint_names)):
      for k in range(len(channels)): 
        f.write(str(transformed_frames[fr, i, k]) + " ")
    f.write("\n")
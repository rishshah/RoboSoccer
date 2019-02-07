from bvh import Bvh
import math

class MotionClip(object):
    def __init__(self, mocap_file: str, constraints_file: str):
        print(mocap_file)
        with open(mocap_file) as f:
            self.mocap = Bvh(f.read())
        with open(constraints_file) as f:
            content = f.readlines()

        content = [x.strip() for x in content] 
        content = [x.split() for x in content] 

        channel_mapping = ['Xrotation', 'Yrotation', 'Zrotation']
        self.mapping = {}
        for joint in content:
            self.mapping[joint[0]] = []
            for i in range(1,4):
                if (joint[i] != '0'):
                    self.mapping[joint[0]].append((channel_mapping[i-1], joint[i].lower()))

    def get_pose(self, time:float):
        pose = {}
        # print('(get_pose) time, ftime '  + str(time)  + " "  + str(self.mocap.frame_time))
        frame = int(round(time / self.mocap.frame_time)) 
        if frame >= self.mocap.nframes:
            return None

        print("(get_pose) FN : ", frame)

        for joint in self.mocap.get_joints_names():
            for channel in self.mapping[joint]:
                curr_angle = self.mocap.frame_joint_channel(frame, joint, channel[0])
                # init_angle = self.mocap.frame_joint_channel(0, joint, channel[0])
                pose[str(channel[1]).lower()] =  curr_angle;
        return pose

    def similarity(self, time:float, actual_pose, keys):
        target_pose = self.get_pose(time)
        if target_pose is not None:
            dist = self.euclead_distance(actual_pose, target_pose, keys)
            # if dist > 50:
            #     print(actual_pose, target_pose)
            return - dist

    def euclead_distance(self, a, b, keys):
        ans = 0
        for x in b.keys():
            if x in keys:
                ans += (a[x] - b[x])*(a[x] - b[x])
        return math.sqrt(ans) 

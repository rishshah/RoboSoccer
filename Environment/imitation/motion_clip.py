from bvh import Bvh
import math

class MotionClip(object):
    def __init__(self, mocap_file: str, constraints_file: str):
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
        frame = math.floor(time / self.mocap.frame_time)
        if frame >= self.mocap.nframes:
            print("(get_pose) frame >= " + str(frame >= self.mocap.nframes))
            return None

        # print("(get_pose) FN : ", time , frame)
        for joint in self.mocap.get_joints_names():
            for channel in self.mapping[joint]:
                curr_angle = self.mocap.frame_joint_channel(frame, joint, channel[0])
                pose[str(channel[1]).lower()] =  curr_angle;
        return pose

    def similarity(self, time, actual_pose, keys):
        target_pose = self.get_pose(time)
        if target_pose is not None:
            ret = self.get_pose(time + self.mocap.frame_time)
            # for joint in ret:
            #     if joint in [
            #         "he2",
            #         "lae1",
            #         "lle3",
            #         "lle4",
            #         "lle5",
            #         "rae1",
            #         "rle3",
            #         "rle4",
            #         "rle5",
            #     ]:
            #         ret[joint] *= -1
            return ret, self.euclead_distance(actual_pose, target_pose, keys)

    def euclead_distance(self, a, b, keys):
        ans = 0
        for x in b.keys():
            # if x in keys:
            #     if x in [
            #         "he2",
            #         "lae1",
            #         "lle3",
            #         "lle4",
            #         "lle5",
            #         "rae1",
            #         "rle3",
            #         "rle4",
            #         "rle5",
            #     ]:
            #         b[x] *= -1
                # print("(Diff) ", x, a[x], b[x])
                ans += (a[x] - b[x])*(a[x] - b[x])
        return ans

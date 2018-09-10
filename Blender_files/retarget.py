from bvh import Bvh

with open('Retargetting/run.bvh') as f:
	mocap = Bvh(f.read())

with open('model/constraints_updated.txt') as f:
	content = f.readlines()

content = [x.strip() for x in content] 
content = [x.split() for x in content] 
channel_mapping = ['Xrotation', 'Yrotation', 'Zrotation']

mapping = {}
for joint in content:
	mapping[joint[0]] = []
	for i in range(1,4):
		if (joint[i] != '0'):
			mapping[joint[0]].append((channel_mapping[i-1], joint[i]))

start_skill_str = "STARTSKILL SKILL_TEST\n"
start_frame_str = "STARTSTATE\n"
end_frame_str   = "ENDSTATE\n"
end_skill_str = "ENDSKILL\n"

theta = 3
prev_frame = 0
k = 0.5

with open('model/test.skl', 'w') as f:
	f.write(start_skill_str+ "\n")	
	
	
	for frame in range(2, mocap.nframes, 4):
		if (frame == 2):
			frame_data_str = "settar "
		else	:
			frame_data_str = "inctar "
		wait_frame_str  = "wait " + str(round(k * mocap.frame_time * (frame - prev_frame),1)) + " end\n"
		write = False
		for joint in mocap.get_joints_names():
			for channel in mapping[joint]:
				this_frame_val = mocap.frame_joint_channel(frame, joint, channel[0])
				prev_frame_val = mocap.frame_joint_channel(prev_frame, joint, channel[0])
				print(frame, prev_frame, channel, this_frame_val, prev_frame_val)
				if "6" not in channel[1] and "5" not in channel[1] and abs(this_frame_val - prev_frame_val) > theta: 
					write = True
					frame_data_str += " " + str(channel[1]) + " " + str(round(this_frame_val - prev_frame_val, 1))
		if (write):
			prev_frame = frame		
			f.write(start_frame_str)
			f.write(frame_data_str + " end \n")
			f.write(wait_frame_str)
			f.write(end_frame_str + "\n")

	f.write(end_skill_str)

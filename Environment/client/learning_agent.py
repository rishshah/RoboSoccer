# coding=utf-8
import sys
import random as rand
from agent import BaseAgent
from monitor import Monitor

HOST, PORT = sys.argv[1:]
PORT = int(PORT)


class LearningAgent(BaseAgent):
    def run_every_cycle(self, preceptors):
        # Runs every server cycle
        self.set_hinge_joint(name=rand.choice(["rle1", "rle2", "rle3", "rle4", "rle5", "rle6"]),
                             axis1_speed=rand.randint(-1000, 1000))  # Move right leg randomly

        self.set_hinge_joint(name=rand.choice(["lle1", "lle2", "lle3", "lle4", "lle5", "lle6"]),
                             axis1_speed=rand.randint(-1000, 1000))  # Move left leg randomly

        self.set_hinge_joint(name=rand.choice(["rae1", "rae2", "rae3", "rae4"]),
                             axis1_speed=rand.randint(-1000, 1000))  # Move right arm randomly

        self.set_hinge_joint(name=rand.choice(["lae1", "lae2", "lae3", "lae4"]),
                             axis1_speed=rand.randint(-1000, 1000))  # Move left arm randomly

        print(preceptors)


agent = LearningAgent(host=HOST, port=PORT, teamname="UTAustinVilla_Base")
monitor = Monitor()

try:

    agent.start_cycle()
    
    monitor.set_time(10)
    monitor.print_status()

    monitor.set_playmode()
    monitor.print_status()
    
        
except(KeyboardInterrupt, SystemExit):
    agent.disconnect()
    monitor.disconnect()
    sys.exit()

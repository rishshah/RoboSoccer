import sys
sys.path.append('../')

from server.simspark_server import SimSparkServer
import client.effectors as ef
import sexpdata, math

class BaseAgent(SimSparkServer):
    MODEL_PATH = "rsg/agent/nao/nao.rsg"  # Defaults to Nao model
    FALLEN_PARAM = 9
    SPURIOUS_THRESHOLD = 20
    K = 0.9
    def __init__(self, teamname: str, player_number: int, host: str, port: int):
        """
        Args:
            teamname: Name of team to join, creates new team if it doesn't exist
            player_number: Position on team, auto selects number if 0
            host: address of server
            port: tcp port of server
        """
        super().__init__(host=host, port=port)
        self.teamname = teamname
        self.player_number = player_number
        self.cycle_message = ""

        self.acc = [0,0,0]
        self.time = 0
        self.gyr  = [0,0,0]
        self.state = {}
        self.pos = [0,0,0]
        self.orr = 0

    # Commands
    def synchronize(self):
        """Sent after every cycle of the server if AgentSyncMode is enabled, auto appended after every cycle"""
        # TODO(LOGGING) Log appending syncronize message
        self.cycle_message += ef.synchronize()

    def set_hinge_joint(self, name: str, axis1_speed: float):
        """
        Sets speed of axis on hinge joint

        Args:
            name: Name of joint to set
            axis1_speed: Speed value to set on axis, in radians per second. Speed will be maintained until new value set
        """
        # TODO(LOGGING) Log appending hing_joint message
        self.cycle_message += ef.hinge_joint(name=name, ax1=axis1_speed)

    def beam(self, x_pos: float, y_pos: float, direction: float):
        """
        Set player position at beginning of each half. Middle of field is 0,0

        Args:
            x_pos: X coordinate of player
            y_pos: Y coordinate of player
            direction: Direction of player. 0 points to +X axis, 90 points to +Y axis
        """
        # TODO(GENERAL) Figure out when to use beam message
        # TODO(LOGGING) Log appending beam message
        self.cycle_message += ef.beam(x=x_pos, y=y_pos, rot=direction)

    # Server stuff
    def _parse_preceptors(self, raw_preceptors):
        """Takes raw preceptor data and gives usable data"""
        # print('(_parse_preceptors) message -> ', raw_preceptors)
        
        data = sexpdata.loads("(" + raw_preceptors + ")")
        self.time = data[0][-1][-1]
        self.gyr  = data[2][-1][-3:]
        self.filter_acc(data[3][-1][-3:])
        self.state = {}
        for arg in data[4:]:
            if sexpdata.dumps(arg[0]) == 'HJ':
                self.state[sexpdata.dumps(arg[-2][-1]).replace('j','e')] = float(arg[-1][-1])

        for arg in data[4:]:
            if sexpdata.dumps(arg[0]) == 'See':
                for a in arg[1:] :
                    if sexpdata.dumps(a[0]) == 'mypos':
                        self.pos = a[1:]
                    if sexpdata.dumps(a[0]) == 'myorien':
                        self.orr = a[-1]
                        break
                break

        # for s in self.state:
        #     if s in ["lle4", "rle4","lle3", "rle3"]:
        #         print(s, self.state[s])
        return self.state, self.acc, self.gyr, self.pos, self.orr, float(self.time), self.is_fallen(), 

    def initialize(self):
        """Creates player model, and registers on a team"""
        # TODO(LOGGING) Log initializing agent on server
        self.connect()
        self.send_message(ef.create(filename=self.MODEL_PATH))
        self.send_message(ef.init(player_number=self.player_number,
                                  teamname=self.teamname))
        self.acc = [0,0,0]
        self.acc = [0,0,0]
        self.time = 0
        self.gyr  = [0,0,0]
        self.state = {}

        _, _, _, _, _, time, _, = self._parse_preceptors(self.receive_message())
        return time    

    def step(self, action):
        # Reset message
        self.cycle_message = ""
        
        for joint, angle in action.items():
            self.set_hinge_joint(name=joint, axis1_speed=angle)
            
        # Append sync message
        self.synchronize()
        
        # Send entire message
        self.send_message(self.cycle_message)

        return self._parse_preceptors(self.receive_message())

    def is_fallen(self):
        fallenUp = float(self.acc[0]) < -self.FALLEN_PARAM
        fallenDown = float(self.acc[0]) > self.FALLEN_PARAM
        fallenRight = float(self.acc[1]) < -self.FALLEN_PARAM
        fallenLeft = float(self.acc[0]) >  self.FALLEN_PARAM
        # print('is_fallen) ',accel)
        return fallenUp or fallenDown or fallenRight or fallenLeft
    
    def filter_acc(self, acc):
        for i in range(3):
            if math.fabs(acc[i]) > self.SPURIOUS_THRESHOLD:
                acc[i] = 0
        
        corrected_acc = [0,0,0]
        corrected_acc[0] = -acc[1]
        corrected_acc[1] = acc[0]
        corrected_acc[2] = -acc[2]

        self.acc[0] = self.K*self.acc[0] + (1-self.K)*corrected_acc[0];
        self.acc[1] = self.K*self.acc[1] + (1-self.K)*corrected_acc[1];
        self.acc[2] = self.K*self.acc[2] + (1-self.K)*corrected_acc[2];

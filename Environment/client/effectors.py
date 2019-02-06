def message_factory(effector_string):
    """Makes messages easy to define"""
    def message(**kwargs):
        return effector_string.format(**kwargs)

    return message

# Agent Message Types
create = message_factory("(scene {filename})(syn)")
hinge_joint = message_factory("({name} {ax1})")
synchronize = message_factory("(syn)")
init = message_factory("(init (unum {player_number}) (teamname {teamname}))(syn)")
beam = message_factory("(beam {x} {y} {rot})")

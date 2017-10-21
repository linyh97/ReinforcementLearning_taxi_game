from gym.envs.toy_text import taxi, discrete
from gym.envs.registration import registry, register, make, spec


register(
    id='Assignment1-Taxi-v2',
    entry_point='gym.envs.toy_text.taxi:TaxiEnv',

)


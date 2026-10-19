import jax
print('jax', jax.__version__)
print('devices:', jax.devices())

import openpi, openpi_client
from openpi.training import config as _config

c = _config.get_config('pi05_genie_sim_manip_20260613')
print('config OK:', c.name)
print('action_horizon:', c.model.action_horizon)
print('output_dim:', c.data.output_dim, 'include_waist:', c.data.include_waist)

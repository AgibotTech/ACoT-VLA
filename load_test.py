import time
import numpy as np

from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config

CONFIG = 'pi05_genie_sim_manip_20260613'
CKPT = 'checkpoints/manipulation_pi05'

print('=== loading policy ===', flush=True)
t0 = time.time()
cfg = _config.get_config(CONFIG)
policy = _policy_config.create_trained_policy(cfg, CKPT)
print('loaded in %.1fs' % (time.time() - t0), flush=True)

# Synthetic obs matching the genie_sim wire protocol for G2_omnipicker.
rng = np.random.default_rng(0)
obs = {
    'images': {
        'top_head': rng.integers(0, 255, (400, 640, 3), dtype=np.uint8),
        'hand_left': rng.integers(0, 255, (1056, 1280, 3), dtype=np.uint8),
        'hand_right': rng.integers(0, 255, (1056, 1280, 3), dtype=np.uint8),
    },
    # arm(14) + gripper(2) + waist(5) = 21
    'state': np.zeros(21, dtype=np.float32),
    'prompt': 'pick up the block and place it into the box',
}

print()
print('=== infer (1st call, includes JIT compile) ===', flush=True)
t0 = time.time()
out = policy.infer(obs)
print('infer #1: %.1fs' % (time.time() - t0), flush=True)

acts = np.asarray(out['actions'])
print()
print('actions shape:', acts.shape, 'dtype:', acts.dtype)
print('finite:', bool(np.isfinite(acts).all()))
print('min %.4f  max %.4f  mean %.4f  std %.4f' % (acts.min(), acts.max(), acts.mean(), acts.std()))
print()
print('per-group ranges (protocol slicing):')
print('  left_arm   [0:7]   min %+.4f max %+.4f' % (acts[:, 0:7].min(), acts[:, 0:7].max()))
print('  right_arm  [7:14]  min %+.4f max %+.4f' % (acts[:, 7:14].min(), acts[:, 7:14].max()))
print('  left_eff   [14]    min %+.4f max %+.4f' % (acts[:, 14].min(), acts[:, 14].max()))
print('  right_eff  [15]    min %+.4f max %+.4f' % (acts[:, 15].min(), acts[:, 15].max()))
if acts.shape[1] > 16:
    print('  waist      [16:]   min %+.4f max %+.4f' % (acts[:, 16:].min(), acts[:, 16:].max()))

print()
print('=== infer (2nd call, warm) ===', flush=True)
t0 = time.time()
out2 = policy.infer(obs)
dt = time.time() - t0
print('infer #2: %.3fs  (%.1f Hz)' % (dt, 1.0 / dt))

a2 = np.asarray(out2['actions'])
print('deterministic across calls:', bool(np.allclose(acts, a2)))
print()
print('RESULT: shape==(30,21) ->', acts.shape == (30, 21))

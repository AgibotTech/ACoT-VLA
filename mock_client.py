import time
import numpy as np
from openpi_client import websocket_client_policy

print('=== connecting to ws://localhost:8000 ===', flush=True)
client = websocket_client_policy.WebsocketClientPolicy(host='localhost', port=8000)
print('server metadata:', client.get_server_metadata())

rng = np.random.default_rng(42)
obs = {
    'images': {
        'top_head': rng.integers(0, 255, (400, 640, 3), dtype=np.uint8),
        'hand_left': rng.integers(0, 255, (1056, 1280, 3), dtype=np.uint8),
        'hand_right': rng.integers(0, 255, (1056, 1280, 3), dtype=np.uint8),
    },
    'state': np.zeros(21, dtype=np.float32),
    'prompt': 'pick up the block and place it into the box',
}

print()
print('--- WHAT THE CLIENT SENDS ---')
for k, v in obs.items():
    if k == 'images':
        for cam, img in v.items():
            print('  images[%-11s] %-18s %s' % (cam, img.shape, img.dtype))
    elif isinstance(v, np.ndarray):
        print('  %-20s %-18s %s' % (k, v.shape, v.dtype))
    else:
        print('  %-20s %r' % (k, v))

print()
print('=== infer (1st, JIT) ===', flush=True)
t0 = time.time(); out = client.infer(obs); print('  %.1fs' % (time.time()-t0))

t0 = time.time(); out = client.infer(obs); dt = time.time()-t0
print('=== infer (2nd, warm) === %.3fs (%.1f Hz)' % (dt, 1/dt))

print()
print('--- WHAT THE SERVER RETURNS ---')
for k, v in out.items():
    a = np.asarray(v)
    print('  %-20s %-18s %s' % (k, a.shape, a.dtype))

acts = np.asarray(out['actions'])
print()
print('actions[0]  (first timestep, 21 dims):')
print(' ', np.array2string(acts[0], precision=4, suppress_small=True))
print()
print('round-trip OK:', acts.shape == (30, 21) and bool(np.isfinite(acts).all()))

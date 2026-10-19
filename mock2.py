import sys, time, traceback
import numpy as np

try:
    from openpi_client import websocket_client_policy
    print('connecting...', flush=True)
    client = websocket_client_policy.WebsocketClientPolicy(host='localhost', port=8000)
    print('connected', flush=True)
    print('metadata:', client.get_server_metadata(), flush=True)

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
    print('sending infer...', flush=True)
    t0 = time.time()
    out = client.infer(obs)
    print('infer #1 %.1fs' % (time.time()-t0), flush=True)
    t0 = time.time(); out = client.infer(obs); dt = time.time()-t0
    print('infer #2 %.3fs (%.1f Hz)' % (dt, 1/dt), flush=True)
    print('--- SERVER RETURNS ---', flush=True)
    for k, v in out.items():
        a = np.asarray(v)
        print('  %-18s %-14s %s' % (k, a.shape, a.dtype))
    acts = np.asarray(out['actions'])
    print('actions[0]:', np.array2string(acts[0], precision=4, suppress_small=True))
    print('OK:', acts.shape == (30,21) and bool(np.isfinite(acts).all()))
except Exception:
    traceback.print_exc()
    sys.exit(1)

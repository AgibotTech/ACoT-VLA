import time
import numpy as np, cv2
import websockets.sync.client
from openpi_client import msgpack_numpy

URI = 'ws://localhost:8000'
packer = msgpack_numpy.Packer()

def jpeg(h, w, rng):
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode('.jpg', img)
    return {'encoding': 'JPEG', 'image_data': buf.tobytes(), 'height': h, 'width': w}

def build(task, prompt, waist_vals):
    rng = np.random.default_rng(7)
    return {
        'images': {
            'head': jpeg(400, 640, rng),
            'hand_left': jpeg(1056, 1280, rng),
            'hand_right': jpeg(1056, 1280, rng),
        },
        'states': {
            'head_joint_states': [],
            'arm_joint_states': [0.0]*14,
            'waist_joint_states': waist_vals,
            'gripper_states': [0.0]*2,
        },
        'prompt': prompt,
        'robot_type': 'G2_omnipicker',
        'task_name': task,
        'episode_idx': 0,
        'episode_done': False,
    }

conn = websockets.sync.client.connect(URI, compression=None, max_size=None, open_timeout=30)
print('connected', flush=True)

def call(params):
    conn.send(packer.pack({'method': 'infer', 'params': params}))
    raw = conn.recv()
    if isinstance(raw, str):
        raise RuntimeError('server error: ' + raw)
    return msgpack_numpy.unpackb(raw)

cases = [
    ('sorting_packages', 'sort the packages into the correct bins', [0.0]*5),
    ('sorting_packages', 'sort the packages into the correct bins', [0.1, -0.05, 0.02, 0.0, 0.3]),
    ('place_block_into_box', 'pick up the block and place it into the box', [0.0]*5),
]

for task, prompt, wv in cases:
    p = build(task, prompt, wv)
    t0 = time.time(); r = call(p); dt = time.time()-t0
    if 'error' in r:
        print('ERROR for', task); print(r['error']); continue
    res = r['result']
    keys = sorted(res.keys())
    has_waist = 'waist' in res
    print('')
    print('task=%-22s waist_state=%s' % (task, wv))
    print('  %.3fs  keys=%s' % (dt, keys))
    print('  waist present: %s' % has_waist)
    if has_waist:
        w = np.asarray(res['waist']['values'])
        print('  waist shape %s  kind=%s' % (w.shape, res['waist']['kind']))
        print('  waist[0]:', np.array2string(w[0], precision=4, suppress_small=True))
        print('  waist per-dim range:')
        for i in range(w.shape[1]):
            print('    dim%d  min %+.4f  max %+.4f  std %.5f' % (i, w[:,i].min(), w[:,i].max(), w[:,i].std()))
conn.close()
print('')
print('done')

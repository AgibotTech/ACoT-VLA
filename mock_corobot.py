import time, sys
import numpy as np, cv2
import websockets.sync.client
from openpi_client import msgpack_numpy

URI = 'ws://localhost:8000'

def jpeg(h, w, rng):
    img = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode('.jpg', img)
    assert ok
    return {'encoding': 'JPEG', 'image_data': buf.tobytes(), 'height': h, 'width': w}

rng = np.random.default_rng(42)
params = {
    'images': {
        'head': jpeg(400, 640, rng),
        'hand_left': jpeg(1056, 1280, rng),
        'hand_right': jpeg(1056, 1280, rng),
    },
    'states': {
        'head_joint_states': [],
        'arm_joint_states': [0.0]*14,
        'waist_joint_states': [0.0]*5,
        'gripper_states': [0.0]*2,
    },
    'prompt': 'pick up the block and place it into the box',
    'robot_type': 'G2_omnipicker',
    'task_name': 'place_block_into_box',
    'episode_idx': 0,
    'episode_done': False,
}

print('--- WHAT THE CLIENT SENDS ---')
print('  method: infer')
for cam, d in params['images'].items():
    print('  images[%-11s] JPEG %dx%d  %d bytes' % (cam, d['height'], d['width'], len(d['image_data'])))
for k, v in params['states'].items():
    print('  states[%-19s] len=%d' % (k, len(v)))
print('  prompt: %r' % params['prompt'])
print('  robot_type: %s' % params['robot_type'])
print('', flush=True)

packer = msgpack_numpy.Packer()
conn = websockets.sync.client.connect(URI, compression=None, max_size=None, open_timeout=30)
print('connected (no metadata packet expected)', flush=True)

def call():
    conn.send(packer.pack({'method': 'infer', 'params': params}))
    raw = conn.recv()
    if isinstance(raw, str):
        raise RuntimeError('server error: ' + raw)
    return msgpack_numpy.unpackb(raw)

t0 = time.time(); resp = call(); print('infer #1 %.1fs (JIT)' % (time.time()-t0), flush=True)
if 'error' in resp:
    print('SERVER ERROR:'); print(resp['error']); sys.exit(1)
t0 = time.time(); resp = call(); dt = time.time()-t0
print('infer #2 %.3fs (%.1f Hz)' % (dt, 1/dt))
print('')
print('--- WHAT THE SERVER RETURNS ---')
print('  top-level keys:', list(resp.keys()))
res = resp['result']
for k, v in res.items():
    if isinstance(v, dict):
        vals = np.asarray(v['values'])
        print('  %-16s kind=%-10s values %s' % (k, v.get('kind'), vals.shape))
    else:
        print('  %-16s %s' % (k, np.asarray(v).shape))
print('')
la = np.asarray(res['left_arm']['values'])
print('left_arm[0] (7 joints):', np.array2string(la[0], precision=4, suppress_small=True))
print('horizon H =', la.shape[0])
print('OK:', la.shape[0] == 30)
conn.close()

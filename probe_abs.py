import numpy as np, cv2, websockets.sync.client
from openpi_client import msgpack_numpy
packer = msgpack_numpy.Packer()
def jpeg(h,w,rng):
    img = rng.integers(0,255,(h,w,3),dtype=np.uint8)
    ok,buf = cv2.imencode('.jpg',img)
    return {'encoding':'JPEG','image_data':buf.tobytes(),'height':h,'width':w}
def build(arm, task='open_door', prompt='open the door'):
    rng = np.random.default_rng(7)
    return {'images':{'head':jpeg(400,640,rng),'hand_left':jpeg(1056,1280,rng),'hand_right':jpeg(1056,1280,rng)},
        'states':{'head_joint_states':[],'arm_joint_states':list(arm),'waist_joint_states':[0.0]*5,'gripper_states':[0.0]*2},
        'prompt':prompt,'robot_type':'G2_omnipicker','task_name':task,'episode_idx':0,'episode_done':False}
conn = websockets.sync.client.connect('ws://localhost:8000',compression=None,max_size=None,open_timeout=30)
def call(p):
    conn.send(packer.pack({'method':'infer','params':p}))
    return msgpack_numpy.unpackb(conn.recv())

# Same images, two very different arm states. If output is ABSOLUTE in the
# true sense, predicted joints should sit near each respective state.
for label, off in (('zeros', 0.0), ('offset+0.5', 0.5)):
    arm = [off]*14
    r = call(build(arm))['result']
    la = np.asarray(r['left_arm']['values']); ra = np.asarray(r['right_arm']['values'])
    print(f"state={off:+.2f}  left[0]={np.array2string(la[0],precision=4,suppress_small=True)}")
    print(f"            mean_left={la.mean():+.4f} mean_right={ra.mean():+.4f} kind={r['left_arm']['kind']}")
    print(f"            drift over horizon: left {la[0].mean():+.4f} -> {la[-1].mean():+.4f}")
conn.close()

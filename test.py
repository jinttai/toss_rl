import numpy as np
import mujoco
import mujoco.viewer
import time
from scipy.spatial.transform import Rotation as R


model = mujoco.MjModel.from_xml_path("mujoco_src/spacerobot_twoarm_3dof.xml")
data = mujoco.MjData(model)

model_fixed = mujoco.MjModel.from_xml_path("mujoco_src/spacerobot_twoarm_3dof_base_fixed.xml")
data_fixed = mujoco.MjData(model_fixed)

def quaternion_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def get_relative_rotation_quaternion_manual(q_initial, q_target):
    q_initial_inv = quaternion_conjugate(q_initial)
    q_relative_transform = quaternion_multiply(q_initial_inv, q_target)
    return q_relative_transform

def jacobian_vel(model, data, body_id):
    """ 특정 바디의 자코비안을 이용해 속도를 계산합니다. """
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBodyCom(model, data, jacp, jacr, body_id)
    return jacp @ data.qvel

def _initialize_qpos( qpos_arm_joints):
    """ 로봇팔의 초기 자세를 설정하고, 그에 맞게 베이스의 위치와 방향을 조정합니다. """
    weld_quat, weld_pos = np.array([1, 0, 0, 0]), np.array([1.0, 1.0, 1.0])
    
    data_fixed.qpos[:] = qpos_arm_joints
    data_fixed.qvel[:] = np.zeros_like(data_fixed.qvel)
    mujoco.mj_forward(model_fixed, data_fixed)

    site_id = mujoco.mj_name2id(model_fixed, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    body_id = model_fixed.body("arm1_ee").id
    site_xquat = data_fixed.xquat[body_id]
    site_xpos = data_fixed.site_xpos[site_id]

    data.qpos[7:13] = qpos_arm_joints
    quat_relative = get_relative_rotation_quaternion_manual(site_xquat, weld_quat)
    data.qpos[3:7] = quaternion_multiply(quat_relative, data.qpos[3:7])
    data.qpos[0:3] = weld_pos - _rotate_vector_by_quaternion(site_xpos, quat_relative)
    data.qvel[:] = np.zeros_like(data.qvel)

def _rotate_vector_by_quaternion(vector, quat_rotation_wxyz): 
    # Scipy's Rotation expects quaternion as [x, y, z, w]
    quat_xyzw = quat_rotation_wxyz[[1,2,3,0]] 
    return R.from_quat(quat_xyzw).apply(vector)


# Initialize the robot arm joints
qpos_arm_joints = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Adjust as needed
_initialize_qpos(qpos_arm_joints)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
        for i in range(1, model.nbody-1):
            print(f"Body {i}: {model.body(i).name}, mass: {model.body(i).mass}")
            print(data.xipos[i])
            print("Velocity:", jacobian_vel(model, data, i))
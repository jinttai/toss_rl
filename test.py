import numpy as np
import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("mujoco_src/spacerobot_twoarm_3dof_ee_fixed.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while True:
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
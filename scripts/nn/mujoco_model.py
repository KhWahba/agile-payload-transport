import mujoco as mj
import numpy as np

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)

class MujocoModel:
    def __init__(self, xml_path=None, init_state=None):
        self.model      = mj.MjModel.from_xml_path(xml_path)
        self.data       = mj.MjData(self.model)
        self.num_bodies = self.model.nbody - 1 # exclude world
        self.num_robots = self.num_bodies - 1 # exclude payload
        self.u_nominal = self.model.body('cf21').mass*9.81 /4
        init_state_new  = self.flip_quat_from_dynoplan_to_mujoco(init_state)
        self.setState(init_state_new)
        self.data.qpos  = init_state_new[0:self.model.nq]
        self.data.qvel  = init_state_new[self.model.nq:self.model.nq+self.model.nv]
        self.data.qacc  = np.zeros(self.model.nv)
        self.data.ctrl  = np.zeros(self.model.nu)
        self.u = np.zeros(self.model.nu)

    def step(self, u):
        u = np.asarray(u, dtype=float)
        self.u = u.copy()
        u *= self.u_nominal
        self.data.ctrl  = u
        mj.mj_step(self.model, self.data)
        self.x = np.concatenate((self.data.qpos.copy(), self.data.qvel.copy()))
        self.x = self.flip_quat_from_mujoco_to_dynoplan(self.x)
    
    def setState(self, x):
        self.data.qpos  = x[0:self.model.nq]
        self.data.qvel  = x[self.model.nq:self.model.nq+self.model.nv]
        mj.mj_forward(self.model, self.data)
        self.x = np.concatenate((self.data.qpos.copy(), self.data.qvel.copy()))
        self.x = self.flip_quat_from_mujoco_to_dynoplan(self.x)

    def getState(self):
        return self.x.copy()
    
    def flip_quat_from_dynoplan_to_mujoco(self, x):
        x_new = x.copy()
        for i in range(self.num_bodies):
            old_quat           = x_new[7*i+3:7*i+7].copy()
            x_new[7*i+3]       = old_quat[3]
            x_new[7*i+4:7*i+7] = old_quat[0:3]
        return x_new

    def flip_quat_from_mujoco_to_dynoplan(self, x):
        x_new = x.copy()
        for i in range(self.num_bodies):
            old_quat           = x_new[7*i+3:7*i+7].copy()
            x_new[7*i+3:7*i+6] = old_quat[1:4]
            x_new[7*i+6]       = old_quat[0]
        return x_new



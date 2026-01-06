import mujoco as mj
import numpy as np
from mujoco.glfw import glfw
from mujoco_model import MujocoModel
import yaml
import subprocess
import shutil
import os
from matplotlib.backends.backend_pdf import PdfPages
import rowan
import matplotlib.pyplot as plt

np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True)



class Simulator:
    def __init__(self, xml_path, init_state, log=False):
        self.robot = MujocoModel(xml_path, init_state) 
        self.dt = self.robot.model.opt.timestep
        self.log = log 
        # log
        if log:
            self.x_traj = []
            self.u_traj = []
            self.x_traj.append(self.robot.x)
        
        # ---- NEW: create a hidden GLFW context so MuJoCo can load GL ----
        if not glfw.init():
            raise RuntimeError("GLFW init failed (needed to create an OpenGL context).")

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)                     # hidden window
        glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_API)             # desktop OpenGL (not GLES)
        glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.NATIVE_CONTEXT_API)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_ANY_PROFILE) # compatibility profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)                # <-- 2.1 fixes ARB FBO availability
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

        self._hidden_win = glfw.create_window(64, 64, "", None, None)
        if not self._hidden_win:
            glfw.terminate()
            raise RuntimeError("Failed to create hidden GLFW window/context.")
        glfw.make_context_current(self._hidden_win)
        # -------------------------------------------------------------------- #
        
        # camera settings
        self.cam = mj.MjvCamera()
        mj.mjv_defaultCamera(self.cam)
        self.cam.lookat[:] = [self.robot.data.qpos[0], self.robot.data.qpos[1], self.robot.data.qpos[2] + 0.5]
        self.cam.azimuth = 0
        self.cam.elevation = -0
        self.cam.distance = 2.5
        
        # mujoco options
        self.opt = mj.MjvOption()
        mj.mjv_defaultOption(self.opt)

        # Initialize visualization
        self.scene = mj.MjvScene(self.robot.model, maxgeom=10000)
        self.context = mj.MjrContext(self.robot.model, mj.mjtFontScale.mjFONTSCALE_150.value)

        # Pre-allocated buffers will be created on first render
        self._rgb_buf = None
        # self._depth_buf = None

    # ---- Offscreen helper
    def _prepare_offscreen(self, width: int, height: int):
        mj.mjr_setBuffer(mj.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
        if self.context.offWidth != width or self.context.offHeight != height:
            mj.mjr_resizeOffscreen(width, height, self.context)
        viewport = mj.MjrRect(0, 0, width, height)

        if (self._rgb_buf is None or
            self._rgb_buf.shape[0] != height or
            self._rgb_buf.shape[1] != width):
            self._rgb_buf = np.empty((height, width, 3), dtype=np.uint8)
            # self._depth_buf = np.empty((height, width), dtype=np.float32)

        return viewport

    def _render_frame(self, viewport):
        mj.mjv_updateScene(
            self.robot.model, self.robot.data, self.opt,
            None, self.cam, mj.mjtCatBit.mjCAT_ALL.value, self.scene
        )
        mj.mjr_render(viewport, self.scene, self.context)
        mj.mjr_readPixels(self._rgb_buf, None, viewport, self.context)
        return self._rgb_buf

    # ---- FFmpeg pipe (must exist)
    def _open_ffmpeg(self, out_path: str, width: int, height: int, fps: int, bitrate: str):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("FFmpeg not found on PATH. Please install ffmpeg.")

        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", f"{width}x{height}",
            "-framerate", str(fps),
            "-i", "-",
            "-vf", "vflip,scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264",
            "-preset", "slow",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            out_path
        ]
        return subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def simulate(self, u):
        self.robot.step(u)
        if self.log:
            self.x_traj.append(self.robot.x)
            self.u_traj.append(self.robot.u)

    @staticmethod
    def apply_camera_preset(cam, env_center, env_size, camera_view, scale_factor=1.0):
        cam.lookat[0] = float(env_center[0])
        cam.lookat[1] = float(env_center[1])
        cam.lookat[2] = float(env_center[2])

        # distance heuristic from XY extent
        xy_diag = float(np.sqrt(env_size[0] ** 2 + env_size[1] ** 2))
        base_dist = max(1.0, xy_diag * 0.65) * float(scale_factor)

        v = (camera_view or "").lower()

        if v == "top":
            cam.azimuth = 0.0              # irrelevant for top-down
            cam.elevation = -90.0          # straight down
            cam.distance = max(env_size[0], env_size[1]) * 0.8 * float(scale_factor)
        elif v == "front":
            cam.azimuth = 180.0            # +x looking toward -x
            cam.elevation = -15.0
            cam.distance = base_dist
        elif v == "side":
            cam.azimuth = 90.0             # +y looking toward -y
            cam.elevation = -15.0
            cam.distance = base_dist
        elif v == "diag":
            cam.azimuth = 202.5
            cam.elevation = -15.0
            cam.distance = base_dist
        else:
            # default / unknown: a generic isometric
            cam.azimuth = 45.0
            cam.elevation = -35.0
            cam.distance = base_dist    

    def visualize(self, cam_cfg: dict, video_basename: str = "sim"):

        if not self.log or not hasattr(self, "x_traj") or len(self.x_traj) == 0:
            raise ValueError("No simulation log available to visualize.")

        width  = int(cam_cfg.get("width", 1280))
        height = int(cam_cfg.get("height", 720))
        fps    = int(cam_cfg.get("fps", 60))
        bitrate = cam_cfg.get("bitrate", "6M")
        loops   = int(cam_cfg.get("loops", 1))
        scale_factor = float(cam_cfg.get("scale_factor", 1.0))

        # views can be a list or the string "auto"
        views_cfg = cam_cfg.get("views", ["side", "top", "front", "diag"])
        if isinstance(views_cfg, str) and views_cfg.lower() == "auto":
            views = ["side", "top", "front", "diag"]
        elif isinstance(views_cfg, list):
            views = [str(v).lower() for v in views_cfg]
        else:
            views = ["side", "top", "front", "diag"]

        # environment bounds (required)
        env_min = np.asarray(cam_cfg["env_min"], dtype=float)
        env_max = np.asarray(cam_cfg["env_max"], dtype=float)
        assert env_min.size == 3 and env_max.size == 3, "env_min/env_max must be 3D vectors"

        env_center = 0.5 * (env_min + env_max)
        env_size   = env_max - env_min

        # Prepare offscreen
        viewport = self._prepare_offscreen(width, height)

        # For each view, set camera and encode
        os.makedirs(video_basename, exist_ok=True)
        for view in views:
            # Set preset
            self.apply_camera_preset(self.cam, env_center, env_size, view, scale_factor=scale_factor)

            out_path = f"{video_basename}/{view}.mp4"
            proc = self._open_ffmpeg(out_path, width, height, fps, bitrate)

            try:
                for _ in range(max(1, loops)):
                    for x in self.x_traj:
                        self.robot.setState(x)
                        frame_rgb = self._render_frame(viewport)
                        # Write raw RGB directly to ffmpeg stdin
                        proc.stdin.write(np.ascontiguousarray(frame_rgb).tobytes())
                        # proc.stdin.write(memoryview(np.ascontiguousarray(frame_rgb)))
            finally:
                proc.stdin.close()
                proc.wait()
    def flip_quat_from_dynoplan_to_rowan(self, state: np.ndarray) -> np.ndarray:
        nx = state.shape[0]
        num_bodies = nx // 13
        flipped_state = state.copy()
        for i in range(num_bodies):
            old_quat                   = state[7*i+3 : 7*i+7].copy()
            flipped_state[7*i+3]       = old_quat[3]
            flipped_state[7*i+4:7*i+7] = old_quat[0:3]
        return flipped_state
    
    def plot_sim_trajectories(self, x_ref, u_ref, payload=False, save_path=None):
        self.x_traj = np.array(self.x_traj)
        self.x_traj = np.array([self.flip_quat_from_dynoplan_to_rowan(x) for x in self.x_traj])
        x_ref = np.array([self.flip_quat_from_dynoplan_to_rowan(x) for x in x_ref])
        self.u_traj = np.array(self.u_traj)
        self.u_ref = np.array(u_ref)
        num_steps = self.x_traj.shape[0]
        time_vector = np.arange(num_steps) * self.dt
        title_suffix = " with Payload" if payload else ""

        figs = []
        print("Generating trajectory plots...")

        payload_pos = self.x_traj[:, 0:3]
        payload_pos_ref = x_ref[:, 0:3]
        payload_vel = self.x_traj[:, 14:17]
        payload_vel_ref = x_ref[:, 14:17]

        # Page 1: Payload Position
        fig1, axs1 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for i, label in enumerate(['x', 'y', 'z']):
            axs1[i].plot(time_vector, payload_pos[:, i], label=f'real {label}')
            axs1[i].plot(time_vector, payload_pos_ref[:, i], label=f'ref {label}')
            axs1[i].set_ylabel(f'{label} (m)')
            axs1[i].legend()
            axs1[i].grid(True)
        axs1[2].set_xlabel("Time (s)")
        fig1.suptitle("Payload Position")
        figs.append(fig1)

        # Page 2: Payload Velocity
        fig2, axs2 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for i, label in enumerate(['vx', 'vy', 'vz']):
            axs2[i].plot(time_vector, payload_vel[:, i], label=f'real {label}')
            axs2[i].plot(time_vector, payload_vel_ref[:, i], label=f'ref {label}')

            axs2[i].set_ylabel(f'{label} (m/s)')
            axs2[i].legend()
            axs2[i].grid(True)
        axs2[2].set_xlabel("Time (s)")
        fig2.suptitle("Payload Velocity")
        figs.append(fig2)
        self.num_robots = self.robot.num_robots
        nq = self.robot.model.nq
        # UAV plots
        for i in range(0, self.num_robots):
            uav_pos = self.x_traj[:, 7+7*i : 7+7*i+3]
            uav_pos_ref = x_ref[:, 7+7*i : 7+7*i+3]
            uav_quat = self.x_traj[:, 7+7*i+3 : 7+7*i+7]
            uav_quat_ref = x_ref[:, 7+7*i+3 : 7+7*i+7]
            # velocity
            uav_vel = self.x_traj[:, nq + 6 + 6*i  : nq + 6 + 6*i + 3 ]
            uav_vel_ref = x_ref[:, nq + 6 +6*i : nq +  6 +6*i + 3]
            uav_ang_vel = self.x_traj[:, nq + 6 + 6*i + 3 : nq + 6 + 6*i + 6]
            uav_ang_vel_ref = x_ref[:, nq + 6 + 6*i + 3 : nq + 6 + 6*i + 6]
            # Convert UAV quaternion [w, x, y, z] to roll, pitch, yaw using rowan
            # rowan uses [w, x, y, z] and returns [roll, pitch, yaw] in radians
            uav_rpy = rowan.to_euler(uav_quat, 'xyz')
            uav_rpy_deg = np.rad2deg(uav_rpy)
            uav_rpy_ref = rowan.to_euler(uav_quat_ref, 'xyz')
            uav_rpy_deg_ref = np.rad2deg(uav_rpy_ref)



            # Page 3: UAV Position
            fig3, axs3 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            for i, label in enumerate(['x', 'y', 'z']):
                axs3[i].plot(time_vector, uav_pos[:, i], label=f'UAV {label}')
                axs3[i].plot(time_vector, uav_pos_ref[:, i], label=f'ref {label}')
                axs3[i].set_ylabel(f'{label} (m)')
                axs3[i].legend()
                axs3[i].grid(True)
            axs3[2].set_xlabel("Time (s)")
            fig3.suptitle("UAV Position")
            figs.append(fig3)

            # Page 4: UAV Velocity
            fig4, axs4 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            for i, label in enumerate(['vx', 'vy', 'vz']):
                axs4[i].plot(time_vector, uav_vel[:, i], label=f'UAV {label}')
                axs4[i].plot(time_vector, uav_vel_ref[:, i], label=f'ref {label}')
                axs4[i].set_ylabel(f'{label} (m/s)')
                axs4[i].legend()
                axs4[i].grid(True)
            axs4[2].set_xlabel("Time (s)")
            fig4.suptitle("UAV Velocity")
            figs.append(fig4)

            # Page 5: UAV Angles (roll, pitch, yaw)
            fig5, axs5 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            for i, label in enumerate(['Roll', 'Pitch', 'Yaw']):
                axs5[i].plot(time_vector, uav_rpy_deg[:, i], label=f'UAV {label}')
                axs5[i].plot(time_vector, uav_rpy_deg_ref[:, i], label=f'ref {label}')
                axs5[i].set_ylabel(f'{label} (deg)')
                axs5[i].legend()
                axs5[i].grid(True)
            axs5[2].set_xlabel("Time (s)")
            fig5.suptitle("UAV Euler Angles")
            figs.append(fig5)

            # Page 6: UAV Angular Velocities
            fig6, axs6 = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            for i, label in enumerate(['p', 'q', 'r']):
                axs6[i].plot(time_vector, uav_ang_vel[:, i], label=f'UAV {label}')
                axs6[i].plot(time_vector, uav_ang_vel_ref[:, i], label=f'ref {label}')
                axs6[i].set_ylabel(f'{label} (rad/s)')
                axs6[i].legend()
                axs6[i].grid(True)
            axs6[2].set_xlabel("Time (s)")
            fig6.suptitle("UAV Angular Velocities")
            figs.append(fig6)

        
            # Plot 3: Control Inputs vs Time (4 subplots stacked)
            if self.u_traj.shape[0] > 0:
                time_vector_u = time_vector[0:self.u_traj.shape[0]]
                fig_u, axs_u = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
                u_labels = ["u1", "u2", "u3", "u4"]
                for i in range(4):
                    axs_u[i].plot(time_vector_u, self.u_traj[:, i], label=u_labels[i])
                    axs_u[i].plot(time_vector_u, self.u_ref[:, i], label=u_labels[i]+"_ref")
                    axs_u[i].set_ylabel(u_labels[i])
                    axs_u[i].legend()
                    axs_u[i].grid(True)
                axs_u[3].set_xlabel("Time (s)")
                fig_u.suptitle(f"Control Inputs (Motor Thrusts){title_suffix}")
                figs.append(fig_u)


        if save_path:
            print(f"Saving plots to {save_path}...")
            with PdfPages(save_path) as pdf:
                for fig in figs:
                    fig.tight_layout()
                    pdf.savefig(fig)
            print(f"Plots saved to {save_path}")
            plt.close('all') # Close all figures to free memory
        else:
            plt.tight_layout()
            plt.show()
    

    def __del__(self):
        try:
            mj.mjr_freeContext(self.context)
            mj.mjv_freeScene(self.scene)
        except Exception:
            pass



# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import multiprocessing
import os
import warnings
from contextlib import contextmanager
from multiprocessing import connection
from typing import Any, Callable, Optional, Union

import gym
import numpy as np

from rlinf.envs.sim.libero.safety_auditor import audit_predicate_instances
from rlinf.envs.sim.libero.utils import get_libero_type
from rlinf.envs.venv import (
    BaseVectorEnv,
    CloudpickleWrapper,
    EnvWorker,
    ShArray,
    SubprocEnvWorker,
    SubprocVectorEnv,
    _setup_buf,
)

# ---------------------------------------------------------------------------
# Dynamic Module Import Logic for Libero Pro / Plus
# ---------------------------------------------------------------------------
libero_type = get_libero_type()

if libero_type == "pro":
    try:
        from liberopro.liberopro.envs import OffScreenRenderEnv
    except ImportError as e:
        print(
            f"[Venv] Warning: LIBERO_TYPE=pro but import failed ({e}). Falling back to standard libero..."
        )
        from libero.libero.envs import OffScreenRenderEnv

elif libero_type == "plus":
    try:
        from liberoplus.liberoplus.envs import OffScreenRenderEnv
    except ImportError as e:
        print(
            f"[Venv] Warning: LIBERO_TYPE=plus but import failed ({e}). Falling back to standard libero..."
        )
        from libero.libero.envs import OffScreenRenderEnv

else:
    try:
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError:
        try:
            from liberopro.liberopro.envs import OffScreenRenderEnv
        except ImportError:
            try:
                from liberoplus.liberoplus.envs import OffScreenRenderEnv
            except ImportError:
                raise ImportError(
                    "Could not import OffScreenRenderEnv from libero, liberopro, or liberoplus."
                )


gym_old_venv_step_type = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
gym_new_venv_step_type = tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]
warnings.simplefilter("once", DeprecationWarning)

LIBERO_CAMERA_OBS_NAMES = ("agentview_image", "robot0_eye_in_hand_image")


def _debug_worker_event(message: str) -> None:
    """Emit sparse subprocess diagnostics when explicitly requested."""
    if os.environ.get("RLINF_LIBERO_DEBUG", "0") == "1":
        device = os.environ.get("MUJOCO_EGL_DEVICE_ID", "unset")
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "unset")
        print(
            f"[libero-worker pid={os.getpid()} cuda={cuda_devices} "
            f"egl={device}] {message}",
            flush=True,
        )


@contextmanager
def _egl_process_guard():
    """Optionally serialize EGL work across simulator subprocesses.

    Set ``RLINF_LIBERO_EGL_LOCK_PATH`` to a shared file (for example
    ``/tmp/rlinf-libero-egl.lock``) to enable this conservative fallback.
    Set ``RLINF_LIBERO_EGL_LOCK_SCOPE=per_gpu`` to append the Ray worker's
    isolated ``CUDA_VISIBLE_DEVICES`` value to that path.  This keeps EGL
    calls serialized within one physical GPU while allowing different GPUs
    to render concurrently.
    It is disabled by default because correctly configured EGL drivers can
    render concurrently.  The lock intentionally covers the full env call:
    camera observations are produced internally by ``env.step`` / ``reset``.
    """
    lock_path = _egl_lock_path()
    if not lock_path or os.name != "posix":
        yield
        return

    import fcntl

    lock_dir = os.path.dirname(os.path.abspath(lock_path))
    os.makedirs(lock_dir, exist_ok=True)
    with open(lock_path, "a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _egl_lock_path() -> str:
    """Resolve the optional global or per-GPU EGL lock path."""
    lock_path = os.environ.get("RLINF_LIBERO_EGL_LOCK_PATH", "").strip()
    if not lock_path:
        return ""

    scope = os.environ.get("RLINF_LIBERO_EGL_LOCK_SCOPE", "global").strip().lower()
    if scope in {"global", "host"}:
        return lock_path
    if scope not in {"per_gpu", "gpu", "device"}:
        raise ValueError(
            "RLINF_LIBERO_EGL_LOCK_SCOPE must be global or per_gpu, "
            f"got {scope!r}"
        )

    # RLinf gives each Ray worker an isolated CUDA_VISIBLE_DEVICES value.  It
    # names the physical accelerator even though MuJoCo sees local EGL index 0.
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "unknown").strip()
    safe_device = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in visible_devices
    )
    return f"{lock_path}.gpu-{safe_device or 'unknown'}"


def _normalize_egl_device_id() -> None:
    """Translate a stale global EGL id into the worker-local namespace.

    RLinf isolates each Ray worker with ``CUDA_VISIBLE_DEVICES``.  On drivers
    where EGL enumeration is filtered by that variable, robosuite then sees a
    single device (index 0), even if the scheduler selected physical GPU 4.
    Keep a valid scheduler-provided mapping, but fall back to local index 0
    when it is outside the enumeration visible in this simulator subprocess.
    """
    if os.environ.get("MUJOCO_GL", "").lower() != "egl":
        return
    requested = os.environ.get("MUJOCO_EGL_DEVICE_ID")
    if requested is None:
        return
    try:
        from robosuite.renderers.context.egl_context import EGL

        num_devices = len(EGL.eglQueryDevicesEXT())
        requested_id = int(requested)
    except (ImportError, TypeError, ValueError):
        return
    if num_devices > 0 and not 0 <= requested_id < num_devices:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
        os.environ["EGL_DEVICE_ID"] = "0"


def _get_contact_pairs(env) -> list[tuple[str, str]]:
    """Return normalized MuJoCo geom contact pairs for SafeLIBERO auditing."""
    rob = getattr(env, "env", env)
    while hasattr(rob, "env"):
        rob = rob.env
    sim = rob.sim
    pairs = set()
    for contact in sim.data.contact[: sim.data.ncon]:
        geom1 = sim.model.geom_id2name(contact.geom1) or f"geom:{contact.geom1}"
        geom2 = sim.model.geom_id2name(contact.geom2) or f"geom:{contact.geom2}"
        pairs.add(tuple(sorted((geom1, geom2))))
    return sorted(pairs)


def _set_camera_rendering(env, enabled: bool) -> None:
    """Enable or disable LIBERO camera observables without resetting the env."""
    rob = getattr(env, "env", env)
    while hasattr(rob, "env"):
        rob = rob.env
    observables = getattr(rob, "_observables", None)
    if observables is None:
        return
    for name in LIBERO_CAMERA_OBS_NAMES:
        if name in observables:
            observables[name]._enabled = enabled


def _unwrap_task_env(env):
    """Return the innermost LIBERO task environment."""
    task_env = env
    seen = set()
    while hasattr(task_env, "env") and id(task_env) not in seen:
        seen.add(id(task_env))
        nested = task_env.env
        if nested is task_env:
            break
        task_env = nested
    return task_env


def _make_render_context_current(env) -> None:
    """Re-bind the simulator's offscreen context after non-render RPCs."""
    task_env = _unwrap_task_env(env)
    render_context = getattr(task_env.sim, "_render_context_offscreen", None)
    gl_context = getattr(render_context, "gl_ctx", None)
    if gl_context is not None:
        gl_context.make_current()


def _checked_snapshot_array(
    value: Any,
    *,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
    allow_empty: bool = False,
) -> np.ndarray:
    """Return a copied numeric snapshot array or fail before native MuJoCo."""
    array = np.asarray(value)
    if not allow_empty and array.size == 0:
        raise ValueError(f"counterfactual snapshot field {name!r} is empty")
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(
            f"counterfactual snapshot field {name!r} has shape {array.shape}, "
            f"expected {expected_shape}"
        )
    if array.dtype.kind not in "biufc":
        raise TypeError(
            f"counterfactual snapshot field {name!r} is not numeric: {array.dtype}"
        )
    if array.size and not np.isfinite(array).all():
        raise ValueError(
            f"counterfactual snapshot field {name!r} contains NaN or Inf"
        )
    return array.copy()


def _snapshot_simulator_state(env) -> dict[str, Any]:
    """Capture MuJoCo and LIBERO dynamic-obstacle state for branching."""
    task_env = _unwrap_task_env(env)
    sim = task_env.sim
    generators = getattr(task_env, "mocap_motion_generators", {})
    state = _checked_snapshot_array(
        sim.get_state().flatten(), name="mujoco_state"
    )
    snapshot = {
        "version": 2,
        "mujoco_state": state,
        "mujoco_state_shape": state.shape,
        "mocap_pos": _checked_snapshot_array(
            sim.data.mocap_pos, name="mocap_pos", allow_empty=True
        ),
        "mocap_quat": _checked_snapshot_array(
            sim.data.mocap_quat, name="mocap_quat", allow_empty=True
        ),
        "cur_time": float(getattr(task_env, "cur_time", sim.data.time)),
        "root_timestep": int(getattr(task_env, "timestep", 0)),
        "motion_generators": {
            name: copy.deepcopy(generator.__dict__)
            for name, generator in generators.items()
        },
    }
    # MjSimState only contains time/qpos/qvel.  Preserve the remaining mutable
    # integration inputs used by controllers so a branch starts from the same
    # physical state rather than merely the same visible pose.
    for field in (
        "act",
        "ctrl",
        "qacc_warmstart",
        "qfrc_applied",
        "xfrc_applied",
    ):
        if hasattr(sim.data, field):
            snapshot[field] = _checked_snapshot_array(
                getattr(sim.data, field), name=field, allow_empty=True
            )

    # get_state() itself does not render, but explicitly restore the current
    # EGL context before the immediately following env.step() RPC.
    _make_render_context_current(env)
    timestep = int(snapshot["root_timestep"])
    if timestep < 3 or timestep % 100 == 0:
        _debug_worker_event(
            f"snapshot ok timestep={timestep} state_size={state.size}"
        )
    return snapshot


def _restore_simulator_state(env, snapshot: dict[str, Any]):
    """Restore a branching snapshot and regenerate the corresponding observation."""
    task_env = _unwrap_task_env(env)
    sim = task_env.sim
    if not isinstance(snapshot, dict):
        raise TypeError("counterfactual simulator snapshot must be a dictionary")

    expected_state_shape = np.asarray(sim.get_state().flatten()).shape
    state = _checked_snapshot_array(
        snapshot.get("mujoco_state"),
        name="mujoco_state",
        expected_shape=expected_state_shape,
    )
    mocap_pos = _checked_snapshot_array(
        snapshot.get("mocap_pos"),
        name="mocap_pos",
        expected_shape=sim.data.mocap_pos.shape,
        allow_empty=True,
    )
    mocap_quat = _checked_snapshot_array(
        snapshot.get("mocap_quat"),
        name="mocap_quat",
        expected_shape=sim.data.mocap_quat.shape,
        allow_empty=True,
    )

    sim.set_state_from_flattened(state)
    sim.data.mocap_pos[:] = mocap_pos
    sim.data.mocap_quat[:] = mocap_quat
    for field in (
        "act",
        "ctrl",
        "qacc_warmstart",
        "qfrc_applied",
        "xfrc_applied",
    ):
        if field not in snapshot or not hasattr(sim.data, field):
            continue
        target = getattr(sim.data, field)
        target[:] = _checked_snapshot_array(
            snapshot[field],
            name=field,
            expected_shape=target.shape,
            allow_empty=True,
        )
    generators = getattr(task_env, "mocap_motion_generators", {})
    for name, generator_state in snapshot.get("motion_generators", {}).items():
        if name in generators:
            generators[name].__dict__.clear()
            generators[name].__dict__.update(copy.deepcopy(generator_state))
    if "cur_time" in snapshot:
        task_env.cur_time = float(snapshot["cur_time"])
    sim.forward()
    task_env._post_process()
    # Drop samples and sampling clocks from the future root / previous branch.
    # Reusing that cache makes a restored physics state appear to have a stale
    # or empty image, which is especially harmful for visual policies.
    if hasattr(task_env, "_obs_cache"):
        task_env._obs_cache = {}
    for observable in getattr(task_env, "_observables", {}).values():
        observable.reset()
    # A counterfactual branch restarts the episode clock from its snapshot.
    # robosuite keeps a monotonic timestep that is only reset by a full
    # reset().  Without this, the root episode plus every branch accumulate
    # on the same counter until it crosses horizon, at which point
    # step() raises ValueError: executing action in terminated episode
    # and crashes the env subprocess (surfacing as EOFError in the driver and
    # killing the Ray actors).
    task_env.timestep = 0
    task_env.done = False
    _make_render_context_current(env)
    observation = task_env._get_observations(force_update=True)
    _debug_worker_event(
        "restore ok "
        f"root_timestep={snapshot.get('root_timestep', 'unknown')} "
        f"state_size={state.size}"
    )
    return observation


def _worker(
    parent: connection.Connection,
    p: connection.Connection,
    env_fn_wrapper: CloudpickleWrapper,
    obs_bufs: Optional[Union[dict, tuple, ShArray]] = None,
) -> None:
    def _encode_obs(
        obs: Union[dict, tuple, np.ndarray], buffer: Union[dict, tuple, ShArray]
    ) -> None:
        if isinstance(obs, np.ndarray) and isinstance(buffer, ShArray):
            buffer.save(obs)
        elif isinstance(obs, tuple) and isinstance(buffer, tuple):
            for o, b in zip(obs, buffer):
                _encode_obs(o, b)
        elif isinstance(obs, dict) and isinstance(buffer, dict):
            for k in obs.keys():
                _encode_obs(obs[k], buffer[k])
        return None

    parent.close()
    _normalize_egl_device_id()
    env = env_fn_wrapper.data()
    _debug_worker_event(
        "initialized "
        f"egl_lock={_egl_lock_path() or 'disabled'}"
    )
    worker_step = 0
    try:
        while True:
            try:
                cmd, data = p.recv()
            except EOFError:  # the pipe has been closed
                p.close()
                break
            if cmd == "step":
                worker_step += 1
                if worker_step <= 3 or worker_step % 100 == 0:
                    _debug_worker_event(f"step begin index={worker_step}")
                with _egl_process_guard():
                    env_return = env.step(data)
                if worker_step <= 3 or worker_step % 100 == 0:
                    _debug_worker_event(f"step end index={worker_step}")
                current_libero_type = os.environ.get("LIBERO_TYPE", "standard").lower()
                if current_libero_type == "safe":
                    env_return = list(env_return)
                    info = dict(env_return[3])
                    info["_safelibero_contact_pairs"] = _get_contact_pairs(env)
                    env_return[3] = info
                    env_return = tuple(env_return)
                elif current_libero_type == "safety":
                    env_return = list(env_return)
                    info = dict(env_return[3])
                    info["_libero_safety_predicates"] = audit_predicate_instances(env)
                    env_return[3] = info
                    env_return = tuple(env_return)
                if obs_bufs is not None:
                    _encode_obs(env_return[0], obs_bufs)
                    env_return = (None, *env_return[1:])
                p.send(env_return)
            elif cmd == "reset":
                with _egl_process_guard():
                    retval = env.reset(**data)
                reset_returns_info = (
                    isinstance(retval, (tuple, list))
                    and len(retval) == 2
                    and isinstance(retval[1], dict)
                )
                if reset_returns_info:
                    obs, info = retval
                else:
                    obs = retval
                if obs_bufs is not None:
                    _encode_obs(obs, obs_bufs)
                    obs = None
                if reset_returns_info:
                    p.send((obs, info))
                else:
                    p.send(obs)
            elif cmd == "close":
                p.send(env.close())
                p.close()
                break
            elif cmd == "render":
                with _egl_process_guard():
                    p.send(env.render(**data) if hasattr(env, "render") else None)
            elif cmd == "seed":
                if hasattr(env, "seed"):
                    p.send(env.seed(data))
                else:
                    env.reset(seed=data)
                    p.send(None)
            elif cmd == "getattr":
                p.send(getattr(env, data) if hasattr(env, data) else None)
            elif cmd == "setattr":
                setattr(env.unwrapped, data["key"], data["value"])
            elif cmd == "check_success":
                p.send(env.check_success())
            elif cmd == "get_segmentation_of_interest":
                p.send(env.get_segmentation_of_interest(data))
            elif cmd == "get_sim_state":
                p.send(env.get_sim_state())
            elif cmd == "snapshot_simulator_state":
                p.send(_snapshot_simulator_state(env))
            elif cmd == "restore_simulator_state":
                with _egl_process_guard():
                    p.send(_restore_simulator_state(env, data))
            elif cmd == "set_init_state":
                with _egl_process_guard():
                    obs = env.set_init_state(data)
                p.send(obs)
            elif cmd == "reconfigure":
                env.close()
                seed = data.pop("seed")
                env = OffScreenRenderEnv(**data)
                env.seed(seed)
                p.send(None)
            elif cmd == "get_camera_meta":
                # Compute camera intrinsics/extrinsics and depth near/far
                # from the robosuite sim, which is only reachable inside the
                # worker subprocess.  Returns picklable lists/floats so the
                # driver can back-project pixels to world without GT poses.
                from robosuite.utils import camera_utils

                rob = getattr(env, "env", env)
                while hasattr(rob, "env"):
                    rob = rob.env
                sim = rob.sim
                cam = data.get("camera_name", "agentview")
                h = int(data.get("height", 256))
                w = int(data.get("width", 256))
                K = camera_utils.get_camera_intrinsic_matrix(sim, cam, h, w)
                E = camera_utils.get_camera_extrinsic_matrix(sim, cam)
                extent = float(sim.model.stat.extent)
                near = float(sim.model.vis.map.znear) * extent
                far = float(sim.model.vis.map.zfar) * extent
                p.send(
                    {
                        "camera_name": cam,
                        "height": h,
                        "width": w,
                        "intrinsic_K": K.tolist(),
                        "extrinsic_cam2world": E.tolist(),
                        "depth_near": near,
                        "depth_far": far,
                    }
                )
            elif cmd == "render_camera":
                rob = getattr(env, "env", env)
                while hasattr(rob, "env"):
                    rob = rob.env
                sim = rob.sim
                cam = data.get("camera_name", "agentview")
                h = int(data.get("height", 1024))
                w = int(data.get("width", 1024))
                depth = bool(data.get("depth", False))
                with _egl_process_guard():
                    p.send(
                        sim.render(
                            width=w,
                            height=h,
                            camera_name=cam,
                            depth=depth,
                        )
                    )
            elif cmd == "set_camera_rendering":
                _set_camera_rendering(env, data)
                p.send(None)
            else:
                p.close()
                raise NotImplementedError
    except KeyboardInterrupt:
        p.close()


class ReconfigureSubprocEnvWorker(SubprocEnvWorker):
    def __init__(self, env_fn: Callable[[], gym.Env], share_memory: bool = False):
        ctx = multiprocessing.get_context("spawn")
        self.parent_remote, self.child_remote = ctx.Pipe()
        self.share_memory = share_memory
        self.buffer: Optional[Union[dict, tuple, ShArray]] = None
        if self.share_memory:
            dummy = env_fn()
            obs_space = dummy.observation_space
            dummy.close()
            del dummy
            self.buffer = _setup_buf(obs_space)
        args = (
            self.parent_remote,
            self.child_remote,
            CloudpickleWrapper(env_fn),
            self.buffer,
        )
        self.process = ctx.Process(target=_worker, args=args, daemon=True)
        self.process.start()
        self.child_remote.close()
        EnvWorker.__init__(self, env_fn)

    def reconfigure_env_fn(self, env_fn_param):
        self.parent_remote.send(["reconfigure", env_fn_param])
        return self.parent_remote.recv()


class ReconfigureSubprocEnv(SubprocVectorEnv):
    def __init__(self, env_fns: list[Callable[[], gym.Env]], **kwargs: Any) -> None:
        def worker_fn(fn: Callable[[], gym.Env]) -> ReconfigureSubprocEnvWorker:
            return ReconfigureSubprocEnvWorker(fn, share_memory=False)

        BaseVectorEnv.__init__(self, env_fns, worker_fn, **kwargs)

    def reconfigure_env_fns(self, env_fns, id=None):
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)

        for j, i in enumerate(id):
            self.workers[i].reconfigure_env_fn(env_fns[j])

    def set_camera_rendering(self, enabled: bool, id=None):
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)
        for i in id:
            self.workers[i].parent_remote.send(["set_camera_rendering", enabled])
        for i in id:
            self.workers[i].parent_remote.recv()

    def snapshot_simulator_state(self, id=None):
        """Return counterfactual snapshots for selected subprocesses."""
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if self.is_async:
            self._assert_id(id)
        for i in id:
            self.workers[i].parent_remote.send(["snapshot_simulator_state", None])
        return [self.workers[i].parent_remote.recv() for i in id]

    def restore_simulator_state(self, snapshots, id=None):
        """Restore snapshots and return regenerated observations."""
        self._assert_is_not_closed()
        id = self._wrap_id(id)
        if len(snapshots) != len(id):
            raise ValueError("one simulator snapshot is required per environment id")
        if self.is_async:
            self._assert_id(id)
        for snapshot, env_id in zip(snapshots, id):
            self.workers[env_id].parent_remote.send(
                ["restore_simulator_state", snapshot]
            )
        return np.stack([self.workers[i].parent_remote.recv() for i in id])

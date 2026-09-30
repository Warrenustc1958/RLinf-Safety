from omegaconf import OmegaConf

from rlinf.envs.sim.libero.libero_env import _requires_libero_shadow_continuation


def test_shadow_collect_requires_simulator_continuation():
    cfg = OmegaConf.create(
        {"episode_auditor": {"enabled": True, "mode": "shadow_collect"}}
    )

    assert _requires_libero_shadow_continuation(cfg)


def test_official_eval_keeps_simulator_termination():
    cfg = OmegaConf.create(
        {"episode_auditor": {"enabled": True, "mode": "official_eval"}}
    )

    assert not _requires_libero_shadow_continuation(cfg)


def test_disabled_auditor_keeps_simulator_termination():
    cfg = OmegaConf.create(
        {"episode_auditor": {"enabled": False, "mode": "shadow_collect"}}
    )

    assert not _requires_libero_shadow_continuation(cfg)

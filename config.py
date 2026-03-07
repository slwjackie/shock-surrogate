from dataclasses import dataclass

@dataclass
class SimCfg:
    L: float = 20.0
    Nx: int = 256
    t_end: float = 1.2
    Nt_save: int = 100
    CFL: float = 0.45

    # Train (nominal) PDE params
    nu: float = 0.002
    k: float = 1.5
    E: float = 6.0

    # Proxy regime thresholds
    u_runup_threshold: float = 1.5
    # u_det_threshold: float = 1.7
    # u_nodet_threshold: float = 1.4
    
    # --- Fixed physical-ish regime thresholds (DO NOT auto-calibrate per run) ---
    # # shock strength proxy (max |du/dx| over x,t)
    # g_nodet_threshold: float = 142.0
    # g_det_threshold: float = 147.0

    # # propagation speed proxy (|front_speed|)
    # fs_nodet_threshold: float = 0.0560
    # fs_det_threshold: float = 0.0572
    
    
    g_mean: float = 0.0
    g_std: float = 1.0
    fs_mean: float = 0.0
    fs_std: float = 1.0

    det_radius_threshold: float = 3.5
    no_radius_threshold: float = 0.5
    grad_runup_mult: float = 2.0

@dataclass
class DataCfg:
    seed: int = 0
    n_train: int = 200
    n_val: int = 50
    n_test_profile_ood: int = 80
    n_test_mismatch_ood: int = 80

    min_other_each: int = 10 
    dTdx_min: float = -4.0
    dTdx_max: float =  4.0

    # Profile OOD: nonlinear temperature profile term b*(x/L)^2
    b_ood_min: float = -4.0
    b_ood_max: float = 4.0

    # Solver mismatch OOD parameters (different PDE coefficients)
    nu_ood: float = 0.02
    k_ood: float = 6.0
    E_ood: float = 10.0
    
    # per-case parameter variation to reduce peak_u saturation
    nu_scale_min: float = 0.5
    nu_scale_max: float = 2.0

    t_end_scale_min: float = 0.8
    t_end_scale_max: float = 1.2

@dataclass
class TrainCfg:
    epochs: int = 1200
    batch_cases: int = 8
    batch_times: int = 4
    d_model: int = 128
    nhead: int = 8
    nlayers: int = 4
    lr: float = 2e-4

    w_data: float = 1.0
    w_phys: float = 0.3
    w_causal: float = 0.2
    w_cls: float = 0.5

    causal_windows: int = 5
    gate_threshold: float = 0.02

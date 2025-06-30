import numpy as np

### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

# -------- GLOBAL STATE --------
nInst          = 50
currentPos     = np.zeros(nInst)
days_held      = np.zeros(nInst, dtype=int)      

# -------- TUNABLE PARAMETERS --------
MA_WIN         = 20
VOL_WIN        = 20
CORR_WIN       = 60
CORR_THRESH    = 0.40

Z_IN           = 1.20
Z_OUT          = 0.42
DISP_MIN       = 0.95
MIN_EXTREMES   = 3
HOLD_DAYS      = 3

USD_BASE       = 100       
USD_MAX        = 10_000     
MAX_GROSS      = 100_000     
TOP_K_PER_SIDE = 6          

_last_corr_day = -1
_pair_mask     = None


# ------------------------------------------------------------
def _make_pair_mask(ret_mat: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(ret_mat)
    mask = np.abs(corr) >= CORR_THRESH
    np.fill_diagonal(mask, False)
    return mask
# ------------------------------------------------------------
def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    global currentPos, days_held, _pair_mask, _last_corr_day

    nIns, nt = prcSoFar.shape
    if nt <= max(MA_WIN, VOL_WIN, CORR_WIN):
        return np.zeros(nIns)

    prices = prcSoFar[:, -1]
    ma20   = prcSoFar[:, -MA_WIN:].mean(axis=1)
    spread = prices / ma20 - 1.0

    # Volume-scaled spread
    rets_vol = np.diff(np.log(prcSoFar[:, -(VOL_WIN + 1):]), axis=1)
    vol      = rets_vol.std(axis=1, ddof=1)
    vol[vol == 0] = 1e-8
    spread_n = spread / vol

    zscores = (spread_n - spread_n.mean()) / spread_n.std(ddof=0)

    # Eligibility by correlation
    if nt != _last_corr_day:
        _last_corr_day = nt
        ret_corr = np.diff(np.log(prcSoFar[:, -(CORR_WIN + 1):]), axis=1)
        _pair_mask = _make_pair_mask(ret_corr)
    eligible = _pair_mask.any(axis=1)

    # Regime gates
    disp     = zscores.std(ddof=0)
    extremes = np.sum(np.abs(zscores) > Z_IN)
    allow_new = (disp >= DISP_MIN) and (extremes >= MIN_EXTREMES)

  # Existing positions
    signal = np.zeros(nIns)
    holding = currentPos != 0
    days_held[holding] += 1
    exit_ok = days_held >= HOLD_DAYS
    close_idx = holding & exit_ok & (np.abs(zscores) < Z_OUT)
    signal[close_idx] = 0
    signal[holding & ~close_idx] = np.sign(currentPos[holding & ~close_idx])

    # Trade opening
    if allow_new:
        long_cands  = np.where((zscores < -Z_IN) & eligible & ~holding)[0]
        short_cands = np.where((zscores >  Z_IN) & eligible & ~holding)[0]

        # Keep only the TOP_K_PER_SIDE most extreme on each side
        if long_cands.size  > TOP_K_PER_SIDE:
            long_cands  = long_cands[np.argsort(zscores[long_cands])][:TOP_K_PER_SIDE]
        if short_cands.size > TOP_K_PER_SIDE:
            short_cands = short_cands[np.argsort(-zscores[short_cands])][:TOP_K_PER_SIDE]

        signal[long_cands]  = +1
        signal[short_cands] = -1
        days_held[long_cands]  = 0
        days_held[short_cands] = 0  

    # Sizing
    # Concave (sqrt) scaling to avoid oversizing deep outliers
    edge     = np.clip(np.abs(zscores) - Z_IN, 0, None)
    scale    = np.sqrt(edge / max(Z_IN, 1e-9)) + 1      # 1 … √n
    dollars  = np.minimum(scale * USD_BASE, USD_MAX)

    targetPos = (dollars * signal / prices).astype(int)

    # Portfolio-level gross dollar cap
    gross = np.sum(np.abs(targetPos) * prices)
    if gross > MAX_GROSS and gross > 0:
        factor = MAX_GROSS / gross
        targetPos = (targetPos * factor).astype(int)

    # Update State
    currentPos = targetPos
    days_held[targetPos == 0] = 0
    return currentPos






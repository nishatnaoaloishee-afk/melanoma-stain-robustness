import numpy as np
import scipy.ndimage as ndi
from skimage import color, filters, morphology, measure
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, find_boundaries

# ---------------------------
# HYBRID A = rule-based proposal + RF refinement
# ---------------------------

def normalize_to_01(arr):
    arr = arr.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)

def hematoxylin_inv(img_u8):
    img_f = img_u8.astype(np.float32) / 255.0
    hed = color.rgb2hed(img_f)
    H = hed[:, :, 0]
    return 1.0 - normalize_to_01(H)

def compute_feature_maps(img_u8):
    """
    20×-friendly features for pixel RF.
    Keep interpretable and not too many.
    """
    Hinv = hematoxylin_inv(img_u8).astype(np.float32)
    f = {}
    f["Hinv"] = Hinv
    f["H_g07"] = ndi.gaussian_filter(Hinv, sigma=0.7).astype(np.float32)
    f["H_g12"] = ndi.gaussian_filter(Hinv, sigma=1.2).astype(np.float32)
    f["H_log07"] = (-ndi.gaussian_laplace(Hinv, sigma=0.7)).astype(np.float32)
    f["H_log14"] = (-ndi.gaussian_laplace(Hinv, sigma=1.4)).astype(np.float32)
    f["H_sobel"] = filters.sobel(Hinv).astype(np.float32)

    gray = color.rgb2gray(img_u8.astype(np.float32) / 255.0)
    f["gray"] = normalize_to_01(gray).astype(np.float32)
    return f

def stack_features_at_indices(fmaps, rr, cc):
    keys = list(fmaps.keys())
    X = np.stack([fmaps[k][rr, cc] for k in keys], axis=1).astype(np.float32)
    return X, keys

# ---------- Rule-based proposal (candidate nuclei region) ----------
def candidate_nuclei_mask(img_u8, tissue_mask,
                          gauss_sigma=1.0, sauvola_window=25,
                          min_size=5, opening_radius=1):
    """
    High-recall candidate region: Hinv + Sauvola.
    """
    tissue = tissue_mask.astype(bool)
    Hinv = hematoxylin_inv(img_u8)
    Hs = ndi.gaussian_filter(Hinv, sigma=gauss_sigma) * tissue
    thr = filters.threshold_sauvola(Hs, window_size=sauvola_window)
    cand = (Hs > thr) & tissue
    cand = morphology.remove_small_objects(cand, min_size=min_size)
    cand = morphology.binary_opening(cand, morphology.disk(opening_radius))
    return cand.astype(bool)

def dilate_mask(mask, radius):
    if radius <= 0:
        return mask.astype(bool)
    return morphology.binary_dilation(mask.astype(bool), morphology.disk(radius))

# ---------- Sampling (positives from GT; negatives = hard+easy) ----------
def sample_pixels_hybridA(gt_bin, tissue_mask, cand_mask,
                          n_pos, n_neg, seed=13,
                          hard_neg_ratio=0.6, hard_band_radius=3):
    """
    Positives: GT nuclei pixels within tissue.
    Negatives: mixture of hard negatives near candidate and easy negatives far away.
    """
    rng = np.random.default_rng(seed)
    tissue = tissue_mask.astype(bool)
    gt = gt_bin.astype(bool)
    cand = cand_mask.astype(bool)

    pos_pool = np.where(gt & tissue)
    if len(pos_pool[0]) == 0:
        return None

    cand_d = dilate_mask(cand, hard_band_radius)
    hard_pool = np.where((cand_d & tissue) & (~gt))
    easy_pool = np.where((tissue & (~cand_d)) & (~gt))

    # sample positives
    pos_take = min(n_pos, len(pos_pool[0]))
    pos_idx = rng.choice(len(pos_pool[0]), size=pos_take, replace=False)
    pos_rr = pos_pool[0][pos_idx]
    pos_cc = pos_pool[1][pos_idx]

    # split negatives
    hard_take = int(n_neg * hard_neg_ratio)
    easy_take = n_neg - hard_take

    hard_take = min(hard_take, len(hard_pool[0])) if len(hard_pool[0]) else 0
    easy_take = min(easy_take, len(easy_pool[0])) if len(easy_pool[0]) else 0

    # top up if needed
    total = hard_take + easy_take
    if total < n_neg:
        need = n_neg - total
        if len(hard_pool[0]) > hard_take:
            add = min(need, len(hard_pool[0]) - hard_take)
            hard_take += add
            need -= add
        if need > 0 and len(easy_pool[0]) > easy_take:
            add = min(need, len(easy_pool[0]) - easy_take)
            easy_take += add

    neg_rr, neg_cc = [], []
    if hard_take > 0:
        idx = rng.choice(len(hard_pool[0]), size=hard_take, replace=False)
        neg_rr.append(hard_pool[0][idx]); neg_cc.append(hard_pool[1][idx])
    if easy_take > 0:
        idx = rng.choice(len(easy_pool[0]), size=easy_take, replace=False)
        neg_rr.append(easy_pool[0][idx]); neg_cc.append(easy_pool[1][idx])

    if len(neg_rr) == 0:
        return None

    neg_rr = np.concatenate(neg_rr)
    neg_cc = np.concatenate(neg_cc)

    rr = np.concatenate([pos_rr, neg_rr])
    cc = np.concatenate([pos_cc, neg_cc])
    y = np.concatenate([
        np.ones(len(pos_rr), dtype=np.uint8),
        np.zeros(len(neg_rr), dtype=np.uint8)
    ])
    return rr, cc, y

# ---------- Gated inference ----------
def predict_prob_gated(rf, fmaps, tissue_mask, cand_mask, gate_radius=2):
    """
    Predict only within gate = dilate(candidate) ∩ tissue.
    """
    tissue = tissue_mask.astype(bool)
    gate = dilate_mask(cand_mask, gate_radius) & tissue

    prob = np.zeros(tissue.shape, dtype=np.float32)
    rr, cc = np.where(gate)
    if len(rr) == 0:
        return prob, gate

    X, _ = stack_features_at_indices(fmaps, rr, cc)
    prob_vals = rf.predict_proba(X)[:, 1].astype(np.float32)
    prob[rr, cc] = prob_vals
    return prob, gate

# ---------- Prob -> instances (watershed) ----------
def prob_to_instances(prob, tissue_mask,
                      prob_thresh=0.55,
                      min_area=5, max_area=1200,
                      min_peak_distance=2, dist_sigma=1.0):
    tissue = tissue_mask.astype(bool)
    binary = (prob >= prob_thresh) & tissue
    binary = morphology.remove_small_objects(binary, min_size=min_area)
    binary = morphology.binary_opening(binary, morphology.disk(1))

    if binary.sum() == 0:
        return np.zeros_like(prob, dtype=np.int32), binary

    dist = ndi.distance_transform_edt(binary)
    dist_s = ndi.gaussian_filter(dist, sigma=dist_sigma)

    coords = peak_local_max(dist_s, labels=binary,
                            min_distance=min_peak_distance,
                            footprint=np.ones((3,3), dtype=bool))
    markers = np.zeros_like(prob, dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    if markers.max() == 0:
        markers = measure.label(binary).astype(np.int32)

    labels = watershed(-dist_s, markers, mask=binary).astype(np.int32)

    # area filter
    out = np.zeros_like(labels, dtype=np.int32)
    cur = 1
    for reg in measure.regionprops(labels):
        if not (min_area <= reg.area <= max_area):
            continue
        rr, cc = zip(*reg.coords)
        out[rr, cc] = cur
        cur += 1

    pred_bin = out > 0
    return out, pred_bin

# ---------- Metrics + overlay ----------
def compute_pixel_metrics(pred_bin, gt_bin, tissue_mask):
    tissue = tissue_mask.astype(bool)
    p = pred_bin[tissue].astype(bool)
    g = gt_bin[tissue].astype(bool)

    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    tn = int(np.logical_and(~p, ~g).sum())

    eps = 1e-8
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    dice      = (2 * tp) / (2 * tp + fp + fn + eps)
    iou       = tp / (tp + fp + fn + eps)
    acc       = (tp + tn) / (tp + tn + fp + fn + eps)
    spec      = tn / (tn + fp + eps)

    return dict(tp=tp, fp=fp, fn=fn, tn=tn,
                precision=precision, recall=recall,
                f1=f1, dice=dice, iou=iou,
                accuracy=acc, specificity=spec)

def make_overlay(img_u8, gt_bin, pred_bin):
    gt_b = find_boundaries(gt_bin.astype(bool), mode="outer")
    pr_b = find_boundaries(pred_bin.astype(bool), mode="outer")
    vis = img_u8.copy()
    vis[gt_b] = np.array([0,255,0], dtype=np.uint8)  # GT green
    vis[pr_b] = np.array([255,0,0], dtype=np.uint8)  # Pred red
    return vis


def make_overlay_pred_only(img_u8, pred_bin):
    pr_b = find_boundaries(pred_bin.astype(bool), mode="outer")
    vis = img_u8.copy()
    vis[pr_b] = np.array([255, 0, 0], dtype=np.uint8)  # Pred red
    return vis


def make_overlay_pred_filled(img_u8, pred_bin, alpha=0.35, draw_boundary=True):
    """
    img_u8: HxWx3 uint8
    pred_bin: HxW bool
    alpha: fill transparency (0..1). Higher = stronger fill.
    """
    vis = img_u8.copy().astype(np.float32)

    mask = pred_bin.astype(bool)
    if mask.any():
        fill_color = np.array([255, 0, 0], dtype=np.float32)  # red
        vis[mask] = (1 - alpha) * vis[mask] + alpha * fill_color

    if draw_boundary:
        b = find_boundaries(mask, mode="outer")
        vis[b] = np.array([255, 0, 0], dtype=np.float32)

    return np.clip(vis, 0, 255).astype(np.uint8)

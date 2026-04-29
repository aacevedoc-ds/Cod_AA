#!/usr/bin/env python3
"""
analytics.py — Motor de clustering semantico para Codificador AA.
Recibe JSON por stdin: { "action": "...", "data": {...} }
Devuelve JSON por stdout.
"""
import sys
import json
import numpy as np


# ─── UTILIDADES ───────────────────────────────────────────────────────────────

def oversample_by_weight(matrix, weights):
    w = np.array(weights, dtype=float)
    w_min = w[w > 0].min() if (w > 0).any() else 1.0
    reps = np.clip(np.round(w / w_min).astype(int), 1, 10)
    indices = np.repeat(np.arange(len(matrix)), reps)
    return matrix[indices], indices


def _elbow_k(k_range, inertias):
    k_range = list(k_range)
    if len(inertias) < 3:
        return k_range[0]
    diffs = [inertias[i] - inertias[i + 1] for i in range(len(inertias) - 1)]
    max_diff = max(diffs) if diffs else 0
    if max_diff == 0 or (diffs[0] - diffs[-1]) / max(abs(diffs[0]), 1e-9) < 0.05:
        return 3
    return k_range[diffs.index(max(diffs))]


def _choose_k(k_range, inertias, silhouettes, n):
    """Selección híbrida: silhouette + restricción de balance.

    Para evitar k=2 con cluster dominante (94%), preferimos:
    - el k con mejor silhouette si k>=3 está cerca (<=10%) del top
    - mínimo k=3 cuando n>=50
    """
    k_range = list(k_range)
    if not silhouettes:
        return _elbow_k(k_range, inertias)
    best_idx = max(range(len(silhouettes)), key=lambda i: silhouettes[i])
    best_sil = silhouettes[best_idx]
    best_k = k_range[best_idx]
    # Si n grande y best_k=2, intentar k>=3 con sil >= 90% del best
    if n >= 50 and best_k == 2:
        for i, k in enumerate(k_range):
            if k >= 3 and silhouettes[i] >= best_sil * 0.90:
                return k
    return best_k


# ─── ACCIONES ─────────────────────────────────────────────────────────────────

def tfidf_by_cluster(data):
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = data["texts"]
    labels = data["labels"]
    clusters = {}
    unique = sorted(set(l for l in labels if l >= 0))
    for cid in unique:
        idxs = [i for i, l in enumerate(labels) if l == cid]
        corpus = [texts[i] for i in idxs]
        if len(corpus) < 2:
            clusters[str(cid)] = []
            continue
        vec = TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=1, stop_words=None)
        try:
            mat = vec.fit_transform(corpus)
            scores = np.asarray(mat.mean(axis=0)).flatten()
            top_idx = scores.argsort()[-10:][::-1]
            clusters[str(cid)] = [vec.get_feature_names_out()[i] for i in top_idx]
        except Exception:
            clusters[str(cid)] = []
    return clusters


def cluster_text(data):
    import umap as umap_lib
    import hdbscan as hdb
    from sklearn.metrics import silhouette_score
    emb = np.array(data["embeddings"])
    weights = data.get("weights", [1.0] * len(emb))
    n = len(emb)
    if n < 4:
        raise ValueError("Se necesitan al menos 4 respondentes para clustering")
    # min_cluster_size proporcional al N: queremos pocos clusters grandes (~5-10% c/u)
    # default: max(8, N/15) → con N=300 da mcs=20; con N=100 da mcs=8
    mcs_default = max(8, n // 15)
    mcs = data.get("min_cluster_size", mcs_default)
    ms = data.get("min_samples", max(3, mcs // 4))
    n_neighbors_2d = min(15, n - 1)
    reducer_2d = umap_lib.UMAP(n_components=2, n_neighbors=n_neighbors_2d, min_dist=0.1, metric='cosine', random_state=42)
    coords_2d = reducer_2d.fit_transform(emb)
    n_nd = min(10, n - 2)
    n_neighbors_nd = min(15, n - 1)
    reducer_nd = umap_lib.UMAP(n_components=n_nd, n_neighbors=n_neighbors_nd, metric='cosine', random_state=42)
    emb_nd = reducer_nd.fit_transform(emb)
    emb_os, orig_idx = oversample_by_weight(emb_nd, weights)
    mcs_eff = max(5, min(mcs, len(emb_os) // 4))
    ms_eff = max(1, min(ms, mcs_eff))
    # cluster_selection_epsilon fusiona clusters cuya distancia sea menor a este umbral
    # cluster_selection_method='eom' (default) es más conservador que 'leaf'
    clusterer = hdb.HDBSCAN(
        min_cluster_size=mcs_eff,
        min_samples=ms_eff,
        cluster_selection_method='eom',
        cluster_selection_epsilon=0.5,
    )
    labels_os = clusterer.fit_predict(emb_os)
    labels = np.full(n, -1, dtype=int)
    for i in range(n):
        mask = (orig_idx == i)
        if mask.any():
            votes = labels_os[mask]
            unique_v, counts = np.unique(votes, return_counts=True)
            labels[i] = unique_v[counts.argmax()]
    for cid in list(set(labels)):
        if cid < 0:
            continue
        if (labels == cid).sum() < mcs_eff:
            others = [c for c in set(labels) if c >= 0 and c != cid]
            if others:
                centroids = {c: emb_nd[labels == c].mean(axis=0) for c in others}
                my_c = emb_nd[labels == cid].mean(axis=0)
                nearest = min(others, key=lambda c: np.linalg.norm(my_c - centroids[c]))
                labels[labels == cid] = nearest
    # Normalizar labels a 0..K-1 contiguos (post-fusión puede dejar gaps tipo {1,2,4})
    unique_pos = sorted([c for c in set(labels.tolist()) if c >= 0])
    remap = {old: new for new, old in enumerate(unique_pos)}
    labels = np.array([remap[c] if c >= 0 else -1 for c in labels.tolist()], dtype=int)
    valid_mask = labels >= 0
    sil = 0.0
    if valid_mask.sum() > 1 and len(set(labels[valid_mask])) >= 2:
        sil = float(silhouette_score(emb_nd[valid_mask], labels[valid_mask]))
    return {
        "labels_2d": coords_2d.tolist(),
        "cluster_ids": labels.tolist(),
        "silhouette": round(sil, 3),
        "n_outliers": int((labels < 0).sum())
    }


def cluster_codes(data):
    import umap as umap_lib
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    mat = np.array(data["matrix"], dtype=float)
    weights = data.get("weights", [1.0] * len(mat))
    # Filtrar filas todo-cero (la métrica binaria es indefinida entre vectores nulos)
    row_sums = mat.sum(axis=1)
    valid_mask = row_sums > 0
    n_zero = int((~valid_mask).sum())
    if valid_mask.sum() < 4:
        raise ValueError(f"Solo {int(valid_mask.sum())} respondentes con códigos activos. Pipeline B requiere al menos 4 con códigos.")
    mat_v = mat[valid_mask]
    weights_v = np.array(weights)[valid_mask].tolist()
    n = len(mat_v)
    if n < 4:
        raise ValueError("Se necesitan al menos 4 respondentes para clustering")

    # ── Excluir códigos hiper-dominantes (>60% prevalencia) y vacíos.
    # Razón: si un código aparece en casi todos, no aporta poder discriminante
    # y satura la métrica binaria, colapsando todo en un mismo cluster.
    n_codes = mat_v.shape[1]
    prevalence = mat_v.mean(axis=0)  # proporción no ponderada
    keep_mask = (prevalence > 0.0) & (prevalence < 0.6)
    excluded_codes = [int(i) for i in np.where(~keep_mask)[0]]
    if keep_mask.sum() < 2:
        # Si filtrar deja <2 códigos, conservar todos como fallback
        keep_mask = np.ones(n_codes, dtype=bool)
        excluded_codes = []
    mat_clust = mat_v[:, keep_mask]

    # Tras filtrar, algunas filas pueden quedar todo-cero otra vez (solo tenían el dominante)
    inner_valid = mat_clust.sum(axis=1) > 0
    if inner_valid.sum() < 4:
        # No hay suficiente variabilidad — devolver un único cluster
        labels_full = np.where(valid_mask, 0, -1)
        coords_full = np.zeros((len(mat), 2), dtype=float)
        return {
            "labels_2d": coords_full.tolist(),
            "cluster_ids": labels_full.tolist(),
            "k_chosen": 1,
            "inertias": [],
            "silhouettes": [],
            "n_zero_rows": n_zero,
            "warning": "Sin variabilidad suficiente entre códigos: la mayoría de respuestas comparten el mismo código dominante.",
            "excluded_codes": excluded_codes,
        }

    # Solo clusterizamos sobre filas con variabilidad post-filtro
    mat_for_umap = mat_clust[inner_valid]
    weights_inner = np.array(weights_v)[inner_valid].tolist()
    n_inner = len(mat_for_umap)

    # min_dist=0.3 evita que respondentes idénticos colapsen al mismo punto
    n_neighbors = min(15, n_inner - 1)
    reducer_2d = umap_lib.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.3, metric='hamming', random_state=42)
    coords_2d_inner = reducer_2d.fit_transform(mat_for_umap)
    n_nd = min(5, n_inner - 2)
    reducer_nd = umap_lib.UMAP(n_components=n_nd, n_neighbors=n_neighbors, metric='hamming', random_state=42)
    mat_nd = reducer_nd.fit_transform(mat_for_umap)
    weights_v = weights_inner
    n = n_inner
    # Reconstruir coords_2d y mat_v alineados al espacio post-filtro
    coords_2d = np.zeros((len(mat_v), 2), dtype=float)
    coords_2d[inner_valid] = coords_2d_inner
    mat_os, orig_idx = oversample_by_weight(mat_nd, weights_v)
    k_max = min(9, n_inner // 2)
    k_range = list(range(2, max(3, k_max)))
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbs = km.fit_predict(mat_os)
        inertias.append(float(km.inertia_))
        sil = float(silhouette_score(mat_os, lbs)) if k >= 2 else 0.0
        silhouettes.append(round(sil, 3))

    # Selección robusta: probar k crecientes, exigir balance (cluster mayor < 70%)
    k_chosen = _choose_k_balanced(k_range, inertias, silhouettes, mat_os, n_inner)

    km_final = KMeans(n_clusters=k_chosen, random_state=42, n_init=10)
    labels_os = km_final.fit_predict(mat_os)
    labels_inner = np.zeros(n_inner, dtype=int)
    for i in range(n_inner):
        mask = (orig_idx == i)
        if mask.any():
            votes = labels_os[mask]
            labels_inner[i] = int(np.bincount(votes).argmax())

    # Reconstruir labels_v (tamaño = filas válidas tras filtro inicial)
    # Las filas con SOLO código dominante (excluidas del clustering interno)
    # se les asigna un cluster aparte (k_chosen, etiqueta extra)
    n_v = len(mat_v)
    labels_v = np.full(n_v, k_chosen, dtype=int)  # cluster "dominante"
    labels_v[inner_valid] = labels_inner

    # Si hubo filas con solo dominante, agregar 1 cluster más
    has_dom_cluster = bool((~inner_valid).any())
    k_final = k_chosen + (1 if has_dom_cluster else 0)

    # Expandir a tamaño original; filas todo-cero originales quedan como -1
    n_full = len(mat)
    labels_full = np.full(n_full, -1, dtype=int)
    coords_full = np.zeros((n_full, 2), dtype=float)
    valid_indices = np.where(valid_mask)[0]
    labels_full[valid_indices] = labels_v
    coords_full[valid_indices] = coords_2d

    # Si el cluster dominante no se usó, restaurar
    if not has_dom_cluster:
        # Asegurar que las etiquetas estén en [0, k_chosen-1]
        pass

    # Normalizar labels a 0..K-1 contiguos (puede haber gaps tras fusiones/dominante)
    unique_pos = sorted([c for c in set(labels_full.tolist()) if c >= 0])
    remap = {old: new for new, old in enumerate(unique_pos)}
    labels_full = np.array([remap[c] if c >= 0 else -1 for c in labels_full.tolist()], dtype=int)

    return {
        "labels_2d": coords_full.tolist(),
        "cluster_ids": labels_full.tolist(),
        "k_chosen": k_final,
        "inertias": inertias,
        "silhouettes": silhouettes,
        "n_zero_rows": n_zero,
        "excluded_codes": excluded_codes,
        "n_only_dominant": int((~inner_valid).sum()),
    }


def _choose_k_balanced(k_range, inertias, silhouettes, mat_os, n_inner):
    """Elige k preferenciando balance: que el cluster mayor no exceda el 70% de la muestra.
    Si todos los k tienen cluster dominante, devuelve el de mayor silhouette.
    """
    from sklearn.cluster import KMeans
    if not k_range:
        return 2
    candidates = []
    for ki, k in enumerate(k_range):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbs = km.fit_predict(mat_os)
        _, counts = np.unique(lbs, return_counts=True)
        max_share = counts.max() / counts.sum()
        candidates.append((k, silhouettes[ki] if ki < len(silhouettes) else 0.0, float(max_share)))
    # Filtrar k con balance aceptable (cluster mayor < 70%)
    balanced = [c for c in candidates if c[2] < 0.70]
    if balanced:
        # Entre los balanceados, el de mayor silhouette
        best = max(balanced, key=lambda c: c[1])
        return best[0]
    # Si ninguno es balanceado, el de mejor silhouette
    best = max(candidates, key=lambda c: c[1])
    return best[0]


def concordance(data):
    from sklearn.metrics import adjusted_rand_score
    la = np.array(data["labels_a"])
    lb = np.array(data["labels_b"])
    mask = la >= 0
    la_valid = la[mask]
    lb_valid = lb[mask]
    if len(la_valid) < 4:
        return {"ari": 0.0, "n_shared": int(mask.sum()), "interpretation": "nula"}
    ari = float(adjusted_rand_score(la_valid, lb_valid))
    if ari >= 0.7:
        interp = "alta"
    elif ari >= 0.4:
        interp = "moderada"
    elif ari >= 0.1:
        interp = "baja"
    else:
        interp = "nula"
    return {"ari": round(ari, 3), "n_shared": int(mask.sum()), "interpretation": interp}


def chi2_codes(data):
    from scipy.stats import chi2_contingency
    mat = np.array(data["matrix"], dtype=float)
    code_names = data["code_names"]
    factors = data["factors"]
    weights = np.array(data.get("weights", [1.0] * len(mat)))
    results = []
    for ci, cname in enumerate(code_names):
        code_col = mat[:, ci]
        for fdef in factors:
            fname = fdef["name"]
            cats = fdef["categories"]
            row_vals = fdef["row_values"]
            table = np.zeros((2, len(cats)))
            freq_by_cat = {}
            for ki, cat in enumerate(cats):
                mask = np.array([str(rv) == str(cat) for rv in row_vals])
                w_cat = weights[mask]
                code_cat = code_col[mask]
                w_present = float((code_cat * w_cat).sum())
                w_absent = float(((1 - code_cat) * w_cat).sum())
                table[0, ki] = w_present
                table[1, ki] = w_absent
                total_cat = float(w_cat.sum())
                freq_by_cat[cat] = round((w_present / total_cat * 100) if total_cat > 0 else 0.0, 1)
            n_valid = int((weights > 0).sum())
            try:
                if table.sum() == 0 or table.min() < 0:
                    raise ValueError("tabla vacia")
                chi2, p, dof, expected = chi2_contingency(table)
                chi2_val, p_val = float(chi2), float(p)
            except Exception:
                chi2_val, p_val = None, None
            results.append({
                "code": cname, "factor": fname,
                "chi2": round(chi2_val, 3) if chi2_val is not None else None,
                "p_value": round(p_val, 4) if p_val is not None else None,
                "significant": bool(p_val < 0.05) if p_val is not None else False,
                "n_valid": n_valid, "freq_by_cat": freq_by_cat,
            })
    return {"results": results}


def logistic_codes(data):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    mat = np.array(data["matrix"], dtype=float)
    code_names = data["code_names"]
    factors = data["factors"]
    weights = np.array(data.get("weights", [1.0] * len(mat)))
    n_boot = int(data.get("n_boot", 200))
    results = []
    for fdef in factors:
        fname = fdef["name"]
        cats = fdef["categories"]
        row_vals = np.array(fdef["row_values"])
        for cat in cats:
            y = (row_vals == str(cat)).astype(int)
            n_target = int(y.sum())
            n_other = int((1 - y).sum())
            if n_target < 5 or n_other < 5:
                continue
            try:
                lr = LogisticRegression(max_iter=500, C=1.0, class_weight="balanced", solver="lbfgs")
                lr.fit(mat, y, sample_weight=weights)
                coefs = lr.coef_[0]
                odds_ratios = np.exp(coefs)
                proba = lr.predict_proba(mat)[:, 1]
                auc = float(roc_auc_score(y, proba, sample_weight=weights))

                # P-value por bootstrapping: % de veces que el coef bootstrap cruza cero
                n = len(y)
                rng = np.random.default_rng(42)
                boot_coefs = np.zeros((n_boot, mat.shape[1]))
                ok = 0
                for b in range(n_boot):
                    idx = rng.choice(n, size=n, replace=True)
                    # Saltar bootstrap degenerado (todo y igual)
                    if len(np.unique(y[idx])) < 2:
                        continue
                    try:
                        lr_b = LogisticRegression(max_iter=300, C=1.0, class_weight="balanced", solver="lbfgs")
                        lr_b.fit(mat[idx], y[idx], sample_weight=weights[idx])
                        boot_coefs[ok] = lr_b.coef_[0]
                        ok += 1
                    except Exception:
                        continue
                boot_coefs = boot_coefs[:ok] if ok > 0 else None

                # P-valor bilateral: 2 * min(prop>0, prop<0)
                p_values = np.ones(mat.shape[1])
                if boot_coefs is not None and len(boot_coefs) >= 30:
                    for i in range(mat.shape[1]):
                        col = boot_coefs[:, i]
                        prop_pos = float((col > 0).sum()) / len(col)
                        prop_neg = float((col < 0).sum()) / len(col)
                        p_values[i] = 2 * min(prop_pos, prop_neg)

                order = np.argsort(np.abs(coefs))[::-1]
                codes_result = [
                    {"code": code_names[i], "coef": round(float(coefs[i]), 4),
                     "odds_ratio": round(float(odds_ratios[i]), 3),
                     "p_value": round(float(p_values[i]), 4),
                     "significant": bool(p_values[i] < 0.05),
                     "direction": "aumenta" if coefs[i] > 0 else "disminuye"}
                    for i in order
                ]
                results.append({"factor": fname, "target_category": cat,
                                 "n_target": n_target, "n_other": n_other,
                                 "auc": round(auc, 3),
                                 "n_boot": int(ok if boot_coefs is not None else 0),
                                 "codes": codes_result})
            except Exception as e:
                results.append({"factor": fname, "target_category": cat,
                                 "error": str(e), "codes": []})
    return {"results": results}


# ─── DISPATCHER ───────────────────────────────────────────────────────────────

def handle(action, data):
    if action == "ping":
        return {"ok": True}
    if action == "tfidf":
        return tfidf_by_cluster(data)
    if action == "cluster_text":
        return cluster_text(data)
    if action == "cluster_codes":
        return cluster_codes(data)
    if action == "concordance":
        return concordance(data)
    if action == "chi2_codes":
        return chi2_codes(data)
    if action == "logistic_codes":
        return logistic_codes(data)
    raise ValueError(f"Accion desconocida: {action}")


if __name__ == "__main__":
    payload = json.loads(sys.stdin.read())
    try:
        result = handle(payload["action"], payload.get("data", {}))
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   КЕЙС-СТАДІ — Розділ 3 Дипломної роботи                                     ║
║   Базовий кейс-стаді для "L1": 1.0, "L2": 0.8. Dataset_size = 25000, +       ║
║   + Серія запусків: аналіз чутливості data-driven підходу                    ║
║   Рівень 1: варіація геометрії маніпулятора (4 конфігурації)                 ║
║   Рівень 2: варіація обсягу датасету (5 значень N)                           ║
║                                                                              ║
║   Результати: CSV-файли + зведені графіки + базовий рисунок (конф. A)        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import csv, os, time, warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ════════════════════════════════════════════════════════════════════════════
# БЛОК 0 — СЛОВНИК КОНФІГУРАЦІЙ ЕКСПЕРИМЕНТУ
# ════════════════════════════════════════════════════════════════════════════

GEOMETRY_CONFIGS = [
    {"id": "A", "L1": 1.0, "L2": 0.8, "label": "A: L1=1.0, L2=0.8 (асиметрична, базова)"},
    {"id": "B", "L1": 1.0, "L2": 1.0, "label": "B: L1=1.0, L2=1.0 (симетрична)"},
    {"id": "C", "L1": 0.8, "L2": 1.0, "label": "C: L1=0.8, L2=1.0 (інверсна)"},
    {"id": "D", "L1": 0.5, "L2": 0.5, "label": "D: L1=0.5, L2=0.5 (компактна)"},
]

DATASET_SIZES = [1_000, 5_000, 10_000, 25_000, 50_000]

MLP_CONFIG = {
    "hidden_layer_sizes":  (256, 256, 128),
    "activation":          "relu",
    "solver":              "adam",
    "learning_rate_init":  1e-3,
    "max_iter":            500,
    "early_stopping":      True,
    "validation_fraction": 0.15,
    "n_iter_no_change":    20,
    "tol":                 1e-6,
    "random_state":        42,
}

TRAJ_RADIUS_FRACTION = 0.5
N_TRAJ_POINTS        = 100
N_FIXED_GEOM         = 25_000
L1_BASE, L2_BASE     = 1.0, 0.8
OUTPUT_DIR           = "."
CLR_GEOM = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']

# ════════════════════════════════════════════════════════════════════════════
# БЛОК 1 — АНАЛІТИЧНА КІНЕМАТИКА (model-based baseline)
# ════════════════════════════════════════════════════════════════════════════

def fk(q1, q2, L1, L2):
    x = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
    y = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)
    return x, y

def ik_analytical(x, y, L1, L2, elbow='up'):
    D = (x**2 + y**2 - L1**2 - L2**2) / (2 * L1 * L2)
    if abs(D) > 1.0:
        return None
    sign = -1 if elbow == 'up' else 1
    q2 = np.arctan2(sign * np.sqrt(max(0.0, 1 - D**2)), D)
    q1 = np.arctan2(y, x) - np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
    return float(q1), float(q2)

# ════════════════════════════════════════════════════════════════════════════
# БЛОК 2 — ГЕНЕРАЦІЯ ДАТАСЕТУ
# ════════════════════════════════════════════════════════════════════════════

def generate_dataset(n, L1, L2):
    """
    Генерація одногілкового (elbow-up) датасету для заданої геометрії.
    Цільові точки рівномірно семплюються у полярних координатах
    у межах кільцеподібного робочого простору маніпулятора.
    Вхід мережі X: (x_ee, y_ee); Вихід мережі Y: (q1, q2).
    """
    r_min = abs(L1 - L2) + 0.02
    r_max = L1 + L2 - 0.02
    xy_list, q_list = [], []
    attempts = 0
    while len(xy_list) < n and attempts < n * 6:
        r   = np.sqrt(np.random.uniform(r_min**2, r_max**2))
        ang = np.random.uniform(-np.pi, np.pi)
        xt, yt = r * np.cos(ang), r * np.sin(ang)
        sol = ik_analytical(xt, yt, L1, L2, elbow='up')
        if sol is not None:
            xy_list.append([xt, yt])
            q_list.append(list(sol))
        attempts += 1
    return np.array(xy_list), np.array(q_list)

# ════════════════════════════════════════════════════════════════════════════
# БЛОК 3 — ОДИНОЧНИЙ ЗАПУСК
# ════════════════════════════════════════════════════════════════════════════

def run_single(L1, L2, n_samples, collect_baseline_data=False):
    """
    collect_baseline_data=True — додатково повертає дані для побудови
    6-панельного базового рисунку (рис. 3.1):
      _XY_test_raw   : координати тестових точок (м)
      _Q_pred_deg    : передбачені кути (градуси)
      _Q_true_deg    : істинні кути (градуси)
      _pos_err_mb    : позиційні похибки аналітичного методу (мм)
      _traj_x/_y     : задана кругова траєкторія
      _traj_x_nn/_y_nn : траєкторія відновлена з MLP
      _traj_x_mb/_y_mb : траєкторія відновлена аналітично
    """
    XY_raw, Q_raw = generate_dataset(n_samples, L1, L2)
    actual_n = len(XY_raw)

    scX = MinMaxScaler((-1, 1)); scQ = MinMaxScaler((-1, 1))
    XY  = scX.fit_transform(XY_raw)
    Q   = scQ.fit_transform(Q_raw)

    X_tr, X_te, Q_tr, Q_te = train_test_split(XY, Q, test_size=0.15,
                                               random_state=42)
    mlp = MLPRegressor(**MLP_CONFIG)
    t0  = time.time()
    mlp.fit(X_tr, Q_tr)
    elapsed = time.time() - t0

    Q_pred_norm = mlp.predict(X_te)
    Q_pred_rad  = scQ.inverse_transform(Q_pred_norm)
    Q_true_rad  = scQ.inverse_transform(Q_te)
    XY_test     = scX.inverse_transform(X_te)

    pos_err_nn, pos_err_mb, q_err = [], [], []
    for i in range(len(XY_test)):
        xt, yt = XY_test[i]
        xp, yp = fk(Q_pred_rad[i, 0], Q_pred_rad[i, 1], L1, L2)
        pos_err_nn.append(np.hypot(xp - xt, yp - yt))
        sol = ik_analytical(xt, yt, L1, L2)
        if sol:
            xm, ym = fk(sol[0], sol[1], L1, L2)
            pos_err_mb.append(np.hypot(xm - xt, ym - yt))
        dq = np.abs(Q_pred_rad[i] - Q_true_rad[i])
        dq = np.minimum(dq, 2*np.pi - dq)
        q_err.append(np.degrees(dq))

    pos_err_nn = np.array(pos_err_nn) * 1000
    pos_err_mb = np.array(pos_err_mb) * 1000
    q_err      = np.array(q_err)

    r_traj   = TRAJ_RADIUS_FRACTION * (L1 + L2)
    theta_tr = np.linspace(0, 2*np.pi, N_TRAJ_POINTS, endpoint=False)
    traj_x   = r_traj * np.cos(theta_tr)
    traj_y   = r_traj * np.sin(theta_tr)
    traj_err_nn  = []
    traj_x_nn, traj_y_nn = [], []
    traj_x_mb, traj_y_mb = [], []
    for tx, ty in zip(traj_x, traj_y):
        # MLP
        xy_n = scX.transform([[tx, ty]])
        qp   = scQ.inverse_transform(mlp.predict(xy_n))[0]
        xn, yn = fk(qp[0], qp[1], L1, L2)
        traj_err_nn.append(np.hypot(xn - tx, yn - ty) * 1000)
        traj_x_nn.append(xn); traj_y_nn.append(yn)
        # Аналітичний
        sol_t = ik_analytical(tx, ty, L1, L2)
        if sol_t:
            xm, ym = fk(sol_t[0], sol_t[1], L1, L2)
        else:
            xm, ym = tx, ty
        traj_x_mb.append(xm); traj_y_mb.append(ym)

    result = {
        "L1": L1, "L2": L2, "n_samples": actual_n,
        "n_iter": mlp.n_iter_, "train_time_s": round(elapsed, 2),
        "final_loss": round(mlp.loss_, 6),
        "pos_MAE_mm":  round(float(np.mean(pos_err_nn)), 4),
        "pos_Max_mm":  round(float(np.max(pos_err_nn)), 4),
        "pos_STD_mm":  round(float(np.std(pos_err_nn)), 4),
        "q1_MAE_deg":  round(float(np.mean(q_err[:, 0])), 4),
        "q2_MAE_deg":  round(float(np.mean(q_err[:, 1])), 4),
        "traj_MAE_mm": round(float(np.mean(traj_err_nn)), 4),
        "traj_Max_mm": round(float(np.max(traj_err_nn)), 4),
        "_loss_curve":  mlp.loss_curve_,
        "_pos_err_nn":  pos_err_nn,
        "_traj_err_nn": np.array(traj_err_nn),
        "_theta_tr":    np.degrees(theta_tr),
    }

    # ── Додаткові дані для базового рисунку (рис. 3.1) ──────────────────
    if collect_baseline_data:
        result["_XY_test_raw"]  = XY_test
        result["_Q_pred_deg"]   = np.degrees(Q_pred_rad)
        result["_Q_true_deg"]   = np.degrees(Q_true_rad)
        result["_pos_err_mb"]   = pos_err_mb
        result["_traj_x"]       = traj_x
        result["_traj_y"]       = traj_y
        result["_traj_x_nn"]    = np.array(traj_x_nn)
        result["_traj_y_nn"]    = np.array(traj_y_nn)
        result["_traj_x_mb"]    = np.array(traj_x_mb)
        result["_traj_y_mb"]    = np.array(traj_y_mb)

    return result

# ════════════════════════════════════════════════════════════════════════════
# БЛОК 4 — ЗБЕРЕЖЕННЯ CSV
# ════════════════════════════════════════════════════════════════════════════

CSV_FIELDS = ["run_id", "L1", "L2", "n_samples", "n_iter", "train_time_s",
              "final_loss", "pos_MAE_mm", "pos_Max_mm", "pos_STD_mm",
              "q1_MAE_deg", "q2_MAE_deg", "traj_MAE_mm", "traj_Max_mm"]

def save_csv(results_list, filename):
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in results_list:
            writer.writerow({k: r.get(k, "") for k in CSV_FIELDS})
    print(f"   ✔ Збережено: {filepath}")

# ════════════════════════════════════════════════════════════════════════════
# БЛОК 5 — РІВЕНЬ 1: ВАРІАЦІЯ ГЕОМЕТРІЇ
# ════════════════════════════════════════════════════════════════════════════

print("═" * 68)
print("  КЕЙС-СТАДІ: Аналіз чутливості data-driven підходу")
print("  Апроксимація оберненої кінематики 2-DOF маніпулятора (MLP)")
print("═" * 68)

results_geom = []

print(f"\n{'─'*68}")
print(f"  РІВЕНЬ 1: Варіація геометрії маніпулятора  (N={N_FIXED_GEOM:,})")
print(f"{'─'*68}")

for idx, cfg in enumerate(GEOMETRY_CONFIGS):
    L1, L2 = cfg["L1"], cfg["L2"]
    print(f"\n  ▶ Конфігурація {cfg['id']}: L1={L1}, L2={L2}")
    # Конфігурація A (idx==0): збираємо розширені дані для рис. 3.1
    is_baseline = (idx == 0)
    r = run_single(L1, L2, N_FIXED_GEOM, collect_baseline_data=is_baseline)
    r["run_id"] = f"GEOM_{cfg['id']}"
    r["label"]  = cfg["label"]
    results_geom.append(r)
    print(f"    Ітерацій: {r['n_iter']}  |  Час: {r['train_time_s']} с  "
          f"|  final_loss: {r['final_loss']}")
    print(f"    [В2] MLP: pos MAE = {r['pos_MAE_mm']:.4f} мм  |  "
          f"q₁={r['q1_MAE_deg']:.4f}°  q₂={r['q2_MAE_deg']:.4f}°  "
          f"(без аналітичної моделі, N={r['n_samples']:,})")

save_csv(results_geom, "results_level1_geometry.csv")

# ════════════════════════════════════════════════════════════════════════════
# БЛОК 5.5 — БАЗОВИЙ РИСУНОК (рис. 3.1): 6 панелей для конфігурації A
# ════════════════════════════════════════════════════════════════════════════

print(f"\n{'─'*68}")
print("  Побудова базового рисунку (рис. 3.1, конфігурація A)...")

rb = results_geom[0]   # конфігурація A з повними даними

fig1, axes1 = plt.subplots(2, 3, figsize=(16, 10))
fig1.patch.set_facecolor('#F8F9FA')
fig1.suptitle(
    "Кейс-стаді: Data-Driven апроксимація оберненої кінематики 2-DOF маніпулятора\n"
    "Supervised Learning (MLP, Scikit-learn) vs Аналітичний model-based розв'язок",
    fontsize=13, fontweight='bold'
)

def ax1_style(ax, title):
    ax.set_facecolor('#FAFAFA')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
    ax.grid(True, alpha=0.3)

# ① Крива навчання MLP
ax = axes1[0, 0]
ax1_style(ax, "① Крива навчання MLP")
ax.semilogy(rb['_loss_curve'], color='#1f77b4', lw=2, label='Train loss (MSE)')
ax.set_xlabel("Ітерація", fontsize=10)
ax.set_ylabel("MSE (log)", fontsize=10)
ax.legend(fontsize=9)

# ② Просторовий розподіл похибки (теплова карта)
ax = axes1[0, 1]
ax1_style(ax, "② Просторовий розподіл похибки")
xy   = rb['_XY_test_raw']
errs = rb['_pos_err_nn']
p98  = np.percentile(errs, 98)
sc   = ax.scatter(xy[:, 0], xy[:, 1],
                  c=np.clip(errs, 0, p98),
                  cmap='RdYlGn_r', s=18, alpha=0.75, vmin=0, vmax=p98)
cb = fig1.colorbar(sc, ax=ax, shrink=0.85)
cb.set_label("Похибка MLP [мм]", fontsize=9)
ax.set_xlabel("X [м]", fontsize=10)
ax.set_ylabel("Y [м]", fontsize=10)
ax.set_aspect('equal')
# Кільце робочого простору
L1b, L2b = rb['L1'], rb['L2']
theta_ring = np.linspace(0, 2*np.pi, 300)
for rad in [abs(L1b - L2b) + 0.02, L1b + L2b - 0.02]:
    ax.plot(rad*np.cos(theta_ring), rad*np.sin(theta_ring),
            '--', color='#AAAAAA', lw=0.8, alpha=0.6)

# ③ Розподіл позиційних похибок
ax = axes1[0, 2]
ax1_style(ax, "③ Розподіл позиційних похибок")
mae_nn = rb['pos_MAE_mm']
mae_mb = float(np.mean(rb['_pos_err_mb'])) if len(rb['_pos_err_mb']) > 0 else 0.0
p95 = np.percentile(errs, 95)
ax.hist(rb['_pos_err_mb'] if len(rb['_pos_err_mb']) > 0 else [0.0],
        bins=50, range=(0, max(p95, 1.0)),
        color='#1f77b4', alpha=0.7, density=True,
        label=f"Аналітичний  MAE≈{mae_mb:.5f} мм")
ax.hist(errs, bins=50, range=(0, max(p95, 1.0)),
        color='#d62728', alpha=0.6, density=True,
        label=f"MLP (NN)  MAE≈{mae_nn:.4f} мм")
ax.axvline(mae_nn, color='#d62728', lw=1.8, ls='--')
ax.set_xlabel("Позиційна похибка [мм]", fontsize=10)
ax.set_ylabel("Щільність", fontsize=10)
ax.legend(fontsize=8.5)

# ④ Scatter: Передбачення vs Еталон
ax = axes1[1, 0]
ax1_style(ax, "④ Scatter: Передбачення vs Еталон")
q_pred = rb['_Q_pred_deg']
q_true = rb['_Q_true_deg']
lims   = [min(q_true.min(), q_pred.min()) - 5,
          max(q_true.max(), q_pred.max()) + 5]
ax.scatter(q_true[:, 0], q_pred[:, 0],
           color='#e377c2', s=6, alpha=0.35, label='q₁')
ax.scatter(q_true[:, 1], q_pred[:, 1],
           color='#ff7f0e', s=6, alpha=0.35, label='q₂')
ax.plot(lims, lims, 'k--', lw=1.5, label='Ідеал (y=x)')
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Істинні кути [°]", fontsize=10)
ax.set_ylabel("Передбачені кути [°]", fontsize=10)
ax.set_aspect('equal')
ax.legend(fontsize=8.5)

# ⑤ Відстеження кругової траєкторії
ax = axes1[1, 1]
ax1_style(ax, "⑤ Відстеження кругової траєкторії")
ax.plot(rb['_traj_x'],    rb['_traj_y'],    'k-',  lw=2.5, label='Задана траєкторія')
ax.plot(rb['_traj_x_mb'], rb['_traj_y_mb'], 'b--', lw=1.8, label='Аналітичний')
ax.plot(rb['_traj_x_nn'], rb['_traj_y_nn'], 'r-',  lw=1.5, label='MLP (NN)', alpha=0.85)
ax.set_xlabel("X [м]", fontsize=10)
ax.set_ylabel("Y [м]", fontsize=10)
ax.set_aspect('equal')
ax.legend(fontsize=8.5)

# ⑥ Похибка вздовж траєкторії
ax = axes1[1, 2]
ax1_style(ax, "⑥ Похибка вздовж траєкторії")
traj_err = rb['_traj_err_nn']
traj_mae = rb['traj_MAE_mm']
ax.plot(rb['_theta_tr'], np.zeros_like(rb['_theta_tr']),
        color='#1f77b4', lw=1.8, label='Аналітичний')
ax.plot(rb['_theta_tr'], traj_err,
        color='#d62728', lw=1.8, ls='--', label='MLP (NN)')
ax.axhline(traj_mae, color='#d62728', lw=1.2, ls=':',
           label=f"MLP mean={traj_mae:.3f} мм")
ax.set_xlabel("Кут на траєкторії [°]", fontsize=10)
ax.set_ylabel("Позиційна похибка [мм]", fontsize=10)
ax.set_xlim(0, 360)
ax.legend(fontsize=8.5)

plt.tight_layout(rect=[0, 0, 1, 0.93])
out_baseline = os.path.join(OUTPUT_DIR, 'case_study_baseline.png')
plt.savefig(out_baseline, dpi=150, bbox_inches='tight',
            facecolor=fig1.get_facecolor())
print(f"   ✔ Збережено: {out_baseline}")
plt.show()

# ════════════════════════════════════════════════════════════════════════════
# БЛОК 6 — РІВЕНЬ 2: ВАРІАЦІЯ ОБСЯГУ ДАТАСЕТУ
# ════════════════════════════════════════════════════════════════════════════

results_size = []

print(f"\n{'─'*68}")
print(f"  РІВЕНЬ 2: Варіація обсягу датасету  (L1={L1_BASE}, L2={L2_BASE})")
print(f"{'─'*68}")

for n in DATASET_SIZES:
    print(f"\n  ▶ N = {n:,}")
    r = run_single(L1_BASE, L2_BASE, n)
    r["run_id"] = f"SIZE_{n}"
    r["label"]  = f"N={n:,}"
    results_size.append(r)
    print(f"    Ітерацій: {r['n_iter']}  |  Час: {r['train_time_s']} с  "
          f"|  final_loss: {r['final_loss']}")
    print(f"    [В2] MLP: pos MAE = {r['pos_MAE_mm']:.4f} мм  |  "
          f"q₁={r['q1_MAE_deg']:.4f}°  q₂={r['q2_MAE_deg']:.4f}°  "
          f"(без аналітичної моделі, N={r['n_samples']:,})")

save_csv(results_size, "results_level2_dataset_size.csv")

# ════════════════════════════════════════════════════════════════════════════
# БЛОК 7 — ЗВЕДЕНІ ТАБЛИЦІ У ТЕРМІНАЛ
# ════════════════════════════════════════════════════════════════════════════

print(f"\n{'═'*68}")
print("  ЗВЕДЕНІ ТАБЛИЦІ РЕЗУЛЬТАТІВ")
print(f"{'═'*68}")

print(f"\n  Таблиця 3.2 — Рівень 1: варіація геометрії (N={N_FIXED_GEOM:,})")
sep = "  " + "─"*66
print(sep)
print(f"  {'Конф.':>6} │ L1  │ L2  │ {'MAE[мм]':>8} │ {'Max[мм]':>9} │ "
      f"{'STD[мм]':>8} │ {'q₁[°]':>7} │ {'q₂[°]':>7} │ {'t[с]':>5}")
print(sep)
for r in results_geom:
    cid = r['run_id'].replace("GEOM_", "")
    print(f"  {cid:>6} │{r['L1']:>4} │{r['L2']:>4} │ "
          f"{r['pos_MAE_mm']:>8.4f} │ {r['pos_Max_mm']:>9.2f} │ "
          f"{r['pos_STD_mm']:>8.4f} │ {r['q1_MAE_deg']:>7.4f} │ "
          f"{r['q2_MAE_deg']:>7.4f} │ {r['train_time_s']:>5.1f}")

print(f"\n  Таблиця 3.3 — Рівень 2: варіація N (L1={L1_BASE}, L2={L2_BASE})")
print(sep)
print(f"  {'N':>8} │ {'MAE[мм]':>8} │ {'Max[мм]':>9} │ {'STD[мм]':>8} │ "
      f"{'q₁[°]':>7} │ {'q₂[°]':>7} │ {'Ітер.':>6} │ {'t[с]':>5}")
print(sep)
for r in results_size:
    print(f"  {r['n_samples']:>8,} │ {r['pos_MAE_mm']:>8.4f} │ "
          f"{r['pos_Max_mm']:>9.2f} │ {r['pos_STD_mm']:>8.4f} │ "
          f"{r['q1_MAE_deg']:>7.4f} │ {r['q2_MAE_deg']:>7.4f} │ "
          f"{r['n_iter']:>6} │ {r['train_time_s']:>5.1f}")

# ════════════════════════════════════════════════════════════════════════════
# БЛОК 8 — ЗВЕДЕНА ВІЗУАЛІЗАЦІЯ (12 графіків → sensitivity_analysis.png)
# ════════════════════════════════════════════════════════════════════════════

print(f"\n{'─'*68}")
print("  Побудова зведених графіків (рис. 3.2)...")

CLR_SIZE = plt.cm.viridis(np.linspace(0.15, 0.85, len(DATASET_SIZES)))

fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('#F8F9FA')
gs_main = gridspec.GridSpec(2, 1, figure=fig, hspace=0.48, height_ratios=[1, 1])
gs_top  = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs_main[0],
                                           hspace=0.58, wspace=0.38)
gs_bot  = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs_main[1],
                                           hspace=0.58, wspace=0.38)

fig.suptitle(
    "Аналіз чутливості: Data-Driven апроксимація оберненої кінематики 2-DOF маніпулятора\n"
    "Рівень 1: варіація геометрії (4 конфігурації, N=25 000) │ "
    "Рівень 2: варіація датасету (L1=1.0, L2=0.8, 5 значень N)",
    fontsize=12, fontweight='bold', y=0.995
)

def ax_style(ax, title):
    ax.set_facecolor('#FAFAFA')
    ax.set_title(title, fontsize=9.5, fontweight='bold', pad=4)
    ax.grid(True, alpha=0.3)

# ──── РІВЕНЬ 1: рядок 1 ──────────────────────────────────────────────────────

ax = fig.add_subplot(gs_top[0, 0])
ax_style(ax, "Г1. Криві навчання (Рівень 1)")
for i, r in enumerate(results_geom):
    ax.semilogy(r['_loss_curve'], color=CLR_GEOM[i], lw=1.8,
                label=f"{GEOMETRY_CONFIGS[i]['id']}: L1={r['L1']},L2={r['L2']}")
ax.set_xlabel("Ітерація", fontsize=8.5); ax.set_ylabel("MSE (log)", fontsize=8.5)
ax.legend(fontsize=7)

ax2 = fig.add_subplot(gs_top[0, 1])
ax_style(ax2, "Г2. Позиційна MAE (Рівень 1)")
ids  = [GEOMETRY_CONFIGS[i]['id'] for i in range(len(results_geom))]
maes = [r['pos_MAE_mm'] for r in results_geom]
bars = ax2.bar(ids, maes, color=CLR_GEOM, alpha=0.85, edgecolor='white', width=0.55)
for bar, v in zip(bars, maes):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
             f"{v:.2f}", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
ax2.set_xlabel("Конфігурація", fontsize=8.5); ax2.set_ylabel("MAE [мм]", fontsize=8.5)

ax3 = fig.add_subplot(gs_top[0, 2])
ax_style(ax3, "Г3. Кутові похибки q₁, q₂ (Рівень 1)")
x_pos = np.arange(len(results_geom)); w = 0.38
ax3.bar(x_pos-w/2, [r['q1_MAE_deg'] for r in results_geom], w,
        color='#5B8DB8', alpha=0.85, label='q₁')
ax3.bar(x_pos+w/2, [r['q2_MAE_deg'] for r in results_geom], w,
        color='#E07B54', alpha=0.85, label='q₂')
ax3.set_xticks(x_pos); ax3.set_xticklabels(ids, fontsize=9)
ax3.set_ylabel("MAE [°]", fontsize=8.5); ax3.legend(fontsize=8)

ax4 = fig.add_subplot(gs_top[0, 3])
ax_style(ax4, "Г4. Похибка вздовж траєкторії (Рівень 1)")
for i, r in enumerate(results_geom):
    ax4.plot(r['_theta_tr'], r['_traj_err_nn'], color=CLR_GEOM[i],
             lw=1.6, label=f"Конф.{ids[i]}", alpha=0.85)
ax4.set_xlabel("Кут [°]", fontsize=8.5); ax4.set_ylabel("Похибка [мм]", fontsize=8.5)
ax4.set_xlim(0, 360); ax4.legend(fontsize=7)

# ──── РІВЕНЬ 1: рядок 2 ──────────────────────────────────────────────────────

ax5 = fig.add_subplot(gs_top[1, :2])
ax_style(ax5, "Г5. Розподіл позиційних похибок по конфігураціях (Рівень 1)")
for i, r in enumerate(results_geom):
    p95 = np.percentile(r['_pos_err_nn'], 95)
    ax5.hist(r['_pos_err_nn'], bins=50, range=(0, max(p95, 5.0)),
             color=CLR_GEOM[i], alpha=0.5, density=True,
             label=f"Конф.{ids[i]} MAE={r['pos_MAE_mm']:.2f}мм")
ax5.set_xlabel("Позиційна похибка [мм]", fontsize=8.5)
ax5.set_ylabel("Щільність", fontsize=8.5); ax5.legend(fontsize=8)

ax6 = fig.add_subplot(gs_top[1, 2:])
ax6.set_facecolor('#FAFAFA'); ax6.axis('off')
col6 = ["Конф.", "L1", "L2", "MAE мм", "Max мм", "q₁°", "q₂°", "t с"]
dat6 = [[ids[i], r['L1'], r['L2'],
         f"{r['pos_MAE_mm']:.2f}", f"{r['pos_Max_mm']:.1f}",
         f"{r['q1_MAE_deg']:.3f}", f"{r['q2_MAE_deg']:.3f}",
         f"{r['train_time_s']:.1f}"] for i,r in enumerate(results_geom)]
tbl6 = ax6.table(cellText=dat6, colLabels=col6, loc='center', cellLoc='center')
tbl6.auto_set_font_size(False); tbl6.set_fontsize(9); tbl6.scale(1.05, 1.6)
for (row, col), cell in tbl6.get_celld().items():
    if row == 0:
        cell.set_facecolor('#2E5F8A'); cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 1:
        cell.set_facecolor('#EDF2FA')
ax6.set_title("Г6. Зведена таблиця (Рівень 1)", fontsize=9.5,
              fontweight='bold', pad=8)

# ──── РІВЕНЬ 2: рядок 1 ──────────────────────────────────────────────────────

ns = [r['n_samples'] for r in results_size]

ax7 = fig.add_subplot(gs_bot[0, 0])
ax_style(ax7, "Г7. Sample Efficiency: MAE від N (Рівень 2)")
maes7 = [r['pos_MAE_mm'] for r in results_size]
ax7.semilogx(ns, maes7, 'o-', color='#1f77b4', lw=2.5, ms=8,
             markerfacecolor='white', markeredgewidth=2)
for x, y in zip(ns, maes7):
    ax7.annotate(f"{y:.1f}", (x, y), textcoords="offset points",
                 xytext=(4, 6), fontsize=8)
ax7.set_xlabel("N (log)", fontsize=8.5); ax7.set_ylabel("MAE [мм]", fontsize=8.5)
ax7.set_xticks(ns)
ax7.set_xticklabels([f"{n:,}" for n in ns], rotation=22, fontsize=7.5)

ax8 = fig.add_subplot(gs_bot[0, 1])
ax_style(ax8, "Г8. Криві навчання по N (Рівень 2)")
for i, r in enumerate(results_size):
    ax8.semilogy(r['_loss_curve'], color=CLR_SIZE[i], lw=1.8,
                 label=f"N={r['n_samples']:,}")
ax8.set_xlabel("Ітерація", fontsize=8.5); ax8.set_ylabel("MSE (log)", fontsize=8.5)
ax8.legend(fontsize=7.5)

ax9 = fig.add_subplot(gs_bot[0, 2])
ax_style(ax9, "Г9. Кутові похибки від N (Рівень 2)")
ax9.semilogx(ns, [r['q1_MAE_deg'] for r in results_size],
             's--', color='#5B8DB8', lw=2, ms=8, label='q₁')
ax9.semilogx(ns, [r['q2_MAE_deg'] for r in results_size],
             '^--', color='#E07B54', lw=2, ms=8, label='q₂')
ax9.set_xlabel("N (log)", fontsize=8.5); ax9.set_ylabel("MAE [°]", fontsize=8.5)
ax9.set_xticks(ns)
ax9.set_xticklabels([f"{n:,}" for n in ns], rotation=22, fontsize=7.5)
ax9.legend(fontsize=8)

ax10 = fig.add_subplot(gs_bot[0, 3])
ax_style(ax10, "Г10. Похибка траєкторії по N (Рівень 2)")
for i, r in enumerate(results_size):
    ax10.plot(r['_theta_tr'], r['_traj_err_nn'], color=CLR_SIZE[i],
              lw=1.5, label=f"N={r['n_samples']:,}", alpha=0.85)
ax10.set_xlabel("Кут [°]", fontsize=8.5); ax10.set_ylabel("Похибка [мм]", fontsize=8.5)
ax10.set_xlim(0, 360); ax10.legend(fontsize=7.5)

# ──── РІВЕНЬ 2: рядок 2 ──────────────────────────────────────────────────────

ax11 = fig.add_subplot(gs_bot[1, :2])
ax_style(ax11, "Г11. Boxplot позиційних похибок по N (Рівень 2)")
bp = ax11.boxplot([r['_pos_err_nn'] for r in results_size],
                  patch_artist=True,
                  medianprops=dict(color='black', lw=2),
                  flierprops=dict(marker='.', alpha=0.3, ms=3))
for patch, clr in zip(bp['boxes'], CLR_SIZE):
    patch.set_facecolor(clr); patch.set_alpha(0.75)
ax11.set_xticklabels([f"N={r['n_samples']:,}" for r in results_size],
                     rotation=15, fontsize=8)
ax11.set_ylabel("Позиційна похибка [мм]", fontsize=8.5)
ax11.set_ylim(bottom=0)

ax12 = fig.add_subplot(gs_bot[1, 2:])
ax12.set_facecolor('#FAFAFA'); ax12.axis('off')
col12 = ["N", "MAE мм", "Max мм", "STD мм", "q₁°", "q₂°", "Ітер.", "t с"]
dat12 = [[f"{r['n_samples']:,}", f"{r['pos_MAE_mm']:.2f}",
          f"{r['pos_Max_mm']:.1f}", f"{r['pos_STD_mm']:.2f}",
          f"{r['q1_MAE_deg']:.3f}", f"{r['q2_MAE_deg']:.3f}",
          str(r['n_iter']), f"{r['train_time_s']:.1f}"]
         for r in results_size]
tbl12 = ax12.table(cellText=dat12, colLabels=col12, loc='center', cellLoc='center')
tbl12.auto_set_font_size(False); tbl12.set_fontsize(9); tbl12.scale(1.05, 1.6)
for (row, col), cell in tbl12.get_celld().items():
    if row == 0:
        cell.set_facecolor('#2E5F8A'); cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 1:
        cell.set_facecolor('#EDF2FA')
ax12.set_title("Г12. Зведена таблиця (Рівень 2)", fontsize=9.5,
               fontweight='bold', pad=8)

out_png = os.path.join(OUTPUT_DIR, 'sensitivity_analysis.png')
plt.savefig(out_png, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print(f"   ✔ Збережено: {out_png}")
plt.show()

# ════════════════════════════════════════════════════════════════════════════
print(f"\n{'═'*68}")
print("  ВИКОНАННЯ ЗАВЕРШЕНО. Збережені файли:")
print("   • results_level1_geometry.csv")
print("   • results_level2_dataset_size.csv")
print("   • case_study_baseline.png   ← рис. 3.1 (базовий кейс-стаді, конф. A)")
print("   • sensitivity_analysis.png  ← рис. 3.2 (аналіз чутливості, Г1–Г12)")
print(f"{'═'*68}")
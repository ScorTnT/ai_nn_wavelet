"""
data.py - 족압 데이터 그룹별(Co/Pt) 경향 시각화

다른 시각화 파일들이 512개 세그먼트 단위로 개별 산점도를 만드는 반면,
이 파일은 모든 세그먼트를 **정규화(0~1)** 후 합쳐서
파킨슨(Pt) vs 대조군(Co)의 **밀도 히트맵**을 비교한다.

정규화 → 피험자별 족압 크기 차이 제거 → 패턴(모양)만 비교
2D 히스토그램 히트맵 → 밀집 영역의 차이가 시각적으로 명확
"""
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm


# =============================================================================
# 공통 파라미터
# =============================================================================
input_folder = r"code/data_preprocess/foot/_xlsx_output"
output_folder = r"code/data_visualization/footPressure/_group_comparison"
os.makedirs(output_folder, exist_ok=True)

skip_seconds = 8
segment_size = 512


# =============================================================================
# 파일 목록 → Co / Pt 그룹 분류
# =============================================================================
file_list = sorted(glob.glob(os.path.join(input_folder, "*.xlsx")))
file_list = [f for f in file_list if os.path.isfile(f)]

groups = {"Co": [], "Pt": []}
for fp in file_list:
    bn = os.path.basename(fp)
    if "Co" in bn:
        groups["Co"].append(fp)
    elif "Pt" in bn:
        groups["Pt"].append(fp)

print(f"[INFO] Co(대조군): {len(groups['Co'])}파일, Pt(파킨슨): {len(groups['Pt'])}파일")


# =============================================================================
# 공통: 파일에서 skip_seconds 적용 후 lp, rp 반환
# =============================================================================
def load_and_skip(fp):
    df = pd.read_excel(fp, engine="openpyxl")
    if "right_total" not in df.columns or "left_total" not in df.columns:
        return None, None

    rp = df["right_total"].to_numpy()
    lp = df["left_total"].to_numpy()

    if "time" in df.columns:
        time_col = df["time"].to_numpy()
        time_s = []
        for t in time_col:
            try:
                time_s.append(float(t))
            except:
                time_s.append(np.nan)
        time_s = np.array(time_s)
        valid = np.where(time_s >= skip_seconds)[0]
        rp = rp[valid]
        lp = lp[valid]
    else:
        skip_n = int(skip_seconds * 100)
        rp = rp[skip_n:]
        lp = lp[skip_n:]

    return lp, rp


def normalize_segment(seg):
    """세그먼트를 0~1로 min-max 정규화"""
    mn, mx = np.nanmin(seg), np.nanmax(seg)
    if mx == mn:
        return np.zeros_like(seg)
    return (seg - mn) / (mx - mn)


def plot_density_heatmap(x, y, title, xlabel, ylabel, out_path, bins=150):
    """2D 히스토그램 밀도 히트맵 생성"""
    fig, ax = plt.subplots(figsize=(10, 10))

    h = ax.hist2d(x, y, bins=bins, range=[[0, 1], [0, 1]],
                  cmap="hot_r", norm=LogNorm(), cmin=1)
    fig.colorbar(h[3], ax=ax, label="Count (log scale)")

    ax.plot([0, 1], [0, 1], "cyan", ls="--", lw=1.5, alpha=0.7, label="y=x")

    ax.set_title(title, fontsize=13)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=10)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 1) rp(n) vs rp(n+1) — 오른발 n vs n+1  (정규화 + 히트맵)
# =============================================================================
print("\n[1/3] rp(n) vs rp(n+1) 그룹별 밀도 히트맵 생성...")
for gname, gfiles in groups.items():
    label = "Control (정상)" if gname == "Co" else "Parkinson (환자)"

    xs, ys = [], []
    seg_cnt = 0

    for fp in gfiles:
        _, rp = load_and_skip(fp)
        if rp is None:
            continue

        x_all = rp[:-1]
        y_all = rp[1:]
        n_seg = len(x_all) // segment_size

        for i in range(n_seg):
            s, e = i * segment_size, (i + 1) * segment_size
            xseg = x_all[s:e]
            yseg = y_all[s:e]

            # 세그먼트 단위로 정규화 (x, y를 합친 범위로 정규화)
            combined = np.concatenate([xseg, yseg])
            mn, mx = np.nanmin(combined), np.nanmax(combined)
            if mx == mn:
                continue
            xn = (xseg - mn) / (mx - mn)
            yn = (yseg - mn) / (mx - mn)

            xs.append(xn)
            ys.append(yn)
            seg_cnt += 1

    if not xs:
        print(f"  ⚠ {gname}: 데이터 없음")
        continue

    xp = np.concatenate(xs)
    yp = np.concatenate(ys)

    out = os.path.join(output_folder, f"rp_n-n+1_{gname}.png")
    plot_density_heatmap(
        xp, yp,
        title=f"Right Foot: n vs n+1 (normalized)\n[{label}] {len(gfiles)} files, {seg_cnt} segs, {len(xp)} pts",
        xlabel="RP(n) normalized",
        ylabel="RP(n+1) normalized",
        out_path=out
    )
    print(f"  ✓ {label}: {out} ({len(xp)} pts)")


# =============================================================================
# 2) n vs n+1 total — 양발 (정규화 + 히트맵)
# =============================================================================
print("\n[2/3] n vs n+1 total (양발) 그룹별 밀도 히트맵 생성...")
for gname, gfiles in groups.items():
    label = "Control (정상)" if gname == "Co" else "Parkinson (환자)"

    xs_rp, ys_rp = [], []
    xs_lp, ys_lp = [], []
    seg_cnt = 0

    for fp in gfiles:
        lp, rp = load_and_skip(fp)
        if rp is None:
            continue

        xr_all, yr_all = rp[:-1], rp[1:]
        xl_all, yl_all = lp[:-1], lp[1:]
        n_seg = len(xr_all) // segment_size

        for i in range(n_seg):
            s, e = i * segment_size, (i + 1) * segment_size

            # 오른발 정규화
            xrs, yrs = xr_all[s:e], yr_all[s:e]
            cr = np.concatenate([xrs, yrs])
            mnr, mxr = np.nanmin(cr), np.nanmax(cr)
            if mxr != mnr:
                xs_rp.append((xrs - mnr) / (mxr - mnr))
                ys_rp.append((yrs - mnr) / (mxr - mnr))

            # 왼발 정규화
            xls, yls = xl_all[s:e], yl_all[s:e]
            cl = np.concatenate([xls, yls])
            mnl, mxl = np.nanmin(cl), np.nanmax(cl)
            if mxl != mnl:
                xs_lp.append((xls - mnl) / (mxl - mnl))
                ys_lp.append((yls - mnl) / (mxl - mnl))

            seg_cnt += 1

    if not xs_rp and not xs_lp:
        print(f"  ⚠ {gname}: 데이터 없음")
        continue

    # 양발 2×1 서브플롯
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    for ax, foot, xs_list, ys_list, clr in [
        (axes[0], "Right Foot", xs_rp, ys_rp, "hot_r"),
        (axes[1], "Left Foot", xs_lp, ys_lp, "YlGnBu"),
    ]:
        if not xs_list:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue
        xp = np.concatenate(xs_list)
        yp = np.concatenate(ys_list)
        h = ax.hist2d(xp, yp, bins=150, range=[[0, 1], [0, 1]],
                      cmap=clr, norm=LogNorm(), cmin=1)
        fig.colorbar(h[3], ax=ax, label="Count (log)")
        ax.plot([0, 1], [0, 1], "cyan" if clr == "hot_r" else "red",
                ls="--", lw=1.5, alpha=0.7, label="y=x")
        ax.set_title(f"{foot} ({len(xp)} pts)", fontsize=12)
        ax.set_xlabel(f"{foot} Pressure(n) norm.")
        ax.set_ylabel(f"{foot} Pressure(n+1) norm.")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.2)
        ax.legend()

    fig.suptitle(f"Foot Pressure: n vs n+1 (normalized)\n[{label}] {len(gfiles)} files, {seg_cnt} segs",
                 fontsize=14, y=1.02)

    out = os.path.join(output_folder, f"n-n+1_total_{gname}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {label}: {out}")


# =============================================================================
# 3) lp(x) vs rp(y) — 왼발 vs 오른발 (정규화 + 히트맵)
# =============================================================================
print("\n[3/3] lp(x) vs rp(y) 그룹별 밀도 히트맵 생성...")
for gname, gfiles in groups.items():
    label = "Control (정상)" if gname == "Co" else "Parkinson (환자)"

    xs, ys = [], []
    seg_cnt = 0

    for fp in gfiles:
        lp, rp = load_and_skip(fp)
        if rp is None:
            continue

        total = min(len(lp), len(rp))
        n_seg = total // segment_size

        for i in range(n_seg):
            s, e = i * segment_size, (i + 1) * segment_size
            xseg = lp[s:e]  # 왼발
            yseg = rp[s:e]  # 오른발

            # 양발을 합친 범위로 정규화 (좌우 비율 관계 유지)
            combined = np.concatenate([xseg, yseg])
            mn, mx = np.nanmin(combined), np.nanmax(combined)
            if mx == mn:
                continue
            xn = (xseg - mn) / (mx - mn)
            yn = (yseg - mn) / (mx - mn)

            xs.append(xn)
            ys.append(yn)
            seg_cnt += 1

    if not xs:
        print(f"  ⚠ {gname}: 데이터 없음")
        continue

    xp = np.concatenate(xs)
    yp = np.concatenate(ys)

    out = os.path.join(output_folder, f"lp-rp_{gname}.png")
    plot_density_heatmap(
        xp, yp,
        title=f"Left vs Right Foot Pressure (normalized)\n[{label}] {len(gfiles)} files, {seg_cnt} segs, {len(xp)} pts",
        xlabel="Left Foot (normalized)",
        ylabel="Right Foot (normalized)",
        out_path=out
    )
    print(f"  ✓ {label}: {out} ({len(xp)} pts)")


print(f"\n[완료] 총 6장 이미지 → {output_folder}")

## 오른발 족압(blue), 왼발 족압(orange)들로 n - n+1 산점도 그리기
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

# =========================
# 파라미터
# =========================
input_folder = r"code/data_preprocess/foot/_xlsx_output"   
output_folder = r"code/data_visualization/footPressure/_n-n+1_total_v3"
os.makedirs(output_folder, exist_ok=True)
# 데이터 초반 몇 초 건너뛸지
skip_seconds = 8

segment_size = 1024  # 세그먼트 당 포인트 개수
# =========================
# 메인 처리
# =========================
file_list = glob.glob(os.path.join(input_folder, "*.xlsx"))
file_list = [f for f in file_list if os.path.isfile(f)]
# print(f"[INFO] 처리할 파일 수: {len(file_list)}")

# 각 파일 처리
for fp in sorted(file_list):
    basename = os.path.splitext(os.path.basename(fp))[0]
    # print(f"[INFO] 처리 중: {basename}")

    # 엑셀 파일 읽기
    df = pd.read_excel(fp, engine="openpyxl")

    # 오른발 족압 컬럼 추출
    if "right_total" not in df.columns:
        print(f"  ⚠ 'right_total' 컬럼 없음: {basename} - 건너뜀")
        continue
    rp = df["right_total"].to_numpy()
    lp = df["left_total"].to_numpy()

    # 시간 컬럼이 있으면 초단위로 변환하여 skip_seconds 만큼 건너뛰기
    if "time" in df.columns:
        time_col = df["time"].to_numpy()
        time_in_seconds = []
        for t in time_col:
            try:
                time_in_seconds.append(float(t))
            except:
                time_in_seconds.append(np.nan)
        time_in_seconds = np.array(time_in_seconds)
        valid_indices = np.where(time_in_seconds >= skip_seconds)[0]
        rp = rp[valid_indices]
    else:
        # 시간 컬럼 없으면 단순히 처음 skip_seconds 초에 해당하는 샘플 수만큼 건너뛰기
        sample_rate = 100  # 가정: 100Hz 샘플링
        skip_samples = int(skip_seconds * sample_rate)
        rp = rp[skip_samples:]
        lp = lp[skip_samples:]

    # n - n+1 산점도 좌표 계산
    x_coords_rp_all = rp[:-1]
    y_coords_rp_all = rp[1:]

    x_coords_lp_all = lp[:-1]
    y_coords_lp_all = lp[1:]

    # 세그먼트 단위로 나누어 처리
    total_points = len(x_coords_rp_all)
    num_segments = total_points // segment_size

    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = start_idx + segment_size

        x_coords_rp = x_coords_rp_all[start_idx:end_idx]
        y_coords_rp = y_coords_rp_all[start_idx:end_idx]
        x_coords_lp = x_coords_lp_all[start_idx:end_idx]
        y_coords_lp = y_coords_lp_all[start_idx:end_idx]

        # 산점도 그리기
        plt.figure(figsize=(8, 8))
        plt.scatter(x_coords_rp, y_coords_rp, alpha=0.5, s=5, color='blue', label='Right Foot')
        plt.scatter(x_coords_lp, y_coords_lp, alpha=0.5, s=5, color='orange', label='Left Foot')
        plt.title(f"Right and Left Foot Pressure: n vs n+1 - {basename} (Seg {i+1})")
        plt.xlabel("Foot Pressure at n")
        plt.ylabel("Foot Pressure at n+1")
        plt.axis('equal')
        plt.grid(True)
        plt.legend()

        # 축 눈금 설정
        if len(x_coords_rp) > 0:
            max_val = max(np.max(x_coords_rp), np.max(y_coords_rp), np.max(x_coords_lp), np.max(y_coords_lp))
            min_val = min(np.min(x_coords_rp), np.min(y_coords_rp), np.min(x_coords_lp), np.min(y_coords_lp))
            range_padding = (max_val - min_val) * 0.05 if max_val != min_val else 1.0
            plt.xlim(min_val - range_padding, max_val + range_padding)
            plt.ylim(min_val - range_padding, max_val + range_padding)
        
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
        plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
        plt.gca().xaxis.set_minor_locator(ticker.AutoMinorLocator())
        plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator())
        plt.gca().tick_params(which='both', width=1)
        plt.gca().tick_params(which='major', length=7)
        plt.gca().tick_params(which='minor', length=4)
        
        # 대각선 그리기
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1)
        
        # 그래프 저장
        out_path = os.path.join(output_folder, f"{basename}_n_vs_nplus1_total_seg{i+1:02d}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        # print(f"  ✓ 산점도 저장: {out_path}")

print(f"[완료] 모든 파일 처리 완료.")

# 왼발 족압을 x축으로, 오른발 족압을 y축으로 산점도 그리기
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.ticker as ticker

# =========================
# 파라미터
# =========================
input_folder = r"code/data_preprocess/foot/_xlsx_output"
output_folder = r"code/data_visualization/footPressure/_lp(x)-rp(y)_v4-512"
os.makedirs(output_folder, exist_ok=True)
# 데이터 초반 몇 초 건너뛸지
skip_seconds = 8
segment_size = 512  # 세그먼트 당 포인트 개수
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
    # 왼발, 오른발 족압 컬럼 추출
    if "left_total" not in df.columns:
        print(f"  ⚠ 'left_total' 컬럼 없음: {basename} - 건너뜀")
        continue
    if "right_total" not in df.columns:
        print(f"  ⚠ 'right_total' 컬럼 없음: {basename} - 건너뜀")
        continue
    lp = df["left_total"].to_numpy()
    rp = df["right_total"].to_numpy()
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
        lp = lp[valid_indices]
        rp = rp[valid_indices]
    else:
        # 시간 컬럼 없으면 단순히 처음 skip_seconds 초에 해당하는 샘플 수만큼 건너뛰기
        sample_rate = 100  # 가정: 100Hz 샘플링
        skip_samples = int(skip_seconds * sample_rate)
        lp = lp[skip_samples:]
        rp = rp[skip_samples:]
    # 세그먼트 단위로 나누어 처리
    total_points = min(len(lp), len(rp))
    if total_points < 2:
        print(f"  ⚠ 데이터 포인트 부족: {basename} - 건너뜀")
        continue

    num_segments = total_points // segment_size

    for i in range(num_segments):
        start_idx = i * segment_size
        end_idx = start_idx + segment_size

        x_coords = lp[start_idx:end_idx]
        y_coords = rp[start_idx:end_idx]

        # 산점도 그리기
        plt.figure(figsize=(8, 8))
        sns.scatterplot(x=x_coords, y=y_coords, s=10, color='blue', alpha=0.6, edgecolor=None)
        plt.title(f"Left Foot Pressure vs Right Foot Pressure - {basename} (Seg {i+1})")
        plt.xlabel("Left Foot Pressure (lp)")
        plt.ylabel("Right Foot Pressure (rp)")
        
        if len(x_coords) > 0 and len(y_coords) > 0:
            plt.xlim(0, np.nanmax(x_coords) * 1.1)
            plt.ylim(0, np.nanmax(y_coords) * 1.1)
            
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.gca().set_aspect('equal', adjustable='box')
        
        # 축 눈금 설정
        def format_func(value, tick_number):
            if value >= 1000:
                return f"{int(value/1000)}k"
            else:
                return f"{int(value)}"
        plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
        plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
        plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
        
        # 그래프 저장
        out_path = os.path.join(output_folder, f"{basename}_lp-rp_scatter_seg{i+1:02d}.png")
        plt.savefig(out_path, dpi=150)
        plt.close() 
    print(f"  ✓ 산점도 저장 완료: {basename}")

print(f"[완료] 모든 파일 처리 완료.")
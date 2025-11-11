"""
TXT -> XLSX 변환 스크립트

동작:
- 입력 폴더(재귀 검색)에서 모든 .txt 파일을 찾음
- 각 파일에서 1열(time), 18열(left total), 19열(right total) (1-based) 추출
- 각 txt 파일을 개별 .xlsx 파일로 변환 (1:1 매핑)
  예: JuPt03_04.txt → JuPt03_04.xlsx

사용 예:
python origin-to-total-xlsx.py --input-dir "../../footPressure-/_data/gait-in-parkinsons-disease-1.0.0" --output-dir "./output_xlsx"
"""
input_dir = "footPressure-/_data"
output_dir = "footPressure/_xlsx_output"

import os
import glob
import argparse
import pandas as pd


def process_files(input_dir: str, output_dir: str, pattern: str = "*.txt", cols=(0, 17, 18)):
	os.makedirs(output_dir, exist_ok=True)

	# 재귀적으로 모든 텍스트 파일 찾기
	search_pattern = os.path.join(input_dir, "**", pattern)
	files = glob.glob(search_pattern, recursive=True)
	files = [f for f in files if os.path.isfile(f)]
	print(f"[INFO] 찾은 파일 수: {len(files)}")

	success_count = 0
	fail_count = 0

	# 각 파일을 개별 xlsx로 변환
	for fp in sorted(files):
		basename = os.path.splitext(os.path.basename(fp))[0]
		out_path = os.path.join(output_dir, f"{basename}.xlsx")
		
		try:
			# 공백/탭 등 여러 구분자를 허용
			df = pd.read_csv(fp, sep=r"\s+", header=None, engine="python", encoding="utf-8", comment='#')
		except Exception as e:
			print(f"  ⚠ 파일 읽기 실패: {fp} ({e}) - 건너뜀")
			fail_count += 1
			continue

		# 필요한 컬럼이 존재하지 않으면 NA로 채우기
		max_col = max(cols)
		if df.shape[1] <= max_col:
			# 부족한 컬럼을 추가
			for c in range(df.shape[1], max_col + 1):
				df[c] = pd.NA

		# 선택한 컬럼 추출 (0-based)
		sel = df.loc[:, list(cols)].copy()
		sel.columns = ["time", "left_total", "right_total"]

		# 시간 컬럼이 문자열이면 숫자로 변환 시도
		sel["time"] = pd.to_numeric(sel["time"], errors="coerce")
		sel["left_total"] = pd.to_numeric(sel["left_total"], errors="coerce")
		sel["right_total"] = pd.to_numeric(sel["right_total"], errors="coerce")

		# 엑셀 파일로 저장
		sel.to_excel(out_path, sheet_name="data", index=False, engine="openpyxl")
		print(f"  ✓ {basename}.txt → {basename}.xlsx ({len(sel)} rows)")
		success_count += 1

	print(f"\n[완료] 성공: {success_count}개, 실패: {fail_count}개")


def main():
	parser = argparse.ArgumentParser(description="Convert TXT files to individual XLSX files (select cols 1,18,19)")
	parser.add_argument("--input-dir", "-i", required=True,default=input_dir, help="입력 루트 폴더 (재귀 검색) - 예: footPressure-/_data/...")
	parser.add_argument("--output-dir", "-o", required=False, default=output_dir, help="출력 폴더")
	args = parser.parse_args()

	cols = tuple(int(x) for x in args.cols.split(","))
	process_files(args.input_dir, args.output_dir, pattern=args.pattern, cols=cols)


if __name__ == "__main__":
	main()
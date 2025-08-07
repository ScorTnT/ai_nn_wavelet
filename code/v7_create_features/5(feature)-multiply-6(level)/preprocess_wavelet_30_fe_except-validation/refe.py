import os

def create_reference_csv_for_training_a_f_pre():
    """
    training-a-f-pre 폴더의 파일들을 기반으로 REFERENCE.csv를 생성합니다.
    기존 training-a to f 폴더들의 REFERENCE.csv를 참고하여 라벨을 매핑합니다.
    형식: 파일명,라벨 (헤더 없음, -1: 정상, 1: 비정상)
    """
    training_dir = "/workspace/training-z-pre"
    
    # training-a-f-pre에 있는 wav 파일들 수집
    wav_files = []
    if os.path.exists(training_dir):
        wav_files = [f for f in os.listdir(training_dir) if f.endswith('.wav')]
        wav_files.sort()
    
    print(f"training-a-f-pre에서 찾은 WAV 파일 수: {len(wav_files)}")
    
    if not wav_files:
        print("WAV 파일을 찾을 수 없습니다!")
        return
    
    # 기존 training 폴더들에서 라벨 정보 수집
    all_labels = {}
    
    for letter in ['a', 'b', 'c', 'd', 'e', 'f']:
        ref_file = f"/workspace/training-{letter}/REFERENCE.csv"
        if os.path.exists(ref_file):
            print(f"training-{letter}/REFERENCE.csv에서 라벨 정보 수집 중...")
            try:
                with open(ref_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        if line and ',' in line:
                            parts = line.split(',')
                            if len(parts) >= 2:
                                file_id = parts[0]  # 파일명 (확장자 없음)
                                label = parts[1]    # 라벨 (-1 또는 1)
                                all_labels[file_id] = label
            except Exception as e:
                print(f"Error reading {ref_file}: {e}")
    
    print(f"총 수집된 라벨 정보: {len(all_labels)}개")
    
    # 라벨 분포 확인
    label_counts = {}
    for label in all_labels.values():
        label_counts[label] = label_counts.get(label, 0) + 1
    print(f"수집된 라벨 분포: {label_counts}")
    
    # training-a-f-pre의 파일들에 대한 REFERENCE.csv 생성
    reference_lines = []
    found_labels = 0
    missing_labels = []
    
    for wav_file in wav_files:
        file_id = wav_file.replace('.wav', '')
        
        if file_id in all_labels:
            reference_lines.append(f"{file_id},{all_labels[file_id]}")
            found_labels += 1
        else:
            missing_labels.append(file_id)
            # 기본값으로 -1 (정상) 설정
            reference_lines.append(f"{file_id},-1")
    
    print(f"라벨이 매핑된 파일: {found_labels}/{len(wav_files)}")
    
    if missing_labels:
        print(f"라벨을 찾을 수 없는 파일 수: {len(missing_labels)}")
        if len(missing_labels) <= 10:
            print(f"라벨을 찾을 수 없는 파일들: {missing_labels}")
        else:
            print(f"라벨을 찾을 수 없는 파일들 (처음 10개): {missing_labels[:10]}")
    
    # REFERENCE.csv 저장 (헤더 없음)
    reference_csv_path = os.path.join(training_dir, 'REFERENCE.csv')
    with open(reference_csv_path, 'w') as f:
        for line in reference_lines:
            f.write(line + '\n')
    
    print(f"\nREFERENCE.csv가 생성되었습니다: {reference_csv_path}")
    print(f"총 레코드 수: {len(reference_lines)}")
    
    # 생성된 라벨 분포 확인
    generated_label_counts = {}
    for line in reference_lines:
        label = line.split(',')[1]
        generated_label_counts[label] = generated_label_counts.get(label, 0) + 1
    
    print(f"\n생성된 라벨 분포:")
    for label, count in sorted(generated_label_counts.items()):
        label_type = "정상" if label == "-1" else "비정상" if label == "1" else "기타"
        print(f"  {label} ({label_type}): {count}개")
    
    # 생성된 파일의 처음 몇 줄 확인
    print(f"\n생성된 REFERENCE.csv의 처음 10줄:")
    with open(reference_csv_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines[:10]):
            print(f"  {i+1}: {line.strip()}")
    
    return reference_csv_path

if __name__ == "__main__":
    create_reference_csv_for_training_a_f_pre()
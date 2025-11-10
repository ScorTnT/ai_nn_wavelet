# visual.py
가장 초기 버전
.wav 파일을 읽어서 sr=none으로 설정하여 원본 샘플링 레이트 유지
각각의 음성 파일을 plot으로 그려서 저장
- offset : n, n+{offset} 변위 옵션  
- target_duration : 초단위 목표 길이 옵션
  

# visual_n-n2.py
SKIP_SAMPLES 옵션 추가: 기존 샘플링 레이트를 유지해서 음성 파일을 읽고 점을 찍을 때 건너뛰는 변수, offset이랑 완전히 다름.
- skip_duration : 음성 파일의 앞 부분 건너뛰기 옵션
- SKIP_SAMPLES : 건너 뛸 샘플 수 
  
  
# visual_n-n3.py
skip_duration 옵션 없음. n+3 방식이나 n+2 방식에서 큰 차이 없다고 판단(2025-09-30).
  
일단은 n+3 방식은 중단 상태(2025-09-30).
  
  
# visual_n-n2_segmentation.py
목표 샘플링 레이트 추가와 세그먼트 옵션 추가, 기존 파일의 샘플링 레이트를 변경할 수 있게 만들고 분할시에 몇개씩 자를건지 추가. 
예)  
    offset = 1   # n, n+1 방식
    
    target_duration = 8.0  # 8초 목표 길이
    
    skip_duration = 8.0    # 초반 8초 건너뛰기
    
    segment_size = 1024  # 각 세그먼트 크기
    
    target_sampling_rate = 2000  # 목표 샘플링 레이트
    
    8초 * 2000(목표 샘플링 레이트) = 16000 샘플
    
    16000 / 1024(세그먼트 크기) = 15.625
    
    15개 세그먼트로 나눠서 저장하고 나머지 0.625 세그먼트는 버림.
  
  
# visual_n-n2_segmentation-s1-s2.py
segmentation을 수축기 이완기로 변경

  
  
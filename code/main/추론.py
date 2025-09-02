import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 모델 로드 (가장 최근 모델 사용)
model_path = '/workspace/best_model_conv1d.h5'
model = load_model(model_path)

print("=== 모델 정보 ===")
print(f"모델 경로: {model_path}")
model.summary()

# 2. 테스트 데이터 로드
test_data_path = '/workspace/Test/Test30-1024.csv'
test_data = np.loadtxt(test_data_path, delimiter=',', dtype=np.float32)

print(f"\n=== 테스트 데이터 정보 ===")
print(f"데이터 shape: {test_data.shape}")

# 3. 특성과 라벨 분리
X_test = test_data[:, :-1]  # 마지막 컬럼 제외한 모든 특성
y_test = test_data[:, -1].astype(int)   # 마지막 컬럼이 라벨

print(f"특성 shape: {X_test.shape}")
print(f"라벨 shape: {y_test.shape}")
print(f"라벨 분포: {np.bincount(y_test)}")

# 4. 데이터 정규화 (훈련할 때와 동일하게)
scaler = StandardScaler()
X_test_scaled = scaler.fit_transform(X_test)

print(f"\n=== 정규화 후 통계 ===")
print(f"평균: {np.mean(X_test_scaled, axis=0)[:5]}...")  # 처음 5개만 출력
print(f"표준편차: {np.std(X_test_scaled, axis=0)[:5]}...")

# 5. 모델 예측
print("\n=== 모델 예측 수행 ===")
y_pred_proba = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_proba, axis=1)

print(f"예측 확률 shape: {y_pred_proba.shape}")
print(f"예측 라벨 shape: {y_pred.shape}")
print(f"예측 라벨 분포: {np.bincount(y_pred)}")

# 6. 혼동행렬 계산
cm = confusion_matrix(y_test, y_pred)
print(f"\n=== 혼동행렬 ===")
print(cm)

# 7. 민감도와 특이도 계산
tn, fp, fn, tp = cm.ravel()

sensitivity = tp / (tp + fn)  # 민감도 (Recall, True Positive Rate)
specificity = tn / (tn + fp)  # 특이도 (True Negative Rate)
precision = tp / (tp + fp)    # 정밀도
accuracy = (tp + tn) / (tp + tn + fp + fn)  # 정확도

print(f"\n=== 성능 지표 ===")
print(f"정확도 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"민감도 (Sensitivity/Recall): {sensitivity:.4f} ({sensitivity*100:.2f}%)")
print(f"특이도 (Specificity): {specificity:.4f} ({specificity*100:.2f}%)")
print(f"정밀도 (Precision): {precision:.4f} ({precision*100:.2f}%)")
print(f"F1-Score: {2 * (precision * sensitivity) / (precision + sensitivity):.4f}")

# 8. 상세한 분류 보고서
print(f"\n=== 상세 분류 보고서 ===")
print(classification_report(y_test, y_pred, target_names=['Class 0', 'Class 1']))

# 9. 혼동행렬 시각화
plt.figure(figsize=(12, 5))

# 혼동행렬 히트맵
plt.subplot(1, 2, 1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')

# 성능 지표 막대그래프
plt.subplot(1, 2, 2)
metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision']
values = [accuracy, sensitivity, specificity, precision]
colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']

bars = plt.bar(metrics, values, color=colors)
plt.ylim(0, 1)
plt.ylabel('Score')
plt.title('Performance Metrics')

# 막대 위에 값 표시
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('/workspace/model_evaluation_results.png', dpi=150, bbox_inches='tight')
plt.show()

# 10. 예측 확률 분포 분석
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(y_pred_proba[:, 0], bins=50, alpha=0.7, label='Class 0 Probability')
plt.hist(y_pred_proba[:, 1], bins=50, alpha=0.7, label='Class 1 Probability')
plt.xlabel('Prediction Probability')
plt.ylabel('Frequency')
plt.title('Prediction Probability Distribution')
plt.legend()

plt.subplot(1, 2, 2)
# 클래스별 예측 확률 박스플롯
class_0_probs = y_pred_proba[y_test == 0, 1]  # 실제 클래스 0의 클래스 1 예측 확률
class_1_probs = y_pred_proba[y_test == 1, 1]  # 실제 클래스 1의 클래스 1 예측 확률

plt.boxplot([class_0_probs, class_1_probs], labels=['Actual Class 0', 'Actual Class 1'])
plt.ylabel('Predicted Probability for Class 1')
plt.title('Prediction Probability by Actual Class')

plt.tight_layout()
plt.savefig('/workspace/prediction_probability_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# 11. 결과 요약 저장
results_summary = f"""
=== 모델 평가 결과 요약 ===
모델: {model_path}
테스트 데이터: {test_data_path}
테스트 샘플 수: {len(y_test)}

=== 혼동행렬 ===
           Predicted
Actual     0    1
   0     {tn}   {fp}
   1     {fn}   {tp}

=== 성능 지표 ===
정확도 (Accuracy): {accuracy:.4f} ({accuracy*100:.2f}%)
민감도 (Sensitivity): {sensitivity:.4f} ({sensitivity*100:.2f}%)
특이도 (Specificity): {specificity:.4f} ({specificity*100:.2f}%)
정밀도 (Precision): {precision:.4f} ({precision*100:.2f}%)
F1-Score: {2 * (precision * sensitivity) / (precision + sensitivity):.4f}

=== 해석 ===
- 민감도 {sensitivity*100:.1f}%: 실제 양성 중 {sensitivity*100:.1f}%를 올바르게 탐지
- 특이도 {specificity*100:.1f}%: 실제 음성 중 {specificity*100:.1f}%를 올바르게 분류
- 정밀도 {precision*100:.1f}%: 양성으로 예측한 것 중 {precision*100:.1f}%가 실제 양성
"""

with open('/workspace/model_evaluation_summary.txt', 'w', encoding='utf-8') as f:
    f.write(results_summary)

print(results_summary)
print(f"\n결과가 '/workspace/model_evaluation_summary.txt'에 저장되었습니다.")
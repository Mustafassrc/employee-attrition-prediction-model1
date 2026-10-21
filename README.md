# Employee Attrition Prediction — Model 1 (Random Forest)

**English** · [Türkçe](#türkçe)

Predicts whether an employee is likely to leave the company, using HR data and a Random Forest classifier. This version contains the full exploratory analysis and feature-engineering workflow; a streamlined version lives in [employee-attrition-prediction-model2](https://github.com/Mustafassrc/employee-attrition-prediction-model2).

## Dataset

`isciverisi.csv` — IBM HR Analytics Employee Attrition dataset: 1,470 employees, 35 columns (age, department, job role, monthly income, overtime, satisfaction scores, years at company, etc.). Target: `Attrition` (Yes / No).

## Approach

1. Exploratory data analysis: column type detection, categorical and numerical summaries, correlation heatmap
2. Baseline Random Forest on label-encoded data
3. Feature engineering: missing-value analysis, outlier capping, dropping constant columns, label and one-hot encoding, standard scaling
4. Random Forest with 5-fold cross-validation
5. Hyperparameter tuning with `GridSearchCV`
6. Feature importance visualization

## Results (30 % test set)

| Model | Accuracy | CV accuracy |
|---|:-:|:-:|
| Random Forest (tuned) | 0.82 | 0.85 |

**Limitation:** the dataset is imbalanced (~16 % attrition). The model identifies employees who stay very well, but catches only a small share of those who leave (recall for the "leave" class ≈ 0.1). Class weighting or resampling (e.g. SMOTE) would be the natural next step.

## Getting started

```bash
git clone https://github.com/Mustafassrc/employee-attrition-prediction-model1.git
cd employee-attrition-prediction-model1
pip install -r requirements.txt
python employee_attrition_prediction_model1.py
```

## Tech stack

Python · pandas · NumPy · scikit-learn · Matplotlib · seaborn

---

## Türkçe

İK verilerini kullanarak bir çalışanın şirketten ayrılma ihtimalini Random Forest ile tahmin eden model. Bu sürüm, keşifçi veri analizinin ve özellik mühendisliğinin tamamını içerir; sadeleştirilmiş sürüm için [model2](https://github.com/Mustafassrc/employee-attrition-prediction-model2) reposuna bakabilirsiniz.

**Veri seti:** IBM HR Analytics Employee Attrition; 1.470 çalışan, 35 sütun. Hedef değişken `Attrition` (Yes / No).

**Yöntem:** Keşifçi veri analizi → temel Random Forest modeli → özellik mühendisliği (eksik/aykırı değer işlemleri, sabit sütunların çıkarılması, label ve one-hot encoding, ölçeklendirme) → 5 katlı çapraz doğrulama → `GridSearchCV` ile hiperparametre optimizasyonu → özellik önem dereceleri.

**Sonuçlar:** Test doğruluğu 0.82 · Çapraz doğrulama doğruluğu 0.85

**Sınırlama:** Veri dengesiz (çalışanların ~%16'sı ayrılıyor). Model kalanları çok iyi tanıyor ancak ayrılanların yalnızca küçük bir kısmını yakalıyor (ayrılan sınıfı için duyarlılık ≈ 0.1). Sınıf ağırlıklandırma veya SMOTE gibi yöntemler bir sonraki adım olabilir.

**Çalıştırma:**

```bash
pip install -r requirements.txt
python employee_attrition_prediction_model1.py
```

import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = r"D:\MDO\footpress\v1-50sec\_rp(n)-rp(n+1)-50sec"

def parse_subject_and_label(filename: str):
    name, _ = os.path.splitext(filename)
    subject_id = name.split("_")[0] 
    group_tag = subject_id[2:4]

    if group_tag == "Co":
        label = 0
    elif group_tag == "Pt":
        label = 1
    else:
        return None, None

    return subject_id, label

def build_manifest(root: str):
    rows = []
    target_folder = root

    for path in glob.glob(os.path.join(target_folder, "*.png")):
        filename = os.path.basename(path)
        subject_id, label = parse_subject_and_label(filename)
        if subject_id is None:
            continue

        rows.append((path, subject_id, label))

    df = pd.DataFrame(rows, columns=["path", "subject", "label"])
    return df

df = build_manifest(ROOT)
print(df.head())
print(df['label'].value_counts())
print("subjects:", df['subject'].nunique())

subjects = df['subject'].unique()

subj_labels = []
for s in subjects:
    lab = df[df['subject'] == s]['label'].iloc[0]
    subj_labels.append(lab)

subj_df = pd.DataFrame({"subject": subjects, "label": subj_labels})

subj_trainval, subj_test = train_test_split(
    subj_df,
    test_size=0.2,
    stratify=subj_df['label'],
    random_state=42
)

subj_train, subj_val = train_test_split(
    subj_trainval,
    test_size=0.2,
    stratify=subj_trainval['label'],
    random_state=42
)

train_df = df[df['subject'].isin(subj_train['subject'])]
val_df   = df[df['subject'].isin(subj_val['subject'])]
test_df  = df[df['subject'].isin(subj_test['subject'])]

print(len(train_df), len(val_df), len(test_df))

print(df['label'].value_counts())

print("train subject 수:", subj_train['subject'].nunique())
print("val subject 수:", subj_val['subject'].nunique())
print("test subject 수:", subj_test['subject'].nunique())

print(set(train_df['subject']) & set(val_df['subject']))
print(set(train_df['subject']) & set(test_df['subject']))
print(set(val_df['subject']) & set(test_df['subject']))

print(len(train_df), len(val_df), len(test_df))

SAVE_DIR = r"D:\MDO\footpress\result\v1-50sec\_rp(n)-rp(n+1)-50sec"
os.makedirs(SAVE_DIR, exist_ok=True)

train_df.to_csv(os.path.join(SAVE_DIR, "train.csv"), index=False)
val_df.to_csv(os.path.join(SAVE_DIR, "val.csv"), index=False)
test_df.to_csv(os.path.join(SAVE_DIR, "test.csv"), index=False)

print("CSV 저장 완료!")
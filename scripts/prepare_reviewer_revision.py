from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'ASSISTments_2009-2010_Corrected_Skill_Builder.csv'
OUT = ROOT / 'results' / 'reviewer_revision_audit'
OUT.mkdir(parents=True, exist_ok=True)

usecols = ['order_id', 'user_id', 'problem_id', 'skill_id', 'correct', 'skill_name']
df = pd.read_csv(DATA, encoding='latin1', low_memory=False, usecols=usecols)
raw_rows = len(df)
missing_skill_rows = int(df['skill_id'].isna().sum())
work = df[df['skill_id'].notna()].copy()
work['skill_id'] = work['skill_id'].astype(str).str.strip()
top11 = work['skill_id'].value_counts().head(11).index.tolist()
subset = work[work['skill_id'].isin(top11)].sort_values(['user_id', 'order_id']).reset_index(drop=True)

summary = {
    'source_file': str(DATA.relative_to(ROOT)),
    'raw_rows': raw_rows,
    'raw_students': int(df['user_id'].nunique()),
    'raw_problems': int(df['problem_id'].nunique()),
    'missing_skill_rows': missing_skill_rows,
    'rows_with_skill_id': int(len(work)),
    'observed_skill_ids': int(work['skill_id'].nunique()),
    'raw_correct_mean': float(df['correct'].mean()),
    'top11_skill_ids': top11,
    'top11_rows': int(len(subset)),
    'top11_students': int(subset['user_id'].nunique()),
    'top11_problems': int(subset['problem_id'].nunique()),
    'top11_correct_mean': float(subset['correct'].mean()),
    'top11_skill_counts': {str(k): int(v) for k, v in subset['skill_id'].value_counts().sort_index().items()},
    'top11_skill_student_counts': {str(k): int(v) for k, v in subset.groupby('skill_id')['user_id'].nunique().sort_index().items()},
    'top11_skill_problem_counts': {str(k): int(v) for k, v in subset.groupby('skill_id')['problem_id'].nunique().sort_index().items()},
    'duplicate_order_ids': int(subset['order_id'].duplicated().sum()),
    'duplicate_student_problem_pairs': int(subset.duplicated(['user_id', 'problem_id']).sum()),
}
(ROOT / 'results' / 'dataset_summary_reviewer_revision.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

seeds = [17, 29, 41, 53, 67]
split_rows = []
for seed in seeds:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    tr, te = next(splitter.split(subset, groups=subset['user_id']))
    train = subset.iloc[tr]
    test = subset.iloc[te]
    split_rows.append({
        'seed': seed,
        'train_rows': int(len(tr)),
        'test_rows': int(len(te)),
        'train_students': int(train['user_id'].nunique()),
        'test_students': int(test['user_id'].nunique()),
        'train_correct_mean': float(train['correct'].mean()),
        'test_correct_mean': float(test['correct'].mean()),
        'train_skill_counts': {str(k): int(v) for k, v in train['skill_id'].value_counts().sort_index().items()},
        'test_skill_counts': {str(k): int(v) for k, v in test['skill_id'].value_counts().sort_index().items()},
    })
(ROOT / 'results' / 'student_level_splits_reviewer_revision.json').write_text(
    json.dumps({'seeds': seeds, 'splits': split_rows}, indent=2), encoding='utf-8'
)
print(json.dumps({'summary': summary, 'splits': split_rows}, indent=2))

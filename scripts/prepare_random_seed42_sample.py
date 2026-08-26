from pathlib import Path
import hashlib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'ASSISTments_2009-2010_Corrected_Skill_Builder.csv'
OUT = ROOT / 'results'
SEED = 42
SAMPLE_PER_SKILL = 600

cols = ['user_id', 'problem_id', 'skill_id', 'correct', 'order_id']
df = pd.read_csv(DATA, encoding='latin1', low_memory=False, usecols=cols)
df = df[df['skill_id'].notna()].copy()
df['skill_id'] = df['skill_id'].astype(str).str.strip()
top_skills = df['skill_id'].value_counts().head(11).index.tolist()
subset = df[df['skill_id'].isin(top_skills)].copy()
subset['_source_row'] = subset.index.astype(int)

parts = []
for skill in top_skills:
    part = subset[subset['skill_id'] == skill].sample(n=min(SAMPLE_PER_SKILL, len(subset[subset['skill_id'] == skill])), random_state=SEED)
    parts.append(part)
sample = pd.concat(parts, ignore_index=True).sort_values(['user_id', 'order_id']).reset_index(drop=True)

# Use a one-way hash for the public manifest; the original public data remain the
# reconstruction source, while the manifest does not expose raw student IDs.
def hash_student(x):
    return hashlib.sha256(f'{SEED}:{x}'.encode('utf-8')).hexdigest()[:16]

manifest = sample[['skill_id', 'problem_id', 'correct', 'order_id', '_source_row', 'user_id']].copy()
manifest['student_hash'] = manifest['user_id'].map(hash_student)
manifest = manifest.drop(columns=['user_id'])
manifest.insert(0, 'sample_row', range(len(manifest)))
manifest.to_csv(OUT / 'random_seed42_sample_manifest.csv', index=False)

summary = {
    'seed': SEED,
    'sample_rule': 'random sample within each of the 11 most frequent skills, up to 600 records per skill',
    'sample_rows': int(len(sample)),
    'skills': top_skills,
    'skill_counts': sample['skill_id'].value_counts().reindex(top_skills).astype(int).to_dict(),
    'students': int(sample['user_id'].nunique()),
    'problems': int(sample['problem_id'].nunique()),
    'correct_mean': float(sample['correct'].mean()),
    'student_level_split_note': 'The manifest is intended for a subsequent student-level 80/20 split; no model result is asserted by this preparation script.'
}
(OUT / 'random_seed42_sample_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))

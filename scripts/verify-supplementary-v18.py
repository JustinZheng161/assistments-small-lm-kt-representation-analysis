from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]/'data'/'supplementary'
for name in ['manifest-dev-v18.json','manifest-audit-v18.json']:
 p=ROOT/name; r=json.loads(p.read_text()); rows=r['rows']
 assert all('user_id' not in x for x in rows)
 assert all('salt' not in x for x in rows)
 print(name, 'rows=',len(rows), 'records=',len({x['record_index'] for x in rows}))
 if 'fold_id' in rows[0]: print('folds=',sorted({x['fold_id'] for x in rows}))
 else: print('conditions=',sorted({x['condition'] for x in rows}))

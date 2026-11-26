"""Language-neutral requirement states. Only adapter-validated execution counts."""
import hashlib
import json


def requirement_id(quote, capability, selector):
    identity = json.dumps([quote, capability, selector], sort_keys=True)
    return 'R-' + hashlib.sha256(identity.encode()).hexdigest()[:12]


def build_ledger(requirements, evidence):
    rows = []
    for requirement in requirements:
        records = [item for item in evidence if item['requirement_id'] == requirement['id']]
        # Executed tests and deterministic checks (e.g. an offline compile plus an
        # import audit) verify; static hints and model opinions never do.
        executed = [item for item in records if item['type'] in ('executed_test', 'deterministic_check')]
        state = ('failed' if any(item['result'] == 'fail' for item in executed)
                 else 'verified' if any(item['result'] == 'pass' for item in executed)
                 else 'unverified')
        rows.append({**requirement, 'state': state, 'evidence': records})
    verified = sum(row['state'] == 'verified' for row in rows)
    return dict(requirements=rows, total=len(rows), verified=verified,
                failed=sum(row['state'] == 'failed' for row in rows),
                unverified=sum(row['state'] == 'unverified' for row in rows),
                coverage=verified/len(rows) if rows else 0.0,
                coverage_complete=bool(rows) and all(row['state'] != 'unverified' for row in rows),
                all_verified=bool(rows) and verified == len(rows),
                scope='Observed execution evidence on tested inputs; not a universal correctness proof.')

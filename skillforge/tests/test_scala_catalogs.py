"""Independent answer checks for development catalogs; never used by generation."""
import ast
from collections import deque
from functools import lru_cache
import itertools
from pathlib import Path
import re

import pytest

from src.benchmark.loader import load_benchmark


DATA = Path(__file__).resolve().parents[1] / 'data'


def program_answer(task_id, text):
    values = list(map(int, text.split()))
    if 'shortest_paths' in task_id:
        n, m, source = values[:3]
        distances = [[float('inf')] * n for _ in range(n)]
        for i in range(n):
            distances[i][i] = 0
        for i in range(m):
            a, b, weight = values[3 + 3*i:6 + 3*i]
            distances[a][b] = min(distances[a][b], weight)
        for k, i, j in itertools.product(range(n), repeat=3):
            distances[i][j] = min(distances[i][j], distances[i][k] + distances[k][j])
        return ' '.join('-1' if d == float('inf') else str(d) for d in distances[source])
    if 'minimum_coins' in task_id:
        n, total = values[:2]
        best = [0] + [total + 1] * total
        for amount in range(1, total + 1):
            for coin in values[2:]:
                if coin <= amount:
                    best[amount] = min(best[amount], best[amount-coin] + 1)
        return str(best[total] if best[total] <= total else -1)
    if 'weighted_intervals' in task_id:
        jobs = [values[1+3*i:4+3*i] for i in range(values[0])]
        times = sorted({0, *(value for job in jobs for value in job[:2])})
        best = {0: 0}
        for time in times:
            best[time] = max((best[t] for t in best if t <= time), default=0)
            for start, end, profit in jobs:
                if end == time:
                    best[time] = max(best[time], best[start] + profit)
        return str(max(best.values()))
    if 'maximum_rectangle' in task_id:
        rows, columns = values[:2]
        matrix = [values[2+i*columns:2+(i+1)*columns] for i in range(rows)]
        answer = float('-inf')
        for top in range(rows):
            sums = [0] * columns
            for bottom in range(top, rows):
                sums = [a+b for a, b in zip(sums, matrix[bottom])]
                ending = float('-inf')
                for value in sums:
                    ending = max(value, ending + value)
                    answer = max(answer, ending)
        return str(answer)
    if 'dag_paths' in task_id:
        n, m = values[:2]
        adjacency, indegree, counts = [[] for _ in range(n)], [0]*n, [0]*n
        counts[0] = 1
        for i in range(m):
            a, b = values[2+2*i:4+2*i]
            adjacency[a].append(b)
            indegree[b] += 1
        queue = deque(i for i in range(n) if not indegree[i])
        while queue:
            a = queue.popleft()
            for b in adjacency[a]:
                counts[b] += counts[a]
                indegree[b] -= 1
                if not indegree[b]:
                    queue.append(b)
        return str(counts[-1] % 1000000007)
    n, q = values[:2]
    differences = [0] * (n+1)
    for i in range(q):
        left, right, delta = values[2+3*i:5+3*i]
        differences[left] += delta
        differences[right+1] -= delta
    return ' '.join(map(str, itertools.accumulate(differences[:-1])))


def function_answer(task_id, assertion):
    arguments, expected = re.fullmatch(r'assert\(solve\((.*)\) == (\d+)\)', assertion).groups()
    if 'balanced_partition' in task_id:
        values = ast.literal_eval('[' + arguments[5:-1] + ']')
        sums = {0}
        for value in values:
            sums |= {s+value for s in sums}
        answer = min(abs(sum(values)-2*s) for s in sums)
    else:
        strings = ast.literal_eval('[' + arguments + ']')
        if 'subsequence' in task_id:
            sets = []
            for string in strings:
                sequences = {''}
                for char in string:
                    sequences |= {s+char for s in sequences}
                sets.append(sequences)
            answer = max(map(len, set.intersection(*sets)))
        else:
            a, b = strings
            @lru_cache(None)
            def distance(i, j):
                if i == len(a):
                    return len(b)-j
                if j == len(b):
                    return len(a)-i
                return min(1+distance(i+1, j), 1+distance(i, j+1),
                           (a[i] != b[j])+distance(i+1, j+1))
            answer = distance(0, 0)
    return answer, int(expected)


@pytest.mark.parametrize('name', ['scala_algorithms', 'scala_functions', 'scala_http'])
def test_catalogs_load_with_separate_scopes_and_disjoint_suites(name):
    dataset = load_benchmark(DATA / (name + '.json'))
    assert {t.split for t in dataset.tasks} == {'train', 'validation', 'hidden'}
    assert {str(t.language) for t in dataset.tasks} == {'scala'}
    assert len({str(t.execution_mode) for t in dataset.tasks}) == 1
    assert all(t.visible_tests.cases and t.hidden_tests.cases for t in dataset.tasks)


def test_program_answers_against_independent_oracles():
    for task in load_benchmark(DATA / 'scala_algorithms.json').tasks:
        for case in task.visible_tests.cases + task.hidden_tests.cases:
            assert program_answer(task.task_id, case.input) == case.expected_output, task.task_id


def test_function_answers_against_independent_oracles():
    for task in load_benchmark(DATA / 'scala_functions.json').tasks:
        for case in task.visible_tests.cases + task.hidden_tests.cases:
            actual, expected = function_answer(task.task_id, case.assertion)
            assert actual == expected, task.task_id


def test_http_answers_against_reference_state_and_algorithm_properties():
    for task in load_benchmark(DATA / 'scala_http.json').tasks:
        for case in task.visible_tests.cases + task.hidden_tests.cases:
            items, queue, counter = {}, deque(), 0
            for request in case.requests:
                method, path, body = (request[k] for k in ('method', 'path', 'body'))
                status, answer = 404, 'missing'
                if 'store' in task.task_id and path.startswith('/items/'):
                    key = path[7:]
                    if method == 'PUT':
                        items[key], status, answer = body, 204, ''
                    elif method == 'GET':
                        status, answer = (200, items[key]) if key in items else (404, 'missing')
                    elif method == 'DELETE':
                        if key in items:
                            del items[key]
                            status, answer = 204, ''
                    else:
                        status, answer = 405, 'method'
                elif 'counter' in task.task_id:
                    if (method, path) == ('GET', '/counter'):
                        status, answer = 200, str(counter)
                    elif (method, path) == ('DELETE', '/counter'):
                        counter, status, answer = 0, 204, ''
                    elif (method, path) == ('POST', '/counter/add'):
                        try:
                            value = int(body)
                            assert -2**63 <= value < 2**63 and -2**63 <= counter+value < 2**63
                            counter += value
                            status, answer = 200, str(counter)
                        except (ValueError, AssertionError):
                            status, answer = 400, 'invalid'
                elif 'statistics' in task.task_id and path == '/statistics':
                    if method != 'POST':
                        status, answer = 405, 'method'
                    else:
                        try:
                            values = list(map(int, body.split()))
                            assert values and all(abs(v) <= 10**9 for v in values)
                            status, answer = 200, f'{len(values)} {min(values)} {max(values)} {sum(values)}'
                        except (ValueError, AssertionError):
                            status, answer = 400, 'invalid'
                elif 'http_edits' in task.task_id and path == '/distance':
                    if method != 'POST':
                        status, answer = 405, 'method'
                    elif body.count('\n') != 1:
                        status, answer = 400, 'invalid'
                    else:
                        import json
                        assertion = 'assert(solve(' + ','.join(json.dumps(s) for s in body.split('\n')) + ') == 0)'
                        status, answer = 200, str(function_answer('edit_distance', assertion)[0])
                elif 'queue' in task.task_id:
                    if (method, path) == ('POST', '/queue'):
                        queue.append(body)
                        status, answer = 201, str(len(queue))
                    elif (method, path) == ('GET', '/queue'):
                        status, answer = (200, queue.popleft()) if queue else (404, 'empty')
                    elif (method, path) == ('DELETE', '/queue'):
                        queue.clear()
                        status, answer = 204, ''
                    elif (method, path) == ('GET', '/queue/size'):
                        status, answer = 200, str(len(queue))
                elif 'http_intervals' in task.task_id and path == '/schedule':
                    values = list(map(int, body.split()))
                    jobs = [values[1+3*i:4+3*i] for i in range(values[0])]
                    if any(start >= end for start, end, profit in jobs):
                        status, answer = 400, 'invalid'
                    else:
                        status, answer = 200, program_answer('weighted_intervals', body)
                assert (status, answer) == (request['expected_status'], request['expected_body']), task.task_id

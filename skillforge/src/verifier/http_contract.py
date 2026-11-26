"""Task-defined HTTP scenarios; no routes or application behavior are built in."""

from dataclasses import dataclass
import json
import re

MAX_HTTP_REQUESTS = 128


@dataclass
class HTTPTestCase:
    requests: list[dict]
    category: str = "normal"
    clean_environment: bool = False
    port: int | None = None
    # Requirement quotes a trusted (human-written) scenario evidences. Only
    # supplied tests may claim coverage; generated tests never set this.
    requirements: tuple = ()

    @property
    def input(self):
        # Expected answers are deliberately excluded from request identities.
        requests = [{k: v for k, v in r.items() if not k.startswith('expected_')} for r in self.requests]
        value = dict(requests=requests,clean_environment=True,port=self.port) if self.clean_environment else requests
        return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _headers(value, *, expected=False):
    if not isinstance(value, dict) or len(value) > 30:
        raise ValueError("HTTP headers must be an object with at most 30 entries")
    result = {}
    for key, text in value.items():
        if (not isinstance(key, str) or not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", key)
                or not isinstance(text, str) or len(text) > 4096
                or any(ord(c) < 32 or ord(c) > 126 for c in text)):
            raise ValueError("HTTP headers must contain valid ASCII names and values")
        name = key.lower()
        if name in result:
            raise ValueError("duplicate HTTP header name")
        if not expected and name in {"host", "content-length", "transfer-encoding", "connection"}:
            raise ValueError(f"HTTP header {name} is controlled by the harness")
        result[name] = text
    return result


def build_http_case(item):
    if set(item) - {"requests", "category", "clean_environment", "port", "requirements"}:
        raise ValueError("HTTP cases accept only requests and category")
    requests = item.get("requests")
    if not isinstance(requests, list) or not 1 <= len(requests) <= MAX_HTTP_REQUESTS:
        raise ValueError(f"HTTP scenario requires 1 to {MAX_HTTP_REQUESTS} requests")
    normalized = []
    for request in requests:
        allowed = {"method", "path", "body", "headers", "expected_status", "expected_body", "expected_headers", "expected_body_one_of", "parallel_group"}
        if not isinstance(request, dict) or set(request) - allowed:
            raise ValueError("invalid HTTP request fields")
        if not {"method", "path", "expected_status", "expected_body"} <= request.keys():
            raise ValueError("HTTP request needs method, path, expected_status, and expected_body")
        method, path = request["method"], request["path"]
        if not isinstance(method, str) or method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}:
            raise ValueError("unsupported HTTP method")
        if (not isinstance(path, str) or not path.startswith('/') or path.startswith('//')
                or '#' in path or len(path) > 4096 or any(ord(c) <= 32 or ord(c) > 126 for c in path)):
            raise ValueError("HTTP path must be a local ASCII path; percent-encode Unicode")
        status = request["expected_status"]
        if type(status) is not int or not 100 <= status <= 599:
            raise ValueError("expected_status must be an HTTP status integer")
        body, expected = request.get("body", ""), request["expected_body"]
        if (not isinstance(body, str) or len(body.encode('utf-8')) > 1048576
                or (expected is not None and (not isinstance(expected, str) or len(expected.encode('utf-8')) > 1048576))):
            raise ValueError("HTTP bodies must be UTF-8 strings up to 1 MiB")
        normalized.append(dict(method=method, path=path, body=body,
                               headers=_headers(request.get("headers", {})),
                               expected_status=status, expected_body=expected,
                               expected_headers=_headers(request.get("expected_headers", {}), expected=True)))
        alternatives=request.get('expected_body_one_of')
        if alternatives is not None:
            if (expected is not None or not isinstance(alternatives,list) or not 1<=len(alternatives)<=16
                    or any(not isinstance(v,str) or len(v.encode('utf-8'))>1048576 for v in alternatives)):
                raise ValueError('expected_body_one_of requires 1 to 16 strings and expected_body null')
            normalized[-1]['expected_body_one_of']=alternatives
        group=request.get('parallel_group',0)
        if type(group) is not int or not 0 <= group <= 16:
            raise ValueError('parallel_group must be an integer from 0 to 16')
        if group:normalized[-1]['parallel_group']=group
    category = item.get("category", "normal")
    if category not in ("normal", "edge", "stress"):
        raise ValueError("HTTP category must be normal, edge, or stress")
    clean=item.get('clean_environment',False);port=item.get('port')
    if type(clean) is not bool or (port is not None and (type(port) is not int or not 1 <= port <= 65535)):
        raise ValueError('clean_environment must be boolean and port a valid integer')
    if clean and port is None:raise ValueError('clean environment scenarios require the task-specified port')
    groups=[r.get('parallel_group',0) for r in normalized]
    for group in set(groups)-{0}:
        positions=[i for i,g in enumerate(groups) if g==group]
        if len(positions)>16 or positions!=list(range(positions[0],positions[-1]+1)):
            raise ValueError('parallel groups must be contiguous and contain at most 16 requests')
    covers = item.get('requirements') or []
    if (not isinstance(covers, (list, tuple)) or len(covers) > 16
            or any(not isinstance(q, str) or not q.strip() or len(q) > 500 for q in covers)):
        raise ValueError('requirements must be a list of up to 16 requirement quotes')
    return HTTPTestCase(normalized, category, clean, port, tuple(q.strip() for q in covers))

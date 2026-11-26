import json

from src.agents.task_spec import (
    _parse_stdio_contracts, _resolve_stdio_contracts, stated_format_sections,
)
from src.agents.test_generator import TestCase, TestSuite
from src.verifier.diagnostics import leaked_representation, shape_failure


GRID = ("Count paths.\n\nConstraints: 1 <= N.\n\n"
        "Input format: The first line contains N M K. The next N lines each contain "
        "exactly M binary digits without spaces, one row per line.\n"
        "Output format: Print one integer, followed by a newline.")


def _contract(explicit, layout):
    return dict(explicit_in_statement=explicit, quoted_rules='', serialization='plain text',
                layout=layout, whitespace='trim', empty_values='not allowed')


def test_labelled_sections_are_read_verbatim():
    sections = stated_format_sections(GRID)
    assert sections['input'].endswith('one row per line.')
    assert sections['output'] == 'Print one integer, followed by a newline.'
    assert all(text in GRID for text in sections.values())


def test_markdown_headings_stop_at_the_next_section():
    text = ("## Input Format\nThe first line has n.\nThe second line has n integers.\n\n"
            "## Output Format\nOutput one integer per line.\n\n## Constraints\n1<=n")
    assert stated_format_sections(text) == {
        'input': 'The first line has n.\nThe second line has n integers.',
        'output': 'Output one integer per line.'}


def test_sample_lines_are_not_mistaken_for_formats():
    assert stated_format_sections('Add two numbers.\nExample\nInput: 1 2\nOutput: 3') == {}


def test_false_not_stated_claim_cannot_replace_a_labelled_format():
    raw = json.dumps(dict(stdin=_contract(False, 'line per row, space-separated integers'),
                          stdout=_contract(False, 'one integer')))
    result = _parse_stdio_contracts(raw, GRID)
    assert result['input_format_origin'] == 'stated'
    assert 'without spaces' in result['input_format']
    assert 'space-separated integers' not in result['input_format']


def test_fully_labelled_statement_needs_no_contract_model_call():
    class Client:
        task_format_output_budget = 512
        def generate_json(self, *args, **kwargs):
            raise AssertionError('formats are stated; no model call expected')
    result = _resolve_stdio_contracts(Client(), GRID, 'scala', 3)
    assert result['output_format'] == 'Print one integer, followed by a newline.'


def test_runtime_object_representations_are_reported_without_a_reference():
    assert leaked_representation('[J@1b6d3586\n') == '[J@1b6d3586'
    assert leaked_representation('<__main__.Node object at 0x10f3a2b50>')
    assert leaked_representation('42\n') is None
    assert 'object representation' in shape_failure('[J@1b6d3586\n', None)


def test_layout_mismatch_hides_the_reference_value():
    reason = shape_failure('1 2 3\n', '987654\n')
    assert reason and '987654' not in reason
    assert shape_failure('5\n', '987654\n') is None
    assert shape_failure('3.0\n', '3\n') is None


def test_text_answers_written_by_a_model_are_not_layout_evidence(monkeypatch):
    from src.verifier import diagnostics
    from src.verifier.sandbox import Status

    class Run:
        status, stdout, stderr, exit_code = Status.PASS, '5\n', '', 0
    monkeypatch.setattr(diagnostics, 'run_code_many', lambda code, inputs, **kw: [Run()])
    suite = TestSuite([TestCase('1 2', 'wrong answer', 'normal')])
    evidence, _ = diagnostics.diagnose_candidate('print(5)', suite=suite)
    assert evidence.failures == []
    suite.reference_outputs = True
    evidence, _ = diagnostics.diagnose_candidate('print(5)', suite=suite)
    assert evidence.failures[0]['status'] == 'output_shape_mismatch'


def _line_edit(source, start, end, replacement):
    import hashlib
    return dict(diagnosis='fix', source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                edits=[dict(start_line=start, end_line=end, replacement=replacement)])


SCALA_SOURCE = """object Main {
  def main(args: Array[String]): Unit = {
    val Array(n, m) = readLine().split(" ").map(_.toInt)
    val total = n + m
    val doubled = total * 2
    println(doubled)
  }
}"""


def test_line_edit_that_rewrites_following_kept_lines_is_rejected():
    import pytest
    from src.agents.source_edits import apply_line_edits
    whole_method = "\n".join(SCALA_SOURCE.splitlines()[1:7])
    with pytest.raises(ValueError, match='duplicate code'):
        apply_line_edits(SCALA_SOURCE, _line_edit(SCALA_SOURCE, 2, 2, whole_method))


def test_ordinary_block_replacement_is_still_applied():
    from src.agents.source_edits import apply_line_edits
    fixed = apply_line_edits(SCALA_SOURCE, _line_edit(
        SCALA_SOURCE, 3, 5, "val Array(n, m) = readLine().trim.split(\" +\").map(_.toInt)\n"
        "val total = n + m\nval doubled = total * 2"), relative_indentation=True)
    assert fixed.count('val total') == 1 and 'trim.split' in fixed


def test_json_decoded_newline_inside_a_literal_is_restored_to_an_escape():
    from src.agents.source_edits import apply_line_edits
    source = 'object Main {\n  val parts = input.split(" ")\n}'
    fixed = apply_line_edits(source, _line_edit(source, 2, 2, "val parts = input.split('\n')"),
                             relative_indentation=True)
    assert "split('\\n')" in fixed and fixed.count('\n') == source.count('\n')


def test_literal_newline_restoration_leaves_code_comments_and_triple_quotes_alone():
    from src.agents.source_edits import restore_literal_newlines
    for text in ("val x = 1\nval y = 2", "// don't split\nval c = 'a'",
                 'val q = """multi\nline"""', "# it's python\nx = '\\n'"):
        assert restore_literal_newlines(text) == text

from src.agents.compiler_quick_fixes import compiler_quick_fix, scala_uppercase_pattern_fix

SOURCE = '''object Main {
  def main(args: Array[String]): Unit = {
    val Array(S, T) = scala.io.Source.stdin.mkString.split("\\n")
    println(solve(S, T) + " S T")  // S and T in a comment
  }
  def solve(S: String, T: String): String = S + T
}'''
ERRORS = 'Not found: S\nNot found: T'


def test_capitalised_pattern_bindings_are_renamed_everywhere_but_literals_and_comments():
    fixed, renames = scala_uppercase_pattern_fix(SOURCE, ERRORS)
    assert renames == {'S': 's', 'T': 't'}
    assert 'val Array(s, t)' in fixed and 'def solve(s: String, t: String): String = s + t' in fixed
    assert '" S T"' in fixed and '// S and T in a comment' in fixed


def test_rename_avoids_existing_lowercase_names():
    source = 'val s = 1\nval Array(S) = Array(2)\nprintln(S + s)'
    fixed, renames = scala_uppercase_pattern_fix(source, 'Not found: S')
    assert renames == {'S': 's2'} and 'println(s2 + s)' in fixed


def test_no_fix_without_a_matching_compiler_error_or_pattern():
    assert scala_uppercase_pattern_fix(SOURCE, 'Type Mismatch Error') is None
    assert scala_uppercase_pattern_fix('val x = Foo + 1', 'Not found: Foo') is None
    assert compiler_quick_fix(SOURCE, ERRORS, language='python') is None


def test_one_line_read_diagnosis_also_covers_destructuring_match_errors():
    from src.agents.refinement_agent import _scala_parsing_diagnosis
    code = ('object Main {\n  def main(a: Array[String]) = {\n    val input = scala.io.StdIn.readLine()\n'
            '    val Array(s, t) = input.split("\\n")\n  }\n}')
    crash = 'Exception in thread "main" scala.MatchError: [Ljava.lang.String;@43a25848'
    assert 'consumed only the first input line' in _scala_parsing_diagnosis(
        [{'stderr': crash}], 'scala', 'stdio', code)


def test_scala_missing_map_key_crash_gets_a_lookup_hint():
    from src.agents.refinement_agent import _scala_parsing_diagnosis
    crash = 'Exception in thread "main" java.util.NoSuchElementException: key not found: D\n\tat scala...'
    hint = _scala_parsing_diagnosis([{'stderr': crash}], 'scala', 'stdio', '')
    assert 'key not found: D' in hint and 'getOrElse' in hint


def test_one_line_read_diagnosis_covers_list_index_errors_and_char_splits():
    from src.agents.refinement_agent import _scala_parsing_diagnosis
    code = "object Main extends App {\n  val lines = scala.io.StdIn.readLine().split('\\n').toList\n  val t = lines(1)\n}"
    crash = 'Caused by: java.lang.IndexOutOfBoundsException: 1\n\tat scala.collection.LinearSeqOps.apply'
    assert 'consumed only the first input line' in _scala_parsing_diagnosis(
        [{'stderr': crash}], 'scala', 'stdio', code)


MAP_SOURCE = '''object Main {
  def main(args: Array[String]): Unit = {
    val counts = "abca".groupBy(identity).view.mapValues(_.length).toMap
    val seen = Map[Char, Int]()
    counts('a') -= 1
    seen.update('b', 1)
  }
}'''
MAP_ERRORS = '''-- [E008] Not Found Error: /tmp/x/Candidate.scala:5:4
5 |    counts('a') -= 1
  |    ^^^^^^^^^^^^^^^^
  |value -= is not a member of Int - did you mean Int.!=?
-- [E008] Not Found Error: /tmp/x/Candidate.scala:6:4
6 |    seen.update('b', 1)
  |    ^^^^^^^^^^^
  |value update is not a member of scala.collection.immutable.Map[Char, Int]'''


def test_in_place_updates_on_an_immutable_map_make_its_declaration_mutable():
    from src.agents.compiler_quick_fixes import scala_immutable_map_fix
    fixed, names = scala_immutable_map_fix(MAP_SOURCE, MAP_ERRORS)
    assert names == ['counts', 'seen']
    assert '.view.mapValues(_.length).to(scala.collection.mutable.Map)' in fixed
    assert 'val seen = scala.collection.mutable.Map[Char, Int]()' in fixed
    assert "counts('a') -= 1" in fixed  # the update itself is untouched


def test_immutable_map_fix_ignores_unrelated_errors():
    from src.agents.compiler_quick_fixes import scala_immutable_map_fix
    assert scala_immutable_map_fix(MAP_SOURCE, 'Type Mismatch Error') is None


def test_immutable_map_fix_handles_nested_key_expressions():
    from src.agents.compiler_quick_fixes import scala_immutable_map_fix
    source = 'val s = "ab"\nval charCount = s.groupBy(identity).view.mapValues(_.size).toMap\ncharCount(s(0)) -= 1'
    errors = '-- [E008] Not Found Error: /t/Candidate.scala:3:0\n3 | x\n  |value -= is not a member of Int'
    fixed, names = scala_immutable_map_fix(source, errors)
    assert names == ['charCount'] and '.to(scala.collection.mutable.Map)' in fixed


def test_map_type_mismatch_converts_only_the_flagged_expression():
    from src.agents.compiler_quick_fixes import scala_map_mismatch_fix
    source = 'object Main {\n  def f(): Either[String, M] =\n        case scala.util.Success(map) => Right(map)\n}'
    errors = ('-- [E007] Type Mismatch Error: /t/Candidate.scala:3:46 \n'
              '3 |        case scala.util.Success(map) => Right(map)\n'
              '  |                                              ^^^\n'
              '  |         Found:    (map : Map[String, String])\n'
              '  |         Required: scala.collection.mutable.Map²[String, String]\n')
    fixed, count = scala_map_mismatch_fix(source, errors)
    assert count == 1
    assert 'Success(map) => Right(map.to(scala.collection.mutable.Map))' in fixed


def test_nullable_java_map_match_wrapped_in_option():
    from src.agents.compiler_quick_fixes import scala_nullable_match_fix
    source = (
        "object Main {\n"
        "  val store = new java.util.concurrent.ConcurrentHashMap[String, String]()\n"
        "  def f(form: Map[String, String]): String = {\n"
        "    store.get(\"t\") match {\n"
        "      case Some(t) =>\n"
        "        form.get(\"k\") match {\n"
        "          case Some(v) => v\n"
        "          case None => \"\"\n"
        "        }\n"
        "      case None => \"missing\"\n"
        "    }\n"
        "  }\n"
        "}\n"
    )
    output = (
        "-- [E007] Type Mismatch Error: /tmp/x/Candidate.scala:10:11 \n"
        "10 |      case None => \"missing\"\n"
        "   |           ^^^^\n"
        "   |           Found:    None.type\n"
        "   |           Required: String\n"
    )
    fixed, count = scala_nullable_match_fix(source, output)
    assert count == 1
    assert 'Option(store.get("t")) match {' in fixed
    assert 'form.get("k") match {' in fixed and 'Option(form.get' not in fixed


def test_nullable_fix_ignores_option_requirements():
    from src.agents.compiler_quick_fixes import scala_nullable_match_fix
    output = ("-- [E007] Type Mismatch Error: /tmp/x/Candidate.scala:2:11 \n"
              "   |           Found:    None.type\n   |           Required: Option[Int]\n")
    assert scala_nullable_match_fix("x.get(1) match {\n  case None => 0\n}", output) is None


def test_replace_all_with_function_becomes_replace_all_in():
    from src.agents.compiler_quick_fixes import scala_replace_all_function_fix
    source = (
        "object Main {\n"
        "  val p = \"\"\"\\{\\{(\\w+)\\}\\}\"\"\".r\n"
        "  def render(t: String, v: Map[String, String]): String = {\n"
        "    t.replaceAll(p.toString, { m =>\n"
        "      v.getOrElse(m.group(1), \"\")\n"
        "    })\n"
        "  }\n"
        "}\n"
    )
    output = ("-- [E081] Type Error: /tmp/x/Candidate.scala:4:31 \n"
              "4 |    t.replaceAll(p.toString, { m =>\n"
              "  |                               ^\n"
              "  |                               Could not infer type for parameter m of anonymous function\n")
    fixed, count = scala_replace_all_function_fix(source, output)
    assert count == 1
    assert "(p.toString).r.replaceAllIn(t, (m: scala.util.matching.Regex.Match) =>" in fixed
    assert "Regex.quoteReplacement {" in fixed and 'v.getOrElse(m.group(1), "")' in fixed
    assert ".replaceAll(" not in fixed


def test_java_regex_literal_braces_are_escaped():
    from src.agents.compiler_quick_fixes import scala_regex_brace_fix
    fixed, count = scala_regex_brace_fix('val p = """{{(\\w+)}}""".r\nval q = """a{3}""".r\nval s = "{{x}}"')
    assert count == 1
    assert fixed.startswith('val p = """\\{\\{(\\w+)\\}\\}""".r')
    assert 'val q = """a{3}""".r' in fixed and 'val s = "{{x}}"' in fixed
    assert scala_regex_brace_fix('val p = """\\{\\{(\\w+)\\}\\}""".r') is None


def test_quote_replacement_wraps_replace_all_in_function():
    from src.agents.compiler_quick_fixes import scala_quote_replacement_fix
    source = 'val out = pattern.replaceAllIn(text, m => {\n  val k = m.group(1)\n  values(k)\n})'
    fixed, count = scala_quote_replacement_fix(source)
    assert count == 1
    assert 'm => scala.util.matching.Regex.quoteReplacement {\n  val k = m.group(1)' in fixed
    assert fixed.count('{') == fixed.count('}')


def test_quote_replacement_leaves_quoted_and_string_replacements_alone():
    from src.agents.compiler_quick_fixes import scala_quote_replacement_fix
    assert scala_quote_replacement_fix('p.replaceAllIn(t, "$1")') is None
    assert scala_quote_replacement_fix('p.replaceAllIn(t, m => Regex.quoteReplacement(v(m)))') is None
    assert scala_quote_replacement_fix('val s = "p.replaceAllIn(t, m => x)"') is None


PRIVATE_ERROR = '''-- [E173] Reference Error: /tmp/Candidate.scala:12:49
12 |    val response = s"{${parser.updatedCount}}"
   |                        ^^^^^^^^^^^^^^^^^^^
   |variable updatedCount cannot be accessed as a member of (parser : HtmlParser) from class Handler.
   |  private variable updatedCount can only be accessed from class HtmlParser.
'''


def test_private_member_read_elsewhere_becomes_visible():
    from src.agents.compiler_quick_fixes import compiler_quick_fix
    source = 'class HtmlParser {\n  private var updatedCount = 0\n  private val other = 1\n}'
    fixed, note = compiler_quick_fix(source, PRIVATE_ERROR, language='scala')
    assert '  var updatedCount = 0' in fixed and 'private val other' in fixed
    assert 'updatedCount' in note


LIST_ERROR = '''-- [E007] Type Mismatch Error: /tmp/Candidate.scala:2:4
2 |    elements.filter(_.ok).map(_.text)
  |    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |    Found:    scala.collection.mutable.ListBuffer[String]
  |    Required: List[String]
'''


def test_mutable_buffer_converted_where_list_required():
    from src.agents.compiler_quick_fixes import compiler_quick_fix
    source = 'def texts: List[String] =\n    elements.filter(_.ok).map(_.text)'
    fixed, _ = compiler_quick_fix(source, LIST_ERROR, language='scala')
    assert fixed.endswith('    (elements.filter(_.ok).map(_.text)).toList')
    unrelated = LIST_ERROR.replace('Required: List[String]', 'Required: Int')
    assert compiler_quick_fix(source, unrelated, language='scala') is None

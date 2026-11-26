import zipfile
from src.agents.sdk_hints import compiler_import_hints


def test_sdk_lookup_preserves_sibling_imports_without_a_fixed_symbol_table(tmp_path):
    directory=tmp_path/'jmods';directory.mkdir()
    with zipfile.ZipFile(directory/'java.base.jmod','w') as archive:
        for name in ('Gadget','Widget'):
            archive.writestr('classes/java/example/'+name+'.class',b'fixture')
        archive.writestr('classes/internal/Hidden.class',b'fixture')
    hints=compiler_import_hints(['Not found: type Gadget'],
        'import java.example.Widget\nval a = new Gadget(); val b = new Widget()',tmp_path)
    assert 'java.example.Gadget' in hints and 'java.example.Widget' in hints
    assert 'Preserve imports for ALL' in hints
    assert 'Hidden' not in hints
    assert compiler_import_hints(['Not found: type Undefined'],'new Undefined()',tmp_path)==''

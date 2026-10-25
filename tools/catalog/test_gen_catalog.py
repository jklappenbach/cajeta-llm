"""Tests for gen-catalog.py.

    python3 -m unittest discover -s tools/catalog

The row test regenerates the page from this checkout's inputs and compares
every entry (section, kernel, source, counterpart, census, role) with the
committed page in the cajeta repository (CATALOG_PAGE, default
../cajeta/docs/kernels/catalog-llm.md). It reads both the old markdown-table
page and the HTML-table page the generator writes now, so it pinned the
generator against the 2026-10-24 page and, since, tells a stale page from a
current one.
"""
import html
import importlib.util
import os
import re
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PAGE = os.environ.get('CATALOG_PAGE',
                      os.path.join(os.path.dirname(REPO), 'cajeta', 'docs', 'kernels', 'catalog-llm.md'))

spec = importlib.util.spec_from_file_location('gen_catalog', os.path.join(HERE, 'gen-catalog.py'))
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

NOT_REGISTERED = None


def split_md_row(line):
    cells = re.split(r'(?<!\\)\|', line.strip())[1:-1]
    return [c.strip().replace('\\|', '|') for c in cells]


def unhtml(s):
    s = re.sub(r'<code>(.*?)</code>', r'`\1`', s)
    s = re.sub(r'<[^>]+>', '', s)
    return html.unescape(s)


def parse_page(text):
    """section title -> list of entries {kernel, source, counterpart, census, role}; the
    standard-library section is a list of names."""
    sections = {}
    order = []
    current = None
    for line in text.split('\n'):
        m = re.match(r'## (.*) \((\d+)\)$', line)
        if m:
            current = m.group(1)
            sections[current] = []
            order.append(current)
            continue
        if current is None:
            continue
        if line.startswith('| `'):                      # old markdown table row
            cells = split_md_row(line)
            name = cells[0].strip('`')
            if len(cells) == 5:
                census = tuple(NOT_REGISTERED if c.strip() == '–' else c.strip()
                               for c in cells[3].split('/'))
                e = dict(kernel=name, source=cells[1].strip('`'), counterpart=cells[2],
                         census=census, role=cells[4])
            elif cells[1].startswith('`') or cells[1] == '':   # probes: kernel, source, role
                e = dict(kernel=name, source=cells[1].strip('`'), role=cells[2])
            else:                                       # fixtures: kernel, where, standing
                e = dict(kernel=name, where=cells[1], role=cells[2])
            sections[current].append(e)
        elif line.startswith('`') and current.startswith('Standard-library'):
            sections[current] = [n.strip(' `') for n in line.split(',')]
    # the HTML tables
    for title in order:
        pass
    for m in re.finditer(r'## (.*?) \(\d+\)\n(.*?)(?=\n## |\Z)', text, re.S):
        title, body = m.group(1), m.group(2)
        if '<table' not in body:
            continue
        head = re.search(r'<thead>(.*?)</thead>', body, re.S).group(1)
        cols = [unhtml(h) for h in re.findall(r'<th[^>]*>(.*?)</th>', head)]
        rows = re.findall(r'<tr class="(kernel|kernel-role)">(.*?)</tr>', body, re.S)
        entries = []
        for (cls, tr), nxt in zip(rows[0::2], rows[1::2]):
            assert cls == 'kernel' and nxt[0] == 'kernel-role', (title, tr)
            tds = re.findall(r'<td([^>]*)>(.*?)</td>', tr, re.S)
            name = unhtml(tds[0][1]).strip("`")
            link = re.search(r'title="([^"]*):(\d+)"', tds[0][1])
            e = dict(kernel=name, role=unhtml(re.search(r'<td[^>]*>(.*?)</td>', nxt[1], re.S).group(1)))
            if cols[:2] == ['kernel', 'where']:
                e['where'] = unhtml(tds[1][1])
            else:
                e['source'] = html.unescape(link.group(1)) if link else ''
            if 'llama.cpp counterpart' in cols:
                e['counterpart'] = unhtml(tds[1][1])
                e['census'] = tuple(NOT_REGISTERED if html.unescape(re.search(r'title="([^"]*)"', a).group(1))
                                    == 'not registered' else html.unescape(re.search(r'title="([^"]*)"', a).group(1))
                                    for a, _ in tds[2:5])
            entries.append(e)
        sections[title] = entries
    return order, sections


class CatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = os.path.join(cls.tmp.name, 'catalog-llm.md')
        gen.main([cls.out, '--date', '2026-10-24'])
        with open(cls.out, encoding='utf-8') as f:
            cls.page = f.read()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_rows_match_committed_page(self):
        if not os.path.exists(PAGE):
            self.skipTest('no committed page at ' + PAGE)
        with open(PAGE, encoding='utf-8') as f:
            want_order, want = parse_page(f.read())
        got_order, got = parse_page(self.page)
        self.assertEqual(want_order, got_order)
        for title in want_order:
            w, g = want[title], got[title]
            self.assertEqual([e if isinstance(e, str) else e['kernel'] for e in w],
                             [e if isinstance(e, str) else e['kernel'] for e in g], title)
            for we, ge in zip(w, g):
                self.assertEqual(we, ge, '%s: %s' % (title, we if isinstance(we, str) else we['kernel']))

    def test_no_markdown_table(self):
        pipe_rows = [l for l in self.page.split('\n') if re.match(r'\s*\|.*\|\s*$', l)]
        self.assertEqual(pipe_rows, [], 'the page must carry no markdown table')

    def test_library_tables_have_five_header_cells(self):
        for head in re.findall(r'<thead>(.*?)</thead>', self.page, re.S):
            cols = re.findall(r'<th[^>]*>(.*?)</th>', head)
            self.assertLessEqual(len(cols), 5)
        heads = [re.findall(r'<th[^>]*>(.*?)</th>', h) for h in re.findall(r'<thead>(.*?)</thead>', self.page, re.S)]
        self.assertIn(['kernel', 'llama.cpp counterpart', 'cuda', 'cpu', 'hip'], heads)

    def test_every_kernel_row_is_followed_by_its_role(self):
        rows = re.findall(r'<tr class="([^"]*)"', self.page)
        self.assertTrue(rows)
        for i, cls in enumerate(rows):
            if cls == 'kernel':
                self.assertEqual(rows[i + 1], 'kernel-role', i)
            else:
                self.assertEqual(rows[i - 1], 'kernel', i)

    def test_no_blank_line_inside_a_table(self):
        # a blank line ends a markdown HTML block, and the rest would render as text
        for t in re.findall(r'<table.*?</table>', self.page, re.S):
            self.assertNotIn('\n\n', t)

    def test_kernel_links_point_at_a_source_line(self):
        for href in re.findall(r'href="([^"]*)"', self.page):
            if 'github.com' in href:
                self.assertRegex(href, r'^https://github\.com/[^/]+/cajeta-llm/blob/main/src/(main|test)/cajeta/.*\.cajeta#L\d+$')


class DocTest(unittest.TestCase):
    def test_doc_skips_annotations_and_line_comments(self):
        lines = ['    /** The role. */', '    @Kernel', '    // a note on the bound', '    @Occupancy(maxThreads = 256)',
                 '    public static void k(KernelBuffer<float32> y) {']
        self.assertEqual(gen.doc_before(lines, 4), 'The role.')

    def test_no_doc_after_a_closing_brace(self):
        self.assertEqual(gen.doc_before(['    }', '', '    @Kernel', '    public static void k() {'], 3), '')

    def test_links_and_code_are_plain_text(self):
        self.assertEqual(gen.clean_doc(['/** {@link #a} with `b` and {@code c}. */']), '#a with b and c.')

    def test_first_sentence_is_whole_however_long(self):
        self.assertEqual(gen.first_sentence('One. Two.'), 'One.')
        long = 'x' * 400 + '.'
        self.assertEqual(gen.first_sentence(long + ' More.'), long)
        self.assertNotIn('\u2026', gen.first_sentence(long))

    def test_the_note_follows_every_documented_role(self):
        model = gen.build(REPO, os.path.join(HERE, 'retired.tsv'))
        notes = {r['kernel']: r['note'] for r in gen.read_tsv(os.path.join(REPO, 'census/oracle-map.tsv'))}
        for row in model['library']:
            note = notes[row['kernel']]
            if note:
                self.assertTrue(row['role'] == note or row['role'].endswith('(%s)' % note), row['kernel'])
            self.assertNotIn('\u2026', row['role'], row['kernel'])


if __name__ == '__main__':
    unittest.main()

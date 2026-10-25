#!/usr/bin/env python3
"""Generate the cajeta.dev kernel catalog page for dev.cajeta.llm.

    python3 tools/catalog/gen-catalog.py ../cajeta/docs/kernels/catalog-llm.md

Inputs, all in this repository:
  - census/oracle-map.tsv      the library kernels: family, llama.cpp counterpart, note
  - census/{nvptx,cpu,amdgpu}.tsv  the committed census tables (cuda, cpu, hip)
  - src/{main,test}/cajeta/dev/cajeta/llm/**.cajeta  each kernel's declaration and doc
  - tools/catalog/retired.tsv  kernels a census table still carries but the source dropped

The page is markdown whose tables are raw HTML, in a shape that cannot wrap
at the site's prose width: three grid columns (kernel linked to its source
line, llama.cpp counterpart, the census as three one-mark sub-columns) and the
kernel's role on its own full-width row underneath.
"""
import argparse
import datetime
import html
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PKG = 'dev/cajeta/llm'
SRC_ROOTS = [('', 'src/main/cajeta/' + PKG), ('test:', 'src/test/cajeta/' + PKG)]
BACKENDS = [('cuda', 'nvptx'), ('cpu', 'cpu'), ('hip', 'amdgpu')]

FAMILIES = [
    ('wave', 'Wave mat-vecs: decode, one row per wave (llama.cpp mmvq)'),
    ('batch', 'Batch GEMMs on the integer dot product'),
    ('coop', 'Cooperative-matrix GEMMs and tiles: prefill and expert GEMMs (llama.cpp mmq, mul_mat_id)'),
    ('scalar', 'Scalar mat-vecs: the host-order reference shape'),
    ('utility', 'Attention, router, mixture-of-experts plumbing, relayout and element-wise kernels'),
]

# The census marks: one glyph per class, the full class in the cell's title.
MARKS = {
    'CHECKED': '✓',
    'RAN-UNCHECKED': '○',
    'SKIP': '–',
    'DECLINED': '✗',
    None: '',
}
ROLE_MAX = 180          # a first sentence longer than this is cut to ROLE_MAX - 3 and an ellipsis
ROLE_WITH_NOTE_MAX = 264  # the oracle-map note rides in parentheses only while the role stays this short
TEMPLATE_INSTANCE = re.compile(r'^([A-Z][A-Za-z0-9]*Tile)_((?:\d+_)+)([a-z][A-Za-z0-9]*)$')


def read_tsv(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        header = None
        for line in f:
            line = line.rstrip('\n')
            if not line or line.startswith('#'):
                continue
            cells = line.split('\t')
            if header is None:
                header = cells
                continue
            cells += [''] * (len(header) - len(cells))
            rows.append(dict(zip(header, cells)))
    return rows


# ---------------------------------------------------------------- sources

def source_files(repo):
    for prefix, root in SRC_ROOTS:
        base = os.path.join(repo, root)
        found = []
        for d, _, files in os.walk(base):
            for fn in files:
                if fn.endswith('.cajeta'):
                    found.append(os.path.relpath(os.path.join(d, fn), base))
        # a kernel name declared twice (a library kernel and its bench probe twin)
        # resolves to the library's own package first
        for shown in sorted(found, key=lambda p: (p.startswith('bench/'), p)):
            full = os.path.join(base, shown)
            yield prefix, shown, os.path.relpath(full, repo), full


def clean_doc(lines):
    """A comment block's text as one line: markers, {@link}, {@code} and backticks removed."""
    text = []
    for ln in lines:
        s = ln.strip()
        s = re.sub(r'^/\*+', '', s)
        s = re.sub(r'\*+/$', '', s)
        s = re.sub(r'^\*+(?!/)', '', s)
        text.append(s.strip())
    t = ' '.join(x for x in text if x)
    t = re.sub(r'\{@link(?:plain)?\s+([^}\s]+)(?:\s+([^}]*))?\}', lambda m: m.group(2) or m.group(1), t)
    t = re.sub(r'\{@code\s+([^}]*)\}', r'\1', t)
    t = t.replace('`', '')
    return re.sub(r'\s+', ' ', t).strip()


def doc_before(lines, i):
    """The block comment that documents the declaration at line i. It must sit
    directly above the declaration's annotations; a line comment among the
    annotations is skipped, but one between the block comment and the first
    annotation is the nearer comment, and then the kernel has no doc."""
    j = i - 1
    top = i                                   # the first line of the annotation block
    while j >= 0:
        s = lines[j].strip()
        if s.startswith('@'):
            top = j
        elif not (s.startswith('//') or s == ''):
            break
        j -= 1
    j = top - 1
    if j < 0 or not lines[j].strip().endswith('*/'):
        return ''
    end = j
    while j >= 0 and '/*' not in lines[j]:
        j -= 1
    if j < 0:
        return ''
    return clean_doc(lines[j:end + 1])


def scan_kernels(repo):
    """name -> (shown path, repo path, line, doc) for every @Kernel method; a kernel
    method of a template tile class is also keyed Class::method, since its
    instances are named Class_<shape>_method."""
    found = {}
    decl = re.compile(r'\bvoid\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(')
    for prefix, shown, rel, full in source_files(repo):
        with open(full, encoding='utf-8') as f:
            text = f.read()
        lines = text.split('\n')
        cls = os.path.splitext(os.path.basename(full))[0]
        template = re.search(r'\bclass\s+%s\s*<' % re.escape(cls), text) is not None
        for i, ln in enumerate(lines):
            if not re.match(r'\s*@Kernel\b', ln):
                continue
            for k in range(i, min(i + 40, len(lines))):
                m = decl.search(lines[k])
                if m:
                    entry = (prefix + shown, rel, k + 1, doc_before(lines, k))
                    found.setdefault(m.group(1), entry)
                    if template:
                        found.setdefault('%s::%s' % (cls, m.group(1)), entry)
                    break
    return found


def first_sentence(text):
    m = re.match(r'(.*?[.!?])(?=\s|$)', text)
    s = m.group(1) if m else text
    if len(s) > ROLE_MAX:
        s = s[:ROLE_MAX - 3] + '\u2026'
    return s


def locate(name, kernels):
    """(shown source, repo path, line, role-from-doc) for a registered kernel, or None."""
    if name in kernels:
        shown, rel, line, doc = kernels[name]
        return shown, rel, line, first_sentence(doc) if doc else ''
    m = TEMPLATE_INSTANCE.match(name)
    if m:
        cls, args, method = m.group(1), m.group(2).strip('_').split('_'), m.group(3)
        member = kernels.get('%s::%s' % (cls, method))
        if member:
            shown, rel, line, _ = member
            role = '%s<%s>: a member of the tile family, one body written over its shape' % (
                cls, ', '.join(args))
            return shown, rel, line, role
    return None


# ---------------------------------------------------------------- model

def build(repo, retired_path):
    oracle = read_tsv(os.path.join(repo, 'census/oracle-map.tsv'))
    census = {}
    for col, fname in BACKENDS:
        for r in read_tsv(os.path.join(repo, 'census', fname + '.tsv')):
            census.setdefault(r['kernel'], {})[col] = r['class']
    kernels = scan_kernels(repo)
    retired = {r['kernel']: r for r in read_tsv(retired_path)} if os.path.exists(retired_path) else {}

    library = []
    for o in oracle:
        name = o['kernel']
        loc = locate(name, kernels)
        doc_role = loc[3] if loc else ''
        note = o.get('note', '')
        role = doc_role or note
        if doc_role and note and len(doc_role) + len(note) + 3 <= ROLE_WITH_NOTE_MAX:
            role = '%s (%s)' % (doc_role, note)
        library.append({
            'kernel': name, 'family': o['family'], 'counterpart': o['counterpart'],
            'census': {c: census.get(name, {}).get(c) for c, _ in BACKENDS},
            'source': loc[0] if loc else '', 'path': loc[1] if loc else '', 'line': loc[2] if loc else 0,
            'role': role,
        })
    in_oracle = {o['kernel'] for o in oracle}
    probes, external, fixtures, unknown = [], [], [], []
    for name in sorted(census, key=str.lower):
        if name in in_oracle:
            continue
        classes = set(census[name].values())
        if 'EXTERNAL' in classes:
            external.append(name)
        elif 'PROBE' in classes:
            loc = locate(name, kernels)
            probes.append({'kernel': name, 'source': loc[0] if loc else '', 'path': loc[1] if loc else '',
                           'line': loc[2] if loc else 0, 'role': loc[3] if loc else ''})
        elif name in retired:
            fixtures.append({'kernel': name, 'where': retired[name]['where'],
                             'standing': retired[name]['standing']})
        else:
            unknown.append(name)
    return {'library': library, 'probes': probes, 'external': external,
            'fixtures': fixtures, 'unknown': unknown, 'kernels': kernels}


# ---------------------------------------------------------------- page

def github_base(repo):
    url = subprocess.run(['git', '-C', repo, 'remote', 'get-url', 'origin'],
                         capture_output=True, text=True, check=True).stdout.strip()
    m = re.match(r'(?:git@github\.com:|https://github\.com/)([^/]+/[^/]+?)(?:\.git)?$', url)
    if not m:
        raise SystemExit('origin is not a GitHub remote: ' + url)
    return 'https://github.com/%s/blob/main/' % m.group(1)


def esc(s):
    return html.escape(s, quote=True)


def kernel_cell(row, base):
    code = '<code>%s</code>' % esc(row['kernel'])
    if row.get('path'):
        title = '%s:%d' % (row['source'], row['line'])
        return '<td class="kernel-name"><a href="%s%s#L%d" title="%s">%s</a></td>' % (
            base, esc(row['path']), row['line'], esc(title), code)
    return '<td class="kernel-name">%s</td>' % code


def census_cell(cls):
    mark = MARKS.get(cls)
    if mark is None:
        raise SystemExit('census class with no mark: %s' % cls)
    return '<td class="census" title="%s">%s</td>' % (esc(cls or 'not registered'), mark)


def inline(text):
    """Escaped text with `code spans` as <code>."""
    return re.sub(r'`([^`]+)`', r'<code>\1</code>', esc(text))


def role_row(text, span):
    return '<tr class="kernel-role"><td colspan="%d">%s</td></tr>' % (span, inline(text))


def library_table(rows, base):
    out = ['<table class="kernel-catalog">',
           '<thead><tr><th>kernel</th><th>llama.cpp counterpart</th>'
           '<th class="census">cuda</th><th class="census">cpu</th><th class="census">hip</th></tr></thead>',
           '<tbody>']
    for r in rows:
        out.append('<tr class="kernel">%s<td>%s</td>%s</tr>' % (
            kernel_cell(r, base), esc(r['counterpart']),
            ''.join(census_cell(r['census'][c]) for c, _ in BACKENDS)))
        out.append(role_row(r['role'], 5))
    out += ['</tbody>', '</table>']
    return '\n'.join(out)


def probe_table(rows, base):
    out = ['<table class="kernel-catalog">', '<thead><tr><th>kernel</th></tr></thead>', '<tbody>']
    for r in rows:
        out.append('<tr class="kernel">%s</tr>' % kernel_cell(r, base))
        out.append(role_row(r['role'], 1))
    out += ['</tbody>', '</table>']
    return '\n'.join(out)


def fixture_table(rows):
    out = ['<table class="kernel-catalog">', '<thead><tr><th>kernel</th><th>where</th></tr></thead>', '<tbody>']
    for r in rows:
        out.append('<tr class="kernel"><td class="kernel-name"><code>%s</code></td><td>%s</td></tr>' % (
            esc(r['kernel']), esc(r['where'])))
        out.append(role_row(r['standing'], 2))
    out += ['</tbody>', '</table>']
    return '\n'.join(out)


LEGEND = ('Census marks, one per backend (hover a mark for its class): '
          '✓ CHECKED, ○ RAN-UNCHECKED, – SKIP, ✗ DECLINED, blank: not registered on that backend. '
          'Each kernel\'s name links to its source line; the line under it is its role.')


def render(model, base, date):
    lib = model['library']
    out = []
    out.append('# Catalog: every cajeta-llm kernel')
    out.append('')
    out.append(
        'Generated from the sources, `census/oracle-map.tsv` and the committed census tables of '
        '`dev.cajeta.llm` on %s by `tools/catalog/gen-catalog.py`. One entry per kernel registered in the '
        'test binary on any backend: the kernel\'s name, linked to its source line, the llama.cpp '
        'counterpart the parity harness compares it against (NONE where llama.cpp has no twin and the '
        'roofline judges it), the census class on `cuda`, `cpu` and `hip` (CHECKED: a test credited its '
        'values against a host or oracle answer; RAN-UNCHECKED and SKIP: excused against a named plan '
        'item; DECLINED: the compiler produced no device code there; blank: not registered on that '
        'backend), and under it the kernel\'s role from its own documentation. The families and routes '
        'are described in [Language models on the device](llm.md); the adaptation machinery in '
        '[One kernel, any device](README.md).' % date)
    out.append('')
    out.append('%d library kernels, plus the probe, external and fixture kernels the binary also '
               'registers (listed last).' % len(lib))
    out.append('')
    out.append(LEGEND)
    out.append('')
    for fam, title in FAMILIES:
        rows = sorted((r for r in lib if r['family'] == fam), key=lambda r: r['kernel'].lower())
        if not rows:
            continue
        out.append('')
        out.append('## %s (%d)' % (title, len(rows)))
        out.append('')
        out.append(library_table(rows, base))
    unfiled = [r for r in lib if r['family'] not in dict(FAMILIES)]
    if unfiled:
        raise SystemExit('oracle-map families with no heading: %s' % sorted({r['family'] for r in unfiled}))
    out.append('')
    out.append('## Bench probes (%d)' % len(model['probes']))
    out.append('')
    out.append('Kernels of the `bench` package and the in-tree probes, built into probe binaries to measure '
               'a mechanism (an LDS layout, a read pattern, a fragment load); never on a model\'s route. '
               'The census classes them PROBE.')
    out.append('')
    out.append(probe_table(model['probes'], base))
    out.append('')
    out.append('## Standard-library kernels the binary registers (%d)' % len(model['external']))
    out.append('')
    out.append('Tensor kernels of the Cajeta standard library (`cajeta.ml`), linked into every program that '
               'uses it: element-wise arithmetic, reductions, matmuls, padding and cropping, bitonic sort '
               'and scan steps. The library does not launch them on a model\'s route; the census classes '
               'them EXTERNAL and the standard library\'s own suites hold them.')
    out.append('')
    out.append(', '.join('`%s`' % k for k in model['external']))
    if model['fixtures']:
        out.append('')
        out.append('')
        out.append('## Fixtures and retired kernels (%d)' % len(model['fixtures']))
        out.append('')
        out.append(fixture_table(model['fixtures']))
    out.append('')
    return '\n'.join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('output', help='the page to write, e.g. ../cajeta/docs/kernels/catalog-llm.md')
    ap.add_argument('--repo', default=REPO, help='the cajeta-llm checkout (default: this one)')
    ap.add_argument('--date', default=datetime.date.today().isoformat())
    ap.add_argument('--retired', default=os.path.join(HERE, 'retired.tsv'))
    a = ap.parse_args(argv)
    model = build(a.repo, a.retired)
    if model['unknown']:
        raise SystemExit('census kernels that are neither in oracle-map.tsv, PROBE, EXTERNAL nor '
                         'tools/catalog/retired.tsv: ' + ', '.join(model['unknown']))
    page = render(model, github_base(a.repo), a.date)
    with open(a.output, 'w', encoding='utf-8') as f:
        f.write(page)
    missing = [r['kernel'] for r in model['library'] + model['probes'] if not r['path']]
    print('%s: %d library kernels, %d probes, %d external, %d fixtures' % (
        a.output, len(model['library']), len(model['probes']), len(model['external']), len(model['fixtures'])))
    if missing:
        print('no source found for: ' + ', '.join(missing), file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())

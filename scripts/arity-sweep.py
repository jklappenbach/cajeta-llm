#!/usr/bin/env python3
"""Static arity sweep (xpu-kernel-adaptor 1.2.3, widened by 1.4.5).

The compiler stops at the FIRST call whose argument count disagrees with the
library's declaration, so drift between the test tree and a library
signature is otherwise found one site per build round trip (~3 minutes on
nvptx, ~55 on cpu). This sweep finds every such site in one pass, before the
build: it reads the declared arities of every static method and constructor
in the library trees given, then checks every `Class.method(` and
`heap Class(` in the test tree against them.

    arity-sweep.py --lib <dir> [--lib <dir> ...] --test <dir>

Exit 0 with a one-line count when nothing drifts, 1 with each drifted site
as file:line otherwise. Unknown classes (cajeta-unit's Assert, a test-local
helper) are not checked: the sweep can only vouch for what it has read.
"""
import argparse
import os
import re
import sys
from collections import defaultdict

DECL = re.compile(r'\bstatic\s+(?:final\s+)?#?[\w<>,\[\]\. ]+?\s+([a-z]\w*)\s*\(')
CTOR_TMPL = r'\b(?:public|protected|private)\s+{cls}\s*\('
CALL = re.compile(r'\b([A-Z]\w*)\.([a-z]\w*)\s*\(')
HEAP = re.compile(r'\b(?:heap|stack)\s+([A-Z]\w*)\s*\(')


def strip_comments_and_strings(text):
    """Blank out comments and string/char literals, keeping newlines so line
    numbers survive. A paren inside a string would otherwise unbalance a call."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith('//', i):
            j = text.find('\n', i)
            j = n if j < 0 else j
            out.append(' ' * (j - i)); i = j
        elif text.startswith('/*', i):
            j = text.find('*/', i + 2)
            j = n if j < 0 else j + 2
            out.append(re.sub(r'[^\n]', ' ', text[i:j])); i = j
        elif c == '"' or c == "'":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == '\\' else 1
            j = min(j + 1, n)
            out.append(c + ' ' * (j - i - 2) + c if j - i >= 2 else c); i = j
        else:
            out.append(c); i += 1
    return ''.join(out)


def arg_count(text, open_paren):
    """Number of top-level arguments of the paren group starting at open_paren,
    or None when the group does not close. Angle brackets nest for templates."""
    depth = 0
    angle = 0
    commas = 0
    nonblank = False
    i = open_paren
    while i < len(text):
        c = text[i]
        if c in '([{':
            depth += 1
        elif c in ')]}':
            depth -= 1
            if depth == 0:
                return 0 if not nonblank else commas + 1
        elif depth == 1:
            if c == '<':
                angle += 1
            elif c == '>':
                angle = max(0, angle - 1)
            elif c == ',' and angle == 0:
                commas += 1
            if not c.isspace():
                nonblank = True
        i += 1
    return None


def cajeta_files(root):
    for d, _, files in os.walk(root):
        for f in files:
            if f.endswith('.cajeta'):
                yield os.path.join(d, f)


def declared(lib_dirs):
    """class -> method -> set(arity); constructors under the method name '<init>'."""
    table = defaultdict(lambda: defaultdict(set))
    for root in lib_dirs:
        for path in cajeta_files(root):
            cls = os.path.basename(path)[:-len('.cajeta')]
            text = strip_comments_and_strings(open(path, encoding='utf-8').read())
            for m in DECL.finditer(text):
                n = arg_count(text, m.end() - 1)
                if n is not None:
                    table[cls][m.group(1)].add(n)
            for m in re.finditer(CTOR_TMPL.format(cls=re.escape(cls)), text):
                n = arg_count(text, m.end() - 1)
                if n is not None:
                    table[cls]['<init>'].add(n)
    return table


def sweep(table, test_dir):
    drift = []
    sites = 0
    for path in cajeta_files(test_dir):
        # A test class that shadows a library class name checks against itself,
        # which is the compiler's resolution too.
        text = strip_comments_and_strings(open(path, encoding='utf-8').read())
        for m in CALL.finditer(text):
            cls, meth = m.group(1), m.group(2)
            if cls not in table or meth not in table[cls]:
                continue
            sites += 1
            n = arg_count(text, m.end() - 1)
            if n is None or n in table[cls][meth]:
                continue
            line = text.count('\n', 0, m.start()) + 1
            drift.append((path, line, f'{cls}.{meth}', n, sorted(table[cls][meth])))
        for m in HEAP.finditer(text):
            cls = m.group(1)
            if cls not in table or '<init>' not in table[cls]:
                continue
            sites += 1
            n = arg_count(text, m.end() - 1)
            if n is None or n in table[cls]['<init>']:
                continue
            line = text.count('\n', 0, m.start()) + 1
            drift.append((path, line, f'heap {cls}', n, sorted(table[cls]['<init>'])))
    return sites, drift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lib', action='append', required=True,
                    help='a library source root whose declarations are authoritative')
    ap.add_argument('--test', required=True, help='the test source root to sweep')
    a = ap.parse_args()
    table = declared(a.lib)
    classes = len(table)
    methods = sum(len(v) for v in table.values())
    sites, drift = sweep(table, a.test)
    if drift:
        print(f'>> arity sweep: {len(drift)} call site(s) disagree with a declaration '
              f'({classes} classes, {methods} static methods/constructors read):')
        for path, line, what, n, want in drift:
            print(f'   {path}:{line}: {what} called with {n} argument(s); declared {want}')
        return 1
    print(f'>> arity sweep: {sites} call sites against {classes} classes '
          f'({methods} static methods/constructors) agree')
    return 0


if __name__ == '__main__':
    sys.exit(main())

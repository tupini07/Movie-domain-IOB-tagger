import subprocess


def call(cmd):
    subprocess.call(["bash", '-c', cmd])


def draw(name):
    call(
        f"fstdraw --isymbols=w/lex.syms --osymbols=w/lex.syms -portrait w/{name}.fsa | dot -Tsvg > draw/{name}.svg")


def fstcompile(name):
    call(
        f"fstcompile --isymbols=w/lex.syms --osymbols=w/lex.syms --keep_osymbols --keep_isymbols w/{name}.txt > w/{name}.fsa")

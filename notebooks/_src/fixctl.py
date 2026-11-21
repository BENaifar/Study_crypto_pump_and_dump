"""Replace control characters (from accidental escape sequences) by the intended LaTeX backslashes."""
import pathlib

BS = chr(92)
for p in pathlib.Path(__file__).resolve().parent.glob("I*.py"):
    s = p.read_text(encoding="utf-8")
    t = s.replace(chr(7), BS + "a").replace(chr(8), BS + "b").replace(chr(12), BS + "f")
    if t != s:
        p.write_text(t, encoding="utf-8")
        print("fixed", p.name)

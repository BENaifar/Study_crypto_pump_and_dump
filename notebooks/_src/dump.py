"""Print the text/markdown outputs of an executed notebook (for reviewing results): python dump.py NAME"""
import re
import sys
from pathlib import Path

import nbformat

nb = nbformat.read(Path(__file__).resolve().parent.parent / f"{sys.argv[1]}.ipynb", as_version=4)
for i, c in enumerate(nb.cells):
    if c.cell_type != "code":
        continue
    for o in c.get("outputs", []):
        if o.output_type == "error":
            print(f"--- cell {i} ERROR", o.ename, o.evalue)
        elif o.output_type == "stream":
            print(f"--- cell {i} stream:", o.text[-1500:])
        else:
            d = o.get("data", {})
            if "text/markdown" in d:
                print(f"--- cell {i} md:", d["text/markdown"])
            elif "text/html" in d:
                t = re.sub(r"<style.*?</style>", "", d["text/html"], flags=re.S)
                t = re.sub(r"<[^>]+>", " ", t)
                print(f"--- cell {i} table:", re.sub(r"\s+", " ", t)[:1500])
            elif "image/png" in d:
                print(f"--- cell {i} [figure]")

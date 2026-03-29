"""One-off: generate utils/disease_hi_generated.py from _strings_utf8.txt."""
import time
from pathlib import Path

from deep_translator import GoogleTranslator

ROOT = Path(__file__).resolve().parents[1]
src = (ROOT / "_strings_utf8.txt").read_text(encoding="utf-8").strip().splitlines()
tr = GoogleTranslator(source="en", target="hi")
out: dict[str, str] = {}

for i, line in enumerate(src):
    for attempt in range(3):
        try:
            out[line] = tr.translate(line)
            break
        except Exception:
            time.sleep(1.5 * (attempt + 1))
            if attempt == 2:
                out[line] = line
    if (i + 1) % 40 == 0:
        print(i + 1, "/", len(src))
    time.sleep(0.15)

parts = [f"    {repr(k)}: {repr(v)}" for k, v in out.items()]
(ROOT / "utils" / "disease_hi_generated.py").write_text(
    "# Auto-generated EN->HI disease strings\n"
    "DISEASE_HI = {\n" + ",\n".join(parts) + "\n}\n",
    encoding="utf-8",
)
print("OK", len(out))

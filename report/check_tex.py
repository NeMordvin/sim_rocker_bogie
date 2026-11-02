"""Простая проверка report.tex без LaTeX: окружения, скобки, ссылки, картинки, символы вне формул."""

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
text = (HERE / "report.tex").read_text(encoding="utf-8")
body = re.sub(r"%.*", "", text)
begins = re.findall(r"\\begin\{(\w+\*?)\}", body)
ends = re.findall(r"\\end\{(\w+\*?)\}", body)
print("окружения:", len(begins), len(ends), sorted(set(begins) ^ set(ends)))
print("баланс скобок:", body.count("{") - body.count("}"))
print("\\todo:", body.count("\\todo{"))
print("символы, которых нет в T2A:", sorted(set(re.findall(r"[α-ωΑ-Ω≈×μ°₀-₉∝≤≥·]", body))))
labels = set(re.findall(r"\\label\{([^}]+)\}", body))
refs = set(re.findall(r"\\ref\{([^}]+)\}", body))
print("ссылки без меток:", refs - labels)
for name in re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", body):
    print(name, (HERE / "fig" / name).exists())

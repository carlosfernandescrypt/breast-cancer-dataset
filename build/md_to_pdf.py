"""Converte um arquivo Markdown em PDF.

Fluxo: Markdown -> HTML (biblioteca `markdown`) -> PDF (Chromium headless).
Não depende de pandoc nem de LaTeX. As imagens com caminho relativo são
resolvidas para caminho absoluto (file://) para renderizar no Chromium.

Uso:
    python build/md_to_pdf.py <entrada.md> <saida.pdf> [titulo]
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

CSS = """
@page { size: A4; margin: 2cm 2.2cm; }
* { box-sizing: border-box; }
body {
  font-family: "Liberation Serif", "DejaVu Serif", Georgia, serif;
  font-size: 11.5pt; line-height: 1.5; color: #1a1a1a; max-width: 100%;
}
h1, h2, h3, h4 { font-family: "Liberation Sans", "DejaVu Sans", Arial, sans-serif;
  color: #0b3d6b; line-height: 1.25; }
h1 { font-size: 20pt; border-bottom: 3px solid #0b3d6b; padding-bottom: 6px;
  margin-top: 0; }
h2 { font-size: 15pt; border-bottom: 1px solid #c5d4e3; padding-bottom: 3px;
  margin-top: 1.4em; }
h3 { font-size: 12.5pt; }
p { text-align: justify; }
code { background: #f3f5f7; padding: 1px 4px; border-radius: 3px;
  font-family: "DejaVu Sans Mono", monospace; font-size: 9.5pt; }
pre { background: #f3f5f7; padding: 10px; border-radius: 5px; overflow-x: auto;
  font-size: 9pt; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; font-size: 10pt; }
th, td { border: 1px solid #c5d4e3; padding: 6px 9px; text-align: center; }
th { background: #0b3d6b; color: #fff; }
tr:nth-child(even) td { background: #f5f8fb; }
img { max-width: 100%; height: auto; display: block; margin: 0.6em auto;
  border: 1px solid #e2e8ef; }
blockquote { border-left: 4px solid #0b3d6b; margin: 1em 0; padding: 2px 14px;
  color: #444; background: #f5f8fb; }
em { color: #333; }
hr { border: none; border-top: 1px solid #c5d4e3; margin: 1.5em 0; }
figcaption, .legenda { text-align: center; font-size: 9.5pt; color: #555;
  font-style: italic; margin-bottom: 1em; }
"""


def encontrar_chromium() -> str:
    for nome in ("google-chrome", "chromium", "chromium-browser"):
        caminho = shutil.which(nome)
        if caminho:
            return caminho
    raise RuntimeError("Chromium/Chrome não encontrado no PATH.")


def converter(md_path: Path, pdf_path: Path, titulo: str = "Relatório") -> None:
    base = md_path.resolve().parent
    texto = md_path.read_text(encoding="utf-8")

    html_corpo = markdown.markdown(
        texto, extensions=["tables", "fenced_code", "toc", "attr_list", "md_in_html"]
    )
    # Resolve caminhos relativos de imagem para file:// absoluto.
    html_corpo = html_corpo.replace('src="../', f'src="file://{base.parent}/')
    html_corpo = html_corpo.replace('src="./', f'src="file://{base}/')
    html_corpo = html_corpo.replace('src="figures/', f'src="file://{base.parent}/figures/')

    html = f"""<!doctype html><html lang="pt-br"><head><meta charset="utf-8">
<title>{titulo}</title><style>{CSS}</style></head>
<body>{html_corpo}</body></html>"""

    with tempfile.TemporaryDirectory() as tmp:
        html_file = Path(tmp) / "doc.html"
        html_file.write_text(html, encoding="utf-8")
        chrome = encontrar_chromium()
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            chrome, "--headless", "--disable-gpu", "--no-sandbox",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}", html_file.as_uri(),
        ], check=True, capture_output=True)
    print(f"PDF gerado: {pdf_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    titulo = sys.argv[3] if len(sys.argv) > 3 else "Relatório"
    converter(Path(sys.argv[1]), Path(sys.argv[2]), titulo)

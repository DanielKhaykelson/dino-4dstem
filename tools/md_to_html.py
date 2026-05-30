"""md_to_html.py -- minimal markdown → self-contained HTML.

Handles the subset used by tools/imc_report.py:
  - ATX headers (# .. ######)
  - paragraphs, bold (**…**), italic (_…_), inline code (`…`)
  - unordered lists (-) with nested indentation
  - pipe tables with --- separator
  - images ![alt](path) — embedded inline as data URIs so the HTML
    file is portable
  - hr (---)
  - horizontal rules
No code blocks, no nested lists, no autolinks — kept tight.
"""
from __future__ import annotations
import argparse, base64, html as html_lib, os, re, sys


CSS = """
body { max-width: 980px; margin: 2em auto; padding: 0 1.5em;
        font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif;
        line-height: 1.55; color: #222; }
h1, h2, h3 { color: #1a3a66; margin-top: 1.6em; }
h1 { border-bottom: 2px solid #1a3a66; padding-bottom: 0.3em; }
h2 { border-bottom: 1px solid #ccd; padding-bottom: 0.2em; }
table { border-collapse: collapse; margin: 1em 0; font-size: 0.92em; }
th, td { border: 1px solid #ccd; padding: 4px 10px; text-align: left; }
th { background: #eef2f7; }
img { max-width: 100%; display: block; margin: 1em auto;
       box-shadow: 0 2px 8px rgba(0,0,0,0.12); }
code { background: #f4f4f8; padding: 1px 5px; border-radius: 3px;
        font-size: 0.92em; }
hr { border: 0; border-top: 1px solid #ccd; margin: 2em 0; }
ul { margin: 0.4em 0; }
li { margin: 0.18em 0; }
.summary { background: #f7fafd; padding: 0.4em 1em;
            border-left: 4px solid #1a3a66; margin: 1em 0; }
"""


def _embed_image(path: str, base_dir: str) -> str:
    """Read image at `path` (relative to base_dir) and return a data URI."""
    full = path if os.path.isabs(path) else os.path.join(base_dir, path)
    if not os.path.exists(full):
        return ""
    ext = os.path.splitext(full)[1].lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
              "gif": "image/gif", "svg": "image/svg+xml"}.get(ext,
                                                                 "image/png")
    with open(full, "rb") as f:
        b = f.read()
    return f"data:{mime};base64,{base64.b64encode(b).decode('ascii')}"


def _inline(s: str) -> str:
    """Convert inline markdown to inline HTML."""
    # Escape HTML first.
    s = html_lib.escape(s)
    # Inline code (do first so its contents don't get bolded etc.).
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    # Bold (**...**) — must come before italic.
    s = re.sub(r"\*\*([^\*]+)\*\*", r"<strong>\1</strong>", s)
    # Italic (_..._ or *...*).
    s = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"<em>\1</em>", s)
    s = re.sub(r"\*([^\*]+)\*", r"<em>\1</em>", s)
    return s


def _process_image_lines(lines, base_dir, embed):
    """Replace ![alt](path) on its own line with an <img> tag."""
    out = []
    img_re = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
    for ln in lines:
        m = img_re.fullmatch(ln.strip())
        if m:
            alt, path = m.group(1), m.group(2)
            if embed:
                src = _embed_image(path, base_dir)
                if not src:
                    out.append(f"<p><em>missing image: {html_lib.escape(path)}</em></p>")
                    continue
            else:
                src = path
            out.append(
                f'<figure><img src="{src}" alt="{html_lib.escape(alt)}">'
                f"<figcaption>{html_lib.escape(alt)}</figcaption></figure>")
        else:
            out.append(ln)
    return out


def md_to_html(md: str, base_dir: str, embed_images: bool = True) -> str:
    """Convert markdown text to a complete HTML document."""
    lines = md.splitlines()
    lines = _process_image_lines(lines, base_dir, embed_images)

    out: list[str] = []
    i = 0
    in_list = False
    while i < len(lines):
        ln = lines[i]
        stripped = ln.strip()

        # Header
        m = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if m:
            if in_list: out.append("</ul>"); in_list = False
            level = len(m.group(1))
            txt = _inline(m.group(2))
            out.append(f"<h{level}>{txt}</h{level}>")
            i += 1; continue

        # Horizontal rule
        if re.match(r"^---+$", stripped) and not (
                i + 1 < len(lines)
                and "|" in lines[i + 1]
                and "|" in (lines[i - 1] if i > 0 else "")):
            if in_list: out.append("</ul>"); in_list = False
            out.append("<hr>")
            i += 1; continue

        # Table (header | --- | rows)
        if "|" in stripped and i + 1 < len(lines) and re.match(
                r"^\|?\s*[:\- ]+\|", lines[i + 1].strip()):
            if in_list: out.append("</ul>"); in_list = False
            hdr_cells = [c.strip() for c in
                            stripped.strip("|").split("|")]
            out.append("<table><thead><tr>")
            for c in hdr_cells:
                out.append(f"<th>{_inline(c)}</th>")
            out.append("</tr></thead><tbody>")
            i += 2          # skip header + separator
            while i < len(lines) and "|" in lines[i].strip():
                row = lines[i].strip().strip("|")
                cells = [c.strip() for c in row.split("|")]
                out.append("<tr>")
                for c in cells:
                    out.append(f"<td>{_inline(c)}</td>")
                out.append("</tr>")
                i += 1
            out.append("</tbody></table>")
            continue

        # Image already converted to <figure ...>
        if stripped.startswith("<figure>"):
            if in_list: out.append("</ul>"); in_list = False
            out.append(stripped); i += 1; continue

        # Unordered list
        if stripped.startswith("- ") or stripped.startswith("* "):
            if not in_list:
                out.append("<ul>"); in_list = True
            out.append(f"<li>{_inline(stripped[2:])}</li>")
            i += 1; continue

        # Blank line
        if not stripped:
            if in_list: out.append("</ul>"); in_list = False
            i += 1; continue

        # Paragraph (collect contiguous non-empty, non-special lines)
        para_lines = [stripped]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (not nxt
                    or nxt.startswith(("#", "- ", "* ", "<figure>"))
                    or re.match(r"^---+$", nxt)
                    or ("|" in nxt and i + 1 < len(lines)
                          and re.match(r"^\|?\s*[:\- ]+\|",
                                          lines[i + 1].strip()))):
                break
            para_lines.append(nxt); i += 1
        if in_list: out.append("</ul>"); in_list = False
        out.append(f"<p>{_inline(' '.join(para_lines))}</p>")

    if in_list: out.append("</ul>")

    # Title from first H1, if any
    title_m = re.search(r"^#\s+(.+)$", md, re.M)
    title = title_m.group(1) if title_m else "Report"

    return (f"<!DOCTYPE html>\n<html lang='en'><head>"
            f"<meta charset='utf-8'>"
            f"<title>{html_lib.escape(title)}</title>"
            f"<style>{CSS}</style>"
            f"</head><body>\n"
            + "\n".join(out)
            + "\n</body></html>\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("md_path")
    ap.add_argument("html_path", nargs="?",
                    default=None,
                    help="output html (default: <md_path>.html)")
    ap.add_argument("--no-embed", action="store_true",
                    help="link to images by relative path instead of "
                         "inlining them (smaller HTML, requires the "
                         "PNGs to live next to the HTML).")
    args = ap.parse_args()
    md_path = os.path.abspath(args.md_path)
    html_path = (os.path.abspath(args.html_path)
                  if args.html_path
                  else os.path.splitext(md_path)[0] + ".html")
    with open(md_path, encoding="utf-8") as f:
        md = f.read()
    base_dir = os.path.dirname(md_path)
    html = md_to_html(md, base_dir, embed_images=not args.no_embed)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[md_to_html] wrote {html_path}  "
            f"({os.path.getsize(html_path) / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

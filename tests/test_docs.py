# -*- coding: utf-8 -*-
"""The documentation matches the code: every relative link of the Markdown files (and every `#anchor` in them) leads somewhere, every
command of the two command-line tools answers `--help` and is described in the manual, and every option used in a command of the
documentation exists."""
import contextlib
import io
import os
import re
import shlex
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def markdown_files():
    out = [os.path.join(ROOT, name) for name in ("README.md", "CHANGELOG.md") if os.path.isfile(os.path.join(ROOT, name))]
    for folder in ("docs", "cases", "results"):
        for here, _, names in os.walk(os.path.join(ROOT, folder)):
            out += [os.path.join(here, n) for n in names if n.endswith(".md")]
    return sorted(out)


def prose_lines(path):
    """The lines of a Markdown file outside fenced code blocks, with the text of inline code removed."""
    fenced = False
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if line.lstrip().startswith("```"):
                fenced = not fenced
                continue
            if not fenced:
                yield number, re.sub(r"`[^`]*`", "", line)


def slug(heading):
    """The anchor GitHub makes of a heading: lower case, no punctuation except `-` and `_`, spaces become `-`."""
    heading = re.sub(r"`", "", heading.strip().lower())
    heading = re.sub(r"[^\w\- ]", "", heading, flags=re.UNICODE)
    return heading.replace(" ", "-")


def anchors(path):
    seen, out = {}, set()
    for _, line in prose_lines(path):
        m = re.match(r"#{1,6}\s+(.*?)\s*#*\s*$", line)
        if m:
            base = slug(re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", m.group(1)))
            count = seen.get(base, 0)
            out.add(base if count == 0 else f"{base}-{count}")
            seen[base] = count + 1
    return out


def links(path):
    """`(line, target)` of every inline link and image of the file that is not a web address."""
    with open(path, encoding="utf-8") as f:
        fenced = False
        for number, line in enumerate(f, 1):
            if line.lstrip().startswith("```"):
                fenced = not fenced
                continue
            if fenced:
                continue
            for target in re.findall(r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", re.sub(r"`[^`]*`", "", line)):
                if not re.match(r"[a-z][a-z0-9+.-]*:", target):
                    yield number, target


class Links(unittest.TestCase):
    def test_there_are_files_to_check(self):
        self.assertGreater(len(markdown_files()), 8)

    def test_every_relative_link_leads_to_a_file_and_every_anchor_to_a_heading(self):
        broken = []
        for path in markdown_files():
            for number, target in links(path):
                name, _, anchor = target.partition("#")
                resolved = os.path.normpath(os.path.join(os.path.dirname(path), name)) if name else path
                where = f"{os.path.relpath(path, ROOT)}:{number}: {target}"
                if not os.path.exists(resolved):
                    broken.append(f"{where} (no such file)")
                elif anchor and resolved.endswith(".md") and anchor not in anchors(resolved):
                    broken.append(f"{where} (no such heading)")
        self.assertEqual(broken, [], "\n" + "\n".join(broken))

    def test_slug_follows_github(self):
        self.assertEqual(slug("The shelters (A3 and S2)"), "the-shelters-a3-and-s2")
        self.assertEqual(slug("Building a `case`"), "building-a-case")
        self.assertEqual(slug("Model options"), "model-options")

    def test_repeated_headings_get_a_suffix(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.md")
            with open(path, "w", encoding="utf-8") as f:
                f.write("# A\n\n## B\n\n## B\n\n```\n## not a heading\n```\n")
            self.assertEqual(anchors(path), {"a", "b", "b-1"})


def run_help(main, argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            main(argv + ["--help"])
        except SystemExit as exc:
            code = exc.code
        else:
            code = None
    return code, out.getvalue()


def commands(main):
    """The sub-commands of a command-line tool, read from its own usage line."""
    code, text = run_help(main, [])
    m = re.search(r"\{([^}]+)\}", text)
    return m.group(1).split(",") if m else []


def documented_commands():
    """`(file, tool, subcommand, options)` for every `python -m evacrl.<tool> <subcommand> ...` in the code blocks of the documentation."""
    out = []
    for path in markdown_files():
        if os.sep + "audits" + os.sep in path:
            continue                      # the audits record what was run then
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for block in re.findall(r"```[^\n]*\n(.*?)```", text, flags=re.S):
            for command in re.sub(r"\\\n\s*", " ", block).splitlines():
                m = re.match(r"\s*(?:\w+=\S+\s+)*(?:python3?\s+-m\s+(evacrl\.(?:experiment|casebuild))|(evacrl-(?:experiment|casebuild)))\s+(\S+)(.*)", command)
                if not m:
                    continue
                tool = (m.group(1) or m.group(2)).replace("evacrl-", "evacrl.")
                try:
                    words = shlex.split(m.group(4).split("#")[0])
                except ValueError:
                    continue
                out.append((os.path.relpath(path, ROOT), tool, m.group(3), [w.split("=")[0] for w in words if w.startswith("--")]))
    return out


class CommandLines(unittest.TestCase):
    TOOLS = {}

    @classmethod
    def setUpClass(cls):
        from evacrl.casebuild.cli import main as casebuild
        from evacrl.experiment.cli import main as experiment
        cls.TOOLS = {"evacrl.casebuild": casebuild, "evacrl.experiment": experiment}

    def test_every_command_answers_help(self):
        for tool, main in self.TOOLS.items():
            names = commands(main)
            self.assertGreaterEqual(len(names), 4, tool)
            for name in names:
                code, text = run_help(main, [name])
                self.assertEqual(code, 0, f"{tool} {name} --help")
                self.assertIn(name, text.split("\n", 1)[0], f"{tool} {name}: usage line")

    def test_every_command_is_in_the_manual(self):
        with open(os.path.join(ROOT, "docs", "manual.md"), encoding="utf-8") as f:
            manual = f.read()
        missing = []
        for tool, main in self.TOOLS.items():
            for name in commands(main):
                if not re.search(rf"(python3?\s+-m\s+{re.escape(tool)}|{re.escape(tool.replace('evacrl.', 'evacrl-'))})\s+{re.escape(name)}\b", manual):
                    missing.append(f"{tool} {name}")
        self.assertEqual(missing, [], "commands the manual does not show: " + ", ".join(missing))

    def test_every_option_in_a_documented_command_exists(self):
        found = documented_commands()
        self.assertGreater(len(found), 10, "the documentation should show the commands")
        helps, wrong = {}, []
        for path, tool, name, options in found:
            main = self.TOOLS[tool]
            self.assertIn(name, commands(main), f"{path}: `{tool} {name}` is not a command")
            if (tool, name) not in helps:
                helps[tool, name] = run_help(main, [name])[1]
            wrong += [f"{path}: {tool} {name} {option}" for option in options if option not in helps[tool, name]]
        self.assertEqual(wrong, [], "\n" + "\n".join(wrong))

    def test_the_command_line_reader_finds_what_it_should(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            global ROOT
            keep, ROOT = ROOT, d
            try:
                with open(os.path.join(d, "README.md"), "w", encoding="utf-8") as f:
                    f.write("```\nFOO=1 python -m evacrl.experiment sp kochi_area2 --runs 3 \\\n    --out x   # note --nothing\n"
                            "evacrl-casebuild validate cases/a\npython other.py --no\n```\n")
                self.assertEqual(documented_commands(),
                                 [("README.md", "evacrl.experiment", "sp", ["--runs", "--out"]),
                                  ("README.md", "evacrl.casebuild", "validate", [])])
            finally:
                ROOT = keep


if __name__ == "__main__":
    unittest.main()

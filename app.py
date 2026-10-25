# -*- coding: utf-8 -*-
"""
Генератор математических лабиринтов: обыкновенные и десятичные дроби, 5–8 класс.
ИИ не используется: примеры берутся из tasks_bank.py, ответы считает программа.

Запуск сайта:            python app.py              -> http://127.0.0.1:5000
Проверка банка задач:    python app.py check
Пробный PDF без сайта:   python app.py demo 5 fraction
"""
import io
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from flask import Flask, flash, redirect, render_template_string, request, send_file, url_for
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas as pdf_canvas

from tasks_bank import TASK_BANK

# ======================================================================
# НАСТРОЙКИ
# ======================================================================
GRADES = [5, 6, 7, 8]
TOPICS = {"decimal": "десятичные дроби", "fraction": "обыкновенные дроби"}
MAX_LABEL_LEN = 9        # максимальная длина ответа на коридоре (символов)
MAX_DENOMINATOR = 200    # слишком громоздкие дроби в коридор не пускаем
RENDER_DPI = 300         # качество формул в PDF

# Шрифт формул. "cm" — как в учебнике, "dejavusans" — крупнее и чётче при печати
matplotlib.rcParams["mathtext.fontset"] = "dejavusans"

# Шрифт с кириллицей для PDF (уже лежит внутри matplotlib, скачивать не нужно)
_FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
pdfmetrics.registerFont(TTFont("DejaVu", os.path.join(_FONT_DIR, "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", os.path.join(_FONT_DIR, "DejaVuSans-Bold.ttf")))


# ======================================================================
# 1. РАЗБОР ВЫРАЖЕНИЯ: "2 1/3 * (0,5 - 1/4)" -> дерево -> ответ и формула
# ======================================================================
TOKEN_RE = re.compile(r"""
    (?P<mixed>\d+[ ]\d+/\d+)      # смешанное число: 2 1/3
  | (?P<frac>\d+/\d+)             # дробь: 3/4 (без пробелов вокруг /)
  | (?P<dec>\d+,\d+)              # десятичная дробь: 1,25
  | (?P<int>\d+)                  # целое число
  | (?P<sqrt>sqrt|√)              # корень
  | (?P<op>[-+*·×:/^()−])         # знаки и скобки
  | (?P<ws>\s+)
""", re.VERBOSE)


def tokenize(s):
    pos, out = 0, []
    while pos < len(s):
        m = TOKEN_RE.match(s, pos)
        if not m:
            raise ValueError(f"непонятный символ «{s[pos]}»")
        pos = m.end()
        kind, val = m.lastgroup, m.group()
        if kind == "ws":
            continue
        if kind == "op":
            val = {"·": "*", "×": "*", "−": "-"}.get(val, val)
        out.append((kind, val))
    return out


def make_num(kind, val):
    """Узел-число: ('num', значение, формула, вид).
    Неправильные дроби разрешены: 7/4, 40/1, 6 7/4 и т.п.
    Запрещён только знаменатель 0."""
    if kind == "int":
        return ("num", Fraction(int(val)), val, kind)
    if kind == "dec":
        a, b = val.split(",")
        return ("num", Fraction(int(a + b), 10 ** len(b)), val.replace(",", "{,}"), kind)
    if kind == "frac":
        n, d = map(int, val.split("/"))
        if d == 0:
            raise ValueError(f"знаменатель равен нулю: {val}")
        return ("num", Fraction(n, d), rf"\frac{{{n}}}{{{d}}}", kind)
    # смешанное число: дробная часть может быть неправильной (6 7/4 = 7 3/4)
    w, fr = val.split(" ")
    n, d = map(int, fr.split("/"))
    if d == 0:
        raise ValueError(f"знаменатель равен нулю: {val}")
    return ("num", int(w) + Fraction(n, d), rf"{w}\frac{{{n}}}{{{d}}}", kind)


class Parser:
    """Порядок действий: скобки -> степень -> минус перед числом -> · : -> + −"""

    def __init__(self, tokens):
        self.t, self.i = tokens, 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def take(self, val=None):
        tok = self.peek()
        if tok[0] is None or (val is not None and tok[1] != val):
            raise ValueError(f"ожидалось «{val}»")
        self.i += 1
        return tok

    def parse(self):
        node = self.expr()
        if self.i != len(self.t):
            raise ValueError("лишние символы в конце")
        return node

    def expr(self):
        node = self.term()
        while self.peek()[0] == "op" and self.peek()[1] in "+-":
            op = self.take()[1]
            node = ("bin", op, node, self.term())
        return node

    def term(self):
        node = self.unary()
        while self.peek()[0] == "op" and self.peek()[1] in "*:/":
            op = self.take()[1]
            node = ("bin", op, node, self.unary())
        return node

    def unary(self):
        if self.peek() == ("op", "-"):
            self.take()
            return ("neg", self.unary())
        if self.peek() == ("op", "+"):
            self.take()
            return self.unary()
        return self.power()

    def power(self):
        base = self.atom()
        if self.peek() == ("op", "^"):
            self.take()
            return ("pow", base, self.unary())
        return base

    def atom(self):
        kind, val = self.peek()
        if kind in ("int", "dec", "frac", "mixed"):
            self.take()
            return make_num(kind, val)
        if kind == "sqrt":
            self.take(); self.take("(")
            inner = self.expr()
            self.take(")")
            return ("sqrt", inner)
        if (kind, val) == ("op", "("):
            self.take()
            inner = self.expr()
            self.take(")")
            return ("par", inner)
        raise ValueError("ожидалось число или скобка")


def evaluate(node):
    """Точное вычисление в обыкновенных дробях (без ошибок округления)."""
    t = node[0]
    if t == "num":
        return node[1]
    if t == "par":
        return evaluate(node[1])
    if t == "neg":
        return -evaluate(node[1])
    if t == "sqrt":
        v = evaluate(node[1])
        if v < 0:
            raise ValueError("корень из отрицательного числа")
        rn, rd = math.isqrt(v.numerator), math.isqrt(v.denominator)
        if rn * rn != v.numerator or rd * rd != v.denominator:
            raise ValueError("корень не извлекается нацело")
        return Fraction(rn, rd)
    if t == "pow":
        b, e = evaluate(node[1]), evaluate(node[2])
        if e.denominator != 1 or abs(e) > 20:
            raise ValueError("показатель степени должен быть небольшим целым")
        if b == 0 and e < 0:
            raise ValueError("ноль в отрицательной степени")
        return b ** int(e)
    op, a, b = node[1], evaluate(node[2]), evaluate(node[3])
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if b == 0:
        raise ValueError("деление на ноль")
    return a / b


def strip_par(node):
    while node[0] == "par":
        node = node[1]
    return node


def to_tex(node):
    """Формула для отрисовки: · вместо *, дробная черта, степени, √."""
    t = node[0]
    if t == "num":
        return node[2]
    if t == "par":
        return r"\left(" + to_tex(node[1]) + r"\right)"
    if t == "neg":
        return "-" + to_tex(node[1])
    if t == "sqrt":
        return r"\sqrt{" + to_tex(node[1]) + "}"
    if t == "pow":
        base = to_tex(node[1])
        if node[1][0] == "num" and node[1][3] in ("frac", "mixed"):
            base = r"\left(" + base + r"\right)"
        return base + "^{" + to_tex(strip_par(node[2])) + "}"
    op = node[1]
    if op == "/":
        return r"\frac{" + to_tex(strip_par(node[2])) + "}{" + to_tex(strip_par(node[3])) + "}"
    sym = {"+": "+", "-": "-", "*": r"\cdot", ":": ":"}[op]
    return f"{to_tex(node[2])} {sym} {to_tex(node[3])}"


def complexity(node):
    """Грубая оценка сложности: больше действий и цифр — сложнее."""
    t = node[0]
    if t == "num":
        return 1 + 0.15 * sum(ch.isdigit() for ch in node[2])
    if t == "par":
        return complexity(node[1])
    if t in ("neg", "sqrt"):
        return 0.5 + complexity(node[1])
    if t == "pow":
        return 1 + complexity(node[1]) + complexity(node[2])
    return 1.5 + complexity(node[2]) + complexity(node[3])


def parse_task(expr):
    node = Parser(tokenize(expr)).parse()
    return node, evaluate(node)


# ======================================================================
# 2. ЗАПИСЬ ОТВЕТОВ: 0,25 / 2 1/3 / -5/6
# ======================================================================
def decimal_places(v):
    """Сколько знаков после запятой; None — если дробь бесконечная."""
    d, twos, fives = v.denominator, 0, 0
    while d % 2 == 0:
        d //= 2; twos += 1
    while d % 5 == 0:
        d //= 5; fives += 1
    return max(twos, fives) if d == 1 else None


def fmt_decimal(v):
    k = decimal_places(v)
    sign = "-" if v < 0 else ""
    v = abs(v)
    s = str(v.numerator * 10 ** k // v.denominator).rjust(k + 1, "0")
    return sign + (s[:-k] + "," + s[-k:] if k else s)


def fmt_fraction(v):
    if v.denominator == 1:
        return str(v.numerator)
    sign = "-" if v < 0 else ""
    v = abs(v)
    w, r = divmod(v.numerator, v.denominator)
    return sign + (f"{w} {r}/{v.denominator}" if w else f"{r}/{v.denominator}")


def format_answer(v, topic):
    return fmt_decimal(v) if topic == "decimal" else fmt_fraction(v)


def show_value(v, topic):
    if topic == "decimal" and decimal_places(v) is not None:
        return fmt_decimal(v)
    return fmt_fraction(v)


def fmt_tex(plain):
    """'2 1/3' -> 2\\frac{1}{3},  '0,25' -> 0{,}25"""
    m = re.fullmatch(r"(-?)(?:(\d+) )?(\d+)/(\d+)", plain)
    if m:
        sign, w, n, d = m.groups()
        return f"{sign}{w or ''}\\frac{{{n}}}{{{d}}}"
    return plain.replace(",", "{,}")


def answer_ok(v, topic):
    """Подходит ли ответ для подписи коридора."""
    if topic == "decimal":
        return decimal_places(v) is not None and len(fmt_decimal(v)) <= MAX_LABEL_LEN
    return v.denominator <= MAX_DENOMINATOR and len(fmt_fraction(v)) <= MAX_LABEL_LEN


# ======================================================================
# 3. БАНК ЗАДАЧ
# ======================================================================
@dataclass
class BankTask:
    text: str           # исходная запись
    tex: str            # формула для печати
    value: Fraction     # точный ответ
    answer: str         # ответ в виде строки
    difficulty: float


def _split_item(item):
    if isinstance(item, (tuple, list)):
        return str(item[0]).strip(), (str(item[1]).strip() if len(item) > 1 else None)
    return str(item).strip(), None


@lru_cache(maxsize=None)
def get_pool(grade, topic):
    pool, seen = [], set()
    for item in TASK_BANK.get(f"{grade}_{topic}", []):
        expr, _ = _split_item(item)
        if expr in seen:
            continue
        seen.add(expr)
        try:
            node, val = parse_task(expr)
        except (ValueError, ZeroDivisionError):
            continue
        if not answer_ok(val, topic):
            continue
        pool.append(BankTask(expr, to_tex(node), val, format_answer(val, topic), complexity(node)))
    return tuple(pool)


# ======================================================================
# 4. ЛОВУШКИ — правдоподобные неправильные ответы (типичные ошибки)
# ======================================================================
def make_distractors(value, topic, allow_negative, k=6):
    used, out = {format_answer(value, topic)}, []

    def try_add(cand):
        if len(out) >= k or (cand < 0 and not allow_negative) or not answer_ok(cand, topic):
            return
        s = format_answer(cand, topic)
        if s not in used:
            used.add(s)
            out.append(s)

    typical, other = [], []
    if topic == "decimal":
        step = Fraction(1, 10 ** (decimal_places(value) or 0))
        typical += [value * 10, value / 10]                       # запятая не там
        other += [value + step, value - step, value + 1, value - 1,
                  value + Fraction(1, 10), value - Fraction(1, 10), value * 100, value / 100]
    else:
        d = value.denominator
        if value != 0:
            typical.append(1 / value)                             # перевернули дробь
        if d > 1:
            typical += [value + Fraction(1, d), value - Fraction(1, d)]
            other += [value + 1, value - 1, value * 2, value / 2,
                      Fraction(value.numerator, d + 1), Fraction(value.numerator + 1, d + 1)]
        else:
            typical += [value + 1, value - 1]
            other += [value * 2, value + Fraction(1, 2), value - Fraction(1, 2), value + 2]
    if allow_negative and value != 0:
        typical.append(-value)                                    # ошибка в знаке
    random.shuffle(typical)
    random.shuffle(other)
    for cand in typical + other:
        try_add(cand)

    if topic == "decimal":
        step = Fraction(1, 10 ** (decimal_places(value) or 0))
    else:
        step = Fraction(1, value.denominator if value.denominator > 1 else random.choice([2, 3, 4]))
    for _ in range(300):
        if len(out) >= k:
            break
        try_add(value + random.choice([-1, 1]) * step * random.randint(2, 12))
    return out


# ======================================================================
# 5. ЛАБИРИНТ
# ======================================================================
class MathMaze:
    """Сетка клеток. Все соседние клетки соединены коридорами.
    Правильный путь не касается сам себя, поэтому неправильный коридор
    всегда ведёт в тупик, а не «срезает» дорогу к финишу."""

    def __init__(self, cols, rows, topic, grade):
        self.cols, self.rows, self.topic, self.grade = cols, rows, topic, grade
        self.n = cols * rows
        self.nb = {i: [] for i in range(self.n)}
        for r in range(rows):
            for c in range(cols):
                i = r * cols + c
                if c + 1 < cols:
                    self._link(i, i + 1)
                if r + 1 < rows:
                    self._link(i, i + cols)
        self.edges = sorted((a, b) for a in range(self.n) for b in self.nb[a] if a < b)
        self.path, self.task, self.traps, self.label = [], {}, {}, {}

    def _link(self, a, b):
        self.nb[a].append(b)
        self.nb[b].append(a)

    @property
    def start(self):
        return self.path[0]

    @property
    def finish(self):
        return self.path[-1]

    def path_edges(self):
        return {tuple(sorted(e)) for e in zip(self.path, self.path[1:])}

    # ---------- правильный путь ----------
    def _random_path(self, target, budget=4000):
        last_row = range((self.rows - 1) * self.cols, self.n)
        path, inside, steps = [0], {0}, [0]

        def can_go(nxt, tail):
            return nxt not in inside and all(x == tail or x not in inside for x in self.nb[nxt])

        def dfs():
            steps[0] += 1
            if steps[0] > budget:
                return False
            tail = path[-1]
            if tail in last_row and len(path) >= target:
                return True
            options = self.nb[tail][:]
            random.shuffle(options)
            for nxt in options:
                if can_go(nxt, tail):
                    path.append(nxt); inside.add(nxt)
                    if dfs():
                        return True
                    path.pop(); inside.discard(nxt)
            return False

        return path if dfs() else None

    def generate_path(self):
        target = int(self.n * 0.65)
        while target >= self.rows:
            for _ in range(12):
                p = self._random_path(target)
                if p:
                    self.path = p
                    return p
            target -= 1
        self.path = [r * self.cols for r in range(self.rows)]   # запасной: прямо вниз
        return self.path

    # ---------- задачи и подписи коридоров ----------
    def fill(self, pool):
        need = len(self.path) - 1                       # финиш без задачи
        shuffled = random.sample(pool, len(pool))
        path_tasks, answers = [], set()
        for t in shuffled:                              # на пути все ответы разные
            if t.answer not in answers:
                path_tasks.append(t)
                answers.add(t.answer)
            if len(path_tasks) == need:
                break
        if len(path_tasks) < need:
            raise ValueError("В банке слишком мало задач с разными ответами. Уменьшите размер лабиринта.")
        path_tasks.sort(key=lambda t: t.difficulty)     # от простого к сложному

        for cell, t in zip(self.path[:-1], path_tasks):
            self.task[cell] = t
        used = {t.text for t in path_tasks}
        rest = [t for t in shuffled if t.text not in used] or list(pool)
        dead = [c for c in range(self.n) if c not in self.path]
        for j, cell in enumerate(dead):
            self.task[cell] = rest[j % len(rest)]

        for cell, t in self.task.items():
            allow_neg = self.grade >= 7 or t.value < 0
            self.traps[cell] = make_distractors(t.value, self.topic, allow_neg)

        self._path_set = set(self.path)
        self.label = {}
        for a, b in zip(self.path, self.path[1:]):
            self.label[tuple(sorted((a, b)))] = self.task[a].answer
        others = [e for e in self.edges if e not in self.label]
        random.shuffle(others)
        others.sort(key=lambda e: -((e[0] in self._path_set) + (e[1] in self._path_set)))
        for e in others:
            self.label[e] = self._pick_trap(e)
        return self

    def _forbidden(self, cell):
        bad = {lab for e, lab in self.label.items() if cell in e}
        if cell in self.task:
            bad.add(self.task[cell].answer)
        return bad

    def _pick_trap(self, edge):
        a, b = edge
        with_task = [x for x in (a, b) if x in self.task]
        on_path = [x for x in with_task if x in self._path_set]
        off_path = [x for x in with_task if x not in self._path_set]
        random.shuffle(off_path)
        owners = on_path + off_path
        bad = self._forbidden(a) | self._forbidden(b)
        for owner in owners:
            for cand in self.traps[owner]:
                if cand not in bad:
                    return cand
        t = self.task[owners[0]]
        for cand in make_distractors(t.value, self.topic, True, k=40):
            if cand not in bad:
                return cand
        return "?"


def create_maze(topic, grade, cols, rows):
    pool = get_pool(grade, topic)
    if len(pool) < 8:
        raise ValueError(f"В банке «{grade}_{topic}» слишком мало подходящих задач ({len(pool)}).")
    maze = MathMaze(cols, rows, topic, grade)
    maze.generate_path()
    return maze.fill(pool)


# ======================================================================
# 6. ОТРИСОВКА PDF
# ======================================================================
@lru_cache(maxsize=4096)
def render_tex(tex, fontsize):
    """Формула -> прозрачная PNG-картинка (байты)."""
    fig = Figure(figsize=(1, 1))
    FigureCanvasAgg(fig)
    fig.text(0, 0, f"${tex}$", fontsize=fontsize, color="black")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=RENDER_DPI, transparent=True,
                bbox_inches="tight", pad_inches=0.02)
    return buf.getvalue()


def draw_tex(c, tex, cx, cy, max_w, max_h, fontsize):
    """Рисует формулу по центру (cx, cy); если не влезает — уменьшает."""
    img = ImageReader(io.BytesIO(render_tex(tex, fontsize)))
    w_px, h_px = img.getSize()
    w, h = w_px * 72 / RENDER_DPI, h_px * 72 / RENDER_DPI
    s = min(1.0, max_w / w, max_h / h)
    w, h = w * s, h * s
    c.drawImage(img, cx - w / 2, cy - h / 2, w, h, mask="auto")


def fit_font(text, font, size, max_w, min_size=8):
    while size > min_size and pdfmetrics.stringWidth(text, font, size) > max_w:
        size -= 0.5
    return size


def draw_maze_page(c, maze, title, answers=False):
    pw, ph = A4
    margin, gx, gy = 12 * mm, 17 * mm, 11 * mm
    top, bottom = ph - 30 * mm, 14 * mm
    cols, rows = maze.cols, maze.rows
    cw = (pw - 2 * margin - (cols - 1) * gx) / cols
    ch = min((top - bottom - (rows - 1) * gy) / rows, cw * 0.75)
    grid_w, grid_h = cols * cw + (cols - 1) * gx, rows * ch + (rows - 1) * gy
    x0 = (pw - grid_w) / 2
    y_top = top - ((top - bottom) - grid_h) / 2

    def cell_xy(i):
        r, col = divmod(i, cols)
        return x0 + col * (cw + gx), y_top - r * (ch + gy) - ch

    # ---- шапка ----
    head = ("ОТВЕТЫ. " if answers else "") + title
    size = fit_font(head, "DejaVu-Bold", 18, pw - 2 * margin)
    c.setFillColorRGB(0, 0, 0)
    c.setFont("DejaVu-Bold", size)
    c.drawCentredString(pw / 2, ph - 16 * mm, head)
    c.setFont("DejaVu", 10)
    if answers:
        c.drawCentredString(pw / 2, ph - 23 * mm,
                            "Правильный путь закрашен зелёным, цифры — порядок прохождения клеток")
    else:
        c.drawString(margin, ph - 23 * mm,
                     "Фамилия, имя: ______________________________     Класс: ________     Дата: __________")

    path_set, pedges = set(maze.path), maze.path_edges()
    cell_green, corr_green = (0.80, 0.94, 0.80), (0.45, 0.80, 0.45)
    lw, sh, rad = 1.6, 1.8 * mm, 4 * mm

    # ---- тени ----
    c.setFillColorRGB(0.15, 0.15, 0.15)
    for i in range(maze.n):
        x, y = cell_xy(i)
        c.roundRect(x + sh, y - sh, cw, ch, rad, fill=1, stroke=0)

    # ---- клетки ----
    c.setLineWidth(lw)
    c.setStrokeColorRGB(0, 0, 0)
    for i in range(maze.n):
        x, y = cell_xy(i)
        c.setFillColorRGB(*(cell_green if answers and i in path_set else (1, 1, 1)))
        c.roundRect(x, y, cw, ch, rad, fill=1, stroke=1)

    # ---- коридоры с ответами ----
    corr_h, corr_w, ov = min(ch * 0.42, 13 * mm), min(cw * 0.42, 22 * mm), lw * 0.75
    for a, b in maze.edges:
        xa, ya = cell_xy(a)
        xb, yb = cell_xy(b)
        c.setFillColorRGB(*(corr_green if answers and (a, b) in pedges else (1, 1, 1)))
        if b == a + 1:                                   # горизонтальный
            x1, x2, cy = xa + cw, xb, ya + ch / 2
            c.rect(x1 - ov, cy - corr_h / 2, (x2 - x1) + 2 * ov, corr_h, fill=1, stroke=0)
            c.line(x1, cy + corr_h / 2, x2, cy + corr_h / 2)
            c.line(x1, cy - corr_h / 2, x2, cy - corr_h / 2)
            box = ((x1 + x2) / 2, cy, (x2 - x1) - 1.5 * mm, corr_h - 1.5 * mm)
        else:                                            # вертикальный
            cx, y1, y2 = xa + cw / 2, yb + ch, ya
            c.rect(cx - corr_w / 2, y1 - ov, corr_w, (y2 - y1) + 2 * ov, fill=1, stroke=0)
            c.line(cx - corr_w / 2, y1, cx - corr_w / 2, y2)
            c.line(cx + corr_w / 2, y1, cx + corr_w / 2, y2)
            box = (cx, (y1 + y2) / 2, corr_w - 1.5 * mm, (y2 - y1) - 1.5 * mm)
        draw_tex(c, fmt_tex(maze.label[(a, b)]), *box, fontsize=11)

    # ---- содержимое клеток ----
    for i in range(maze.n):
        x, y = cell_xy(i)
        if i == maze.finish:
            size = fit_font("ФИНИШ", "DejaVu-Bold", 20, cw - 4 * mm)
            c.setFillColorRGB(0, 0.45, 0)
            c.setFont("DejaVu-Bold", size)
            c.drawCentredString(x + cw / 2, y + ch / 2 - size * 0.35, "ФИНИШ")
            continue
        top_pad = 2 * mm
        if i == maze.start:
            c.setFillColorRGB(0.8, 0, 0)
            c.setFont("DejaVu-Bold", 9)
            c.drawString(x + 3 * mm, y + ch - 4.5 * mm, "СТАРТ")
            top_pad = 6 * mm
        if answers and i in path_set:
            r = 2.6 * mm
            bx, by = x + cw - r - 1.5 * mm, y + ch - r - 1.5 * mm
            c.setFillColorRGB(0.1, 0.5, 0.1)
            c.circle(bx, by, r, fill=1, stroke=0)
            c.setFillColorRGB(1, 1, 1)
            c.setFont("DejaVu-Bold", 7.5)
            c.drawCentredString(bx, by - 2.6, str(maze.path.index(i) + 1))
            top_pad = max(top_pad, 5.5 * mm)
        area_h = ch - top_pad - 1.5 * mm
        draw_tex(c, maze.task[i].tex, x + cw / 2, y + 1.5 * mm + area_h / 2,
                 cw - 5 * mm, area_h - 1.5 * mm, fontsize=15)

    # ---- подвал ----
    c.setFillColorRGB(0.35, 0.35, 0.35)
    c.setFont("DejaVu", 8)
    c.drawCentredString(pw / 2, 7 * mm, "Реши пример в клетке → найди коридор со своим ответом → "
                                        "переходи в следующую клетку → дойди до ФИНИША")


def make_pdf(topic, grade, cols, rows, variants, title, with_answers):
    mazes = [create_maze(topic, grade, cols, rows) for _ in range(variants)]
    names = [title if variants == 1 else f"{title}. Вариант {k + 1}" for k in range(variants)]
    buf = io.BytesIO()
    c = pdf_canvas.Canvas(buf, pagesize=A4)
    c.setTitle(title)
    for maze, name in zip(mazes, names):              # сначала листы учеников
        draw_maze_page(c, maze, name, answers=False)
        c.showPage()
    if with_answers:                                  # потом листы ответов
        for maze, name in zip(mazes, names):
            draw_maze_page(c, maze, name, answers=True)
            c.showPage()
    c.save()
    buf.seek(0)
    return buf


# ======================================================================
# 7. САЙТ
# ======================================================================
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "maze-secret-key")

HTML_PAGE = r"""
<!DOCTYPE html><html lang="ru"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Математические лабиринты</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',sans-serif;background:linear-gradient(135deg,#667eea,#764ba2);min-height:100vh;padding:20px}
.container{max-width:700px;margin:0 auto;background:#fff;border-radius:16px;padding:36px;box-shadow:0 20px 60px rgba(0,0,0,.3)}
header{text-align:center;margin-bottom:26px}h1{font-size:26px;color:#2c3e50;margin-bottom:6px}.sub{color:#7f8c8d;font-size:14px}
.fg{display:flex;flex-direction:column;gap:6px;margin-bottom:14px}label{font-weight:600;color:#34495e;font-size:14px}
input[type=text],select{padding:11px;border:2px solid #e0e0e0;border-radius:8px;font-size:15px;width:100%}
input:focus,select:focus{border-color:#667eea;outline:none}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:12px}.row3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
.hint{font-size:13px;color:#888;margin:-6px 0 14px}
.chk{display:flex;align-items:center;gap:8px;font-weight:400;margin:6px 0 4px}
.btn{padding:16px;background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;border:none;border-radius:10px;font-size:18px;font-weight:700;cursor:pointer;width:100%;margin-top:12px}
.btn:hover{transform:translateY(-2px);box-shadow:0 8px 25px rgba(102,126,234,.4)}
.alert{padding:12px;border-radius:8px;margin-bottom:15px;background:#fdecea;color:#c0392b}
footer{margin-top:24px;text-align:center;color:#999;font-size:12px;border-top:1px solid #eee;padding-top:12px}
@media(max-width:600px){.row2,.row3{grid-template-columns:1fr}}
</style></head><body><div class="container">
<header><h1>🧩 Математические лабиринты</h1>
<p class="sub">Дроби, 5–8 класс. Выберите параметры — получите PDF для печати и лист ответов.</p></header>
{% with msgs = get_flashed_messages() %}{% for msg in msgs %}<div class="alert">{{ msg }}</div>{% endfor %}{% endwith %}
<form action="{{ url_for('generate') }}" method="post">
<div class="row2">
  <div class="fg"><label>📚 Тема</label>
    <select name="topic" id="topic">{% for key, name in topics.items() %}<option value="{{ key }}">{{ name|capitalize }}</option>{% endfor %}</select></div>
  <div class="fg"><label>🎓 Класс</label>
    <select name="grade" id="grade">{% for g in grades %}<option value="{{ g }}">{{ g }} класс</option>{% endfor %}</select></div>
</div>
<p class="hint" id="bankinfo"></p>
<div class="row3">
  <div class="fg"><label>↔ Столбцы</label><select name="cols"><option value="3" selected>3</option><option value="4">4</option></select></div>
  <div class="fg"><label>↕ Строки</label><select name="rows">{% for r in [4,5,6,7] %}<option value="{{ r }}" {{ 'selected' if r == 6 }}>{{ r }}</option>{% endfor %}</select></div>
  <div class="fg"><label>📄 Вариантов</label><select name="variants">{% for v in [1,2,3,4] %}<option value="{{ v }}">{{ v }}</option>{% endfor %}</select></div>
</div>
<div class="fg"><label>✏️ Заголовок (необязательно)</label>
  <input type="text" name="title" maxlength="80" placeholder="Например: Лабиринт по дробям — 5 «Б»"></div>
<label class="chk"><input type="checkbox" name="answers" checked> Добавить лист ответов (лабиринт с закрашенным путём)</label>
<button class="btn" type="submit">🎲 Создать лабиринт и скачать PDF</button>
</form>
<footer>Генератор математических лабиринтов © 2026</footer>
</div>
<script>
const counts = {{ counts|tojson }};
function upd(){
  const k = document.getElementById('grade').value + '_' + document.getElementById('topic').value;
  document.getElementById('bankinfo').textContent = 'Задач в банке для этого выбора: ' + (counts[k] || 0);
}
document.getElementById('grade').onchange = upd;
document.getElementById('topic').onchange = upd;
upd();
</script>
</body></html>
"""


@app.route("/")
def index():
    counts = {f"{g}_{t}": len(get_pool(g, t)) for g in GRADES for t in TOPICS}
    return render_template_string(HTML_PAGE, grades=GRADES, topics=TOPICS, counts=counts)


@app.route("/generate", methods=["POST"])
def generate():
    try:
        topic = request.form["topic"]
        grade = int(request.form["grade"])
        cols = int(request.form.get("cols", 3))
        rows = int(request.form.get("rows", 6))
        variants = int(request.form.get("variants", 1))
        if topic not in TOPICS or grade not in GRADES or cols not in (3, 4) \
                or not 4 <= rows <= 7 or not 1 <= variants <= 4:
            raise ValueError
    except (KeyError, ValueError):
        flash("Неверные параметры формы.")
        return redirect(url_for("index"))

    title = request.form.get("title", "").strip()[:80] or f"Лабиринт: {TOPICS[topic]}, {grade} класс"
    with_answers = request.form.get("answers") == "on"
    try:
        pdf = make_pdf(topic, grade, cols, rows, variants, title, with_answers)
    except ValueError as e:
        flash(str(e))
        return redirect(url_for("index"))
    return send_file(pdf, as_attachment=True, mimetype="application/pdf",
                     download_name=f"labirint_{grade}kl_{topic}.pdf")


# ======================================================================
# 8. ПРОВЕРКА БАНКА И ПРОБНЫЙ PDF
# ======================================================================
def check_bank():
    """Печатает ошибки в ответах, нераспознанные записи и сложность банков."""
    for key, items in TASK_BANK.items():
        try:
            g, topic = key.split("_")
            int(g)
            assert topic in TOPICS
        except Exception:
            print(f"\nНепонятный ключ банка: {key}")
            continue
        good, seen, problems = [], set(), []
        for item in items:
            expr, given = _split_item(item)
            if expr in seen:
                problems.append(f"  [ПОВТОР]      {expr}")
                continue
            seen.add(expr)
            try:
                node, val = parse_task(expr)
            except (ValueError, ZeroDivisionError) as e:
                problems.append(f"  [НЕ РАЗОБРАН] {expr}   ({e})")
                continue
            if given is not None:
                try:
                    given_val = parse_task(given)[1]
                except Exception:
                    given_val = None
                if given_val != val:
                    problems.append(f"  [ОШИБКА]      {expr}   в банке: {given}   правильно: {show_value(val, topic)}")
            if not answer_ok(val, topic):
                problems.append(f"  [ПРОПУЩЕН]    {expr}   ответ {show_value(val, topic)} не годится для коридора")
                continue
            good.append(complexity(node))
        avg = sum(good) / len(good) if good else 0
        print(f"\n=== {key}: записей {len(items)}, годных {len(good)}, средняя сложность {avg:.1f}")
        print("\n".join(problems) if problems else "  проблем нет")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check_bank()
        sys.exit()

    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        g = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        t = sys.argv[3] if len(sys.argv) > 3 else "fraction"
        out = make_pdf(t, g, 3, 6, 1, f"Лабиринт: {TOPICS[t]}, {g} класс", True)
        name = f"demo_{g}_{t}.pdf"
        with open(name, "wb") as f:
            f.write(out.read())
        print(f"Готово: {name}")
        sys.exit()

    port = int(os.environ.get("PORT", 5000))
    print(f"Сайт: http://127.0.0.1:{port}   (остановить: Ctrl+C)")
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")
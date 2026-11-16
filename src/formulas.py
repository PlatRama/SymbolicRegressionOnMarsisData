from __future__ import annotations

import string
import tomllib
import warnings

import numpy as np
import sympy
from scipy.optimize import least_squares
from sympy.parsing.sympy_parser import convert_xor, parse_expr, standard_transformations

from config import FORMULAS_FILE

_POS_FLOOR = 1e-9  # keeps pos(x)^p differentiable for p < 1

_PARSE_FUNCS = {
    "log10": lambda x: sympy.log(x, 10),
    "log": sympy.log,
    "exp": sympy.exp,
    "sqrt": sympy.sqrt,
    "square": lambda x: x ** 2,
    "abs": sympy.Abs,
    "pos": sympy.Function("pos"),
}
_NUMPY_FUNCS = {"pos": lambda x: np.maximum(x, _POS_FLOOR)}


def parse(expression, symbols=()):
    local = dict(_PARSE_FUNCS)
    local.update({s: sympy.Symbol(s) for s in symbols})
    return parse_expr(expression, local_dict=local, transformations=standard_transformations + (convert_xor,))


class Formula:
    def __init__(self, name, expression, params, offset_by=None, offset_init=0.0, description="", origin=""):
        self.name = name
        self.expression = expression
        self.params = dict(params)
        self.offset_by = offset_by
        self.offset_init = float(offset_init)
        self.description = description
        self.origin = origin

        self.expr = parse(expression, symbols=self.params)
        names = {str(s) for s in self.expr.free_symbols}
        missing = set(self.params) - names
        if missing:
            raise ValueError(f"formula {name}: parameters not used in the expression: {missing}")
        self.param_names = list(self.params)
        self.variables = sorted(names - set(self.params))
        symbols = [sympy.Symbol(n) for n in self.param_names + self.variables]
        self._func = sympy.lambdify(symbols, self.expr, modules=[_NUMPY_FUNCS, "numpy"])

    @property
    def features(self):
        return self.variables + ([self.offset_by] if self.offset_by else [])

    def evaluate(self, columns, param_values):
        """Value without the per-group offset. columns: name -> array."""
        out = self._func(*list(param_values), *[columns[v] for v in self.variables])
        return np.broadcast_to(out, len(next(iter(columns.values())))).astype(float)

    def latex(self):
        def name(n):
            return r"\mathrm{" + n.replace("_", r"\_") + "}"
        names = {sympy.Symbol(v): name(v) for v in self.variables}
        names.update({sympy.Symbol(p): p.replace("_", "_{", 1) + "}" if "_" in p else p for p in self.param_names})
        tex = sympy.latex(self.expr, symbol_names=names)
        return tex + (rf" + G({name(self.offset_by)})" if self.offset_by else "")

    def __repr__(self):
        return f"Formula({self.name}: {self.expression}{f' + G[{self.offset_by}]' if self.offset_by else ''})"


def load_formula(name):
    with open(FORMULAS_FILE, "rb") as f:
        entries = tomllib.load(f)
    if name not in entries:
        raise KeyError(f"formula {name!r} not found in {FORMULAS_FILE}. Available: {list(entries)}")
    e = entries[name]
    return Formula(name, e["expression"], e.get("params", {}), offset_by=e.get("offset_by"),
                   offset_init=e.get("offset_init", 0.0), description=e.get("description", ""),
                   origin=e.get("origin", ""))


class FormulaModel:
    def __init__(self, formula, fit_sample=100_000, seed=0):
        self.formula = formula if isinstance(formula, Formula) else load_formula(formula)
        self.fit_sample = fit_sample
        self.seed = seed

    def _columns(self, X):
        return {f: X[f].to_numpy(dtype=float) for f in self.formula.features}

    def _offsets(self, group_values, offsets):
        lookup = dict(zip(self.groups_, offsets))
        unknown = set(np.unique(group_values)) - set(self.groups_)
        if unknown:
            warnings.warn(f"{self.formula.name}: groups not seen in training {unknown}: using the mean offset")
        default = float(np.mean(offsets))
        return np.array([lookup.get(g, default) for g in group_values])

    def _predict(self, cols, theta):
        n = len(self.formula.param_names)
        out = self.formula.evaluate(cols, theta[:n])
        if self.formula.offset_by:
            out = out + self._offsets(cols[self.formula.offset_by], theta[n:])
        return out

    def fit(self, X, y):
        if len(X) > self.fit_sample:
            idx = np.random.default_rng(self.seed).choice(len(X), self.fit_sample, replace=False)
            X, y = X.iloc[idx], y.iloc[idx]
        cols, y = self._columns(X), np.asarray(y, dtype=float)

        theta0 = list(self.formula.params.values())
        self.groups_ = []
        if self.formula.offset_by:
            self.groups_ = sorted(np.unique(cols[self.formula.offset_by]).tolist())
            theta0 += [self.formula.offset_init] * len(self.groups_)

        res = least_squares(lambda t: self._predict(cols, t) - y, np.asarray(theta0, dtype=float),
                            loss="soft_l1", f_scale=1.0)
        self.theta_ = res.x
        names = self.formula.param_names + [f"G[{g:g}]" for g in self.groups_]
        self.params_ = dict(zip(names, res.x))
        return self

    def predict(self, X):
        return self._predict(self._columns(X), self.theta_)


def formula_from_pysr(equation, name, description="", origin=""):
    """Turn a PySR equation into a formulas.toml entry: every float becomes a parameter a, b, c, ..."""
    expr = parse(equation) if isinstance(equation, str) else equation
    floats = sorted(expr.atoms(sympy.Float), key=str)
    letters = [c for c in string.ascii_lowercase if c not in {"e", "i"}]
    params = {}
    for value, letter in zip(floats, letters):
        expr = expr.subs(value, sympy.Symbol(letter))
        params[letter] = float(value)
    params_text = ", ".join(f"{k} = {v:.8g}" for k, v in params.items())
    return (f"[{name}]\n"
            f'description = "{description}"\n'
            f'expression  = "{sympy.sstr(expr).replace("**", "^")}"\n'
            f"params      = {{ {params_text} }}\n"
            f'origin      = "{origin}"\n')

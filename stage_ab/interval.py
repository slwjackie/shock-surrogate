"""Outward binary64 intervals and interval forward-mode differentiation.

Assumptions: IEEE-754 correctly rounded elementary operations, gradual
underflow, and Python Decimal's correctly rounded exp/ln. No BLAS reductions,
fast-math, sampled Jacobians, or ordinary exp/log are used for enclosures.
This is an arithmetic implementation, not a proof-assistant verification.
"""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal, localcontext
import math
from fractions import Fraction


def down(x: float) -> float:
    if not math.isfinite(x):
        raise ArithmeticError("Nonfinite interval operation; certificate unavailable")
    return math.nextafter(x, -math.inf)


def up(x: float) -> float:
    if not math.isfinite(x):
        raise ArithmeticError("Nonfinite interval operation; certificate unavailable")
    return math.nextafter(x, math.inf)


def upper_fraction(x: Fraction) -> float:
    """Round a rational upper bound upward, including summation roundoff."""
    y = float(x)
    if not math.isfinite(y):
        raise ArithmeticError("Bound overflow")
    return math.nextafter(y, math.inf) if Fraction(y) < x else y


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float | None = None

    def __post_init__(self):
        object.__setattr__(self, "lo", float(self.lo))
        object.__setattr__(self, "hi", float(self.lo if self.hi is None else self.hi))
        if not (math.isfinite(self.lo) and math.isfinite(self.hi) and self.lo <= self.hi):
            raise ArithmeticError("Invalid/nonfinite interval")

    @staticmethod
    def cast(x):
        return x if isinstance(x, Interval) else Interval(float(x))

    def __add__(self, other):
        if hasattr(other, "grad"):
            return NotImplemented
        b = self.cast(other)
        return Interval(down(self.lo + b.lo), up(self.hi + b.hi))
    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        if hasattr(other, "grad"):
            return NotImplemented
        return self + (-self.cast(other))

    def __rsub__(self, other):
        return self.cast(other) - self

    def __mul__(self, other):
        if hasattr(other, "grad"):
            return NotImplemented
        b = self.cast(other)
        v = [self.lo*b.lo, self.lo*b.hi, self.hi*b.lo, self.hi*b.hi]
        return Interval(down(min(v)), up(max(v)))
    __rmul__ = __mul__

    def reciprocal(self):
        if self.lo <= 0 <= self.hi:
            raise ArithmeticError("Interval denominator contains zero")
        return Interval(down(1.0/self.hi), up(1.0/self.lo))

    def __truediv__(self, other):
        if hasattr(other, "grad"):
            return NotImplemented
        return self * self.cast(other).reciprocal()

    def __rtruediv__(self, other):
        return self.cast(other) * self.reciprocal()

    def __pow__(self, power):
        if not isinstance(power, int):
            return (self.log()*float(power)).exp()
        if power < 0:
            return (self**(-power)).reciprocal()
        if power == 0:
            return Interval(1)
        if power == 1:
            return self
        if power == 2:
            a, b = self.lo*self.lo, self.hi*self.hi
            low = 0.0 if self.lo <= 0 <= self.hi else down(min(a, b))
            return Interval(low, up(max(a, b)))
        # Binary exponentiation, with interval-safe products.
        result, base, k = Interval(1), self, power
        while k:
            if k % 2:
                result = result*base
            k //= 2
            if k:
                base = base*base
        return result

    def _transcendental(self, name):
        if name == "ln" and self.lo <= 0:
            raise ArithmeticError("Log interval must be positive")
        with localcontext() as ctx:
            ctx.prec = 80
            left = getattr(Decimal.from_float(self.lo), name)().next_minus()
            right = getattr(Decimal.from_float(self.hi), name)().next_plus()
            return Interval(down(float(left)), up(float(right)))

    def exp(self):
        return self._transcendental("exp")

    def log(self):
        return self._transcendental("ln")

    def relu(self):
        return Interval(max(0., self.lo), max(0., self.hi))

    def abs_upper(self):
        return max(abs(self.lo), abs(self.hi))

    def hull(self, other):
        b = self.cast(other)
        return Interval(min(self.lo, b.lo), max(self.hi, b.hi))

    def contains(self, x):
        return self.lo <= x <= self.hi


@dataclass(frozen=True)
class Dual:
    """Interval value and interval gradient (no finite differences)."""
    value: Interval
    grad: tuple[Interval, ...]

    @classmethod
    def variable(cls, value, i, size):
        return cls(Interval.cast(value), tuple(Interval(int(k == i)) for k in range(size)))

    def cast(self, x):
        return x if isinstance(x, Dual) else Dual(Interval.cast(x), tuple(Interval(0) for _ in self.grad))

    def __add__(self, other):
        b = self.cast(other)
        return Dual(self.value+b.value, tuple(x+y for x, y in zip(self.grad, b.grad)))
    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.value, tuple(-x for x in self.grad))

    def __sub__(self, other):
        return self + (-self.cast(other))

    def __rsub__(self, other):
        return self.cast(other) - self

    def __mul__(self, other):
        b = self.cast(other)
        return Dual(self.value*b.value, tuple(x*b.value+y*self.value for x, y in zip(self.grad, b.grad)))
    __rmul__ = __mul__

    def reciprocal(self):
        return Dual(self.value.reciprocal(), tuple(-x/(self.value**2) for x in self.grad))

    def __truediv__(self, other):
        return self * self.cast(other).reciprocal()

    def __rtruediv__(self, other):
        return self.cast(other) * self.reciprocal()

    def exp(self):
        y = self.value.exp()
        return Dual(y, tuple(y*x for x in self.grad))

    def log(self):
        return Dual(self.value.log(), tuple(x/self.value for x in self.grad))

    def __pow__(self, power):
        if not isinstance(power, int):
            return (self.log()*float(power)).exp()
        if power == 0:
            return self.cast(1)
        return Dual(self.value**power, tuple(power*(self.value**(power-1))*x for x in self.grad))


def value(x):
    return x.value if isinstance(x, Dual) else Interval.cast(x)

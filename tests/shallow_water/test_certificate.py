"""Compare compiled depth certificates against independent Decimal calculations."""
import ctypes
from decimal import Decimal, localcontext
import math
from pathlib import Path
import random
import subprocess
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[2]
D = Decimal.from_float


def reference_phi(hl, ul, hr, ur, g, x):
    hl, ul, hr, ur, g, x = map(D, (hl, ul, hr, ur, g, x))

    def wave(h):
        if x <= h:
            return 2 * ((g*x).sqrt() - (g*h).sqrt())
        return (x-h) * (g*(x+h)/(2*x*h)).sqrt()

    return wave(hl) + wave(hr) + ur - ul


class DepthCertificateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = TemporaryDirectory()
        path = Path(cls.tmp.name)
        (path / 'cert.c').write_text(
            '#include "certificate.h"\n'
            'int sign_of(double a,double b,double c,double d,double g,double x)'
            '{return sw_sign_at(a,b,c,d,g,x);}\n'
            'int cert(double a,double b,double c,double d,double g,double x,double t)'
            '{return sw_certify(a,b,c,d,g,x,t);}\n')
        so = path / 'cert.so'
        subprocess.run(['cc', '-std=c99', '-O3', '-fno-fast-math', '-ffp-contract=off',
                        '-shared', '-fPIC', '-I', str(ROOT / 'shallow_water'),
                        str(path / 'cert.c'), '-lm', '-o', str(so)], check=True)
        lib = ctypes.CDLL(str(so))
        cls.sign_of = lib.sign_of
        cls.sign_of.argtypes = [ctypes.c_double] * 6
        cls.cert = lib.cert
        cls.cert.argtypes = [ctypes.c_double] * 7

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_analytical_depth_and_inward_tolerance(self):
        # Two rarefactions: 4*(sqrt(h)-2)+4=0 gives h*=1.
        state = (4., -2., 4., 2., 1.)
        candidate = 1.0625
        cutoff = .0625
        self.assertFalse(self.cert(*state, candidate, math.nextafter(cutoff, 0)))
        self.assertTrue(self.cert(*state, candidate, cutoff+1.e-12))
        self.assertTrue(self.cert(4., .5, 4., .5, 1., 4., 0.))
        # An interval clipping zero still certifies a strictly wet solution.
        self.assertTrue(self.cert(*state, .6, .6))

    def test_wet_dry_boundary_and_invalid_values(self):
        self.assertEqual(self.sign_of(1., -2., 1., 2., 1., 0.), 0)
        self.assertEqual(self.sign_of(1., -2., 1., 2.1, 1., 0.), 1)
        self.assertFalse(self.cert(1., -2., 1., 2., 1., .01, .1))
        self.assertFalse(self.cert(0., 0., 1., 0., 1., 1., .1))
        self.assertFalse(self.cert(1., 0., 1., 0., 1., math.nan, .1))

    def test_signs_near_branch_switches_and_cancellation(self):
        rng = random.Random(897372)
        nonzero = 0
        with localcontext() as ctx:
            ctx.prec = 180
            for _ in range(700):
                hl, hr, g = [math.ldexp(rng.uniform(1, 2), rng.randint(-115, 115))
                             for _ in range(3)]
                ul = rng.uniform(-5, 5)
                for x in (hl, math.nextafter(hl, hr), (hl+hr)/2,
                          math.ldexp(hl, rng.randrange(-10, 10))):
                    # Set right velocity to cancel both wave terms, rounded to a double.
                    center = float(-reference_phi(hl, ul, hr, 0., g, x))
                    for ur in (center, math.nextafter(center, -math.inf),
                               math.nextafter(center, math.inf), rng.uniform(-5, 5)):
                        sign = self.sign_of(hl, ul, hr, ur, g, x)
                        if sign:
                            exact = reference_phi(hl, ul, hr, ur, g, x)
                            self.assertEqual(sign, int(exact > 0) - int(exact < 0))
                            nonzero += 1
        self.assertGreater(nonzero, 1000)

    def test_dam_break_and_wet_states_agree_with_decimal(self):
        self.assertTrue(self.cert(4., 0., 1., 0., 9.81, 2.21, .01))
        self.assertFalse(self.cert(4., 0., 1., 0., 9.81, 2.24, .01))
        rng = random.Random(29387)
        with localcontext() as ctx:
            ctx.prec = 95
            for _ in range(90):
                hl, hr = [rng.uniform(.01, 6) for _ in range(2)]
                ul, ur = [rng.uniform(-1., 1.) for _ in range(2)]
                state = (hl, ul, hr, ur, 9.81)
                low, high = 0., 20.
                for _ in range(60):
                    mid = (low+high)/2
                    if reference_phi(*state, mid) < 0:
                        low = mid
                    else:
                        high = mid
                tol = 0.001
                self.assertTrue(self.cert(*state, low + .75*tol, tol))
                self.assertFalse(self.cert(*state, high + 1.25*tol, tol))


if __name__ == '__main__':
    unittest.main()

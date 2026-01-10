import numpy as np
from scipy.interpolate import BSpline


class LinkFunctionSpline:
    def __init__(self, k, eta_min, eta_max, degree=3, clamp=True):
        self.k = int(k)
        self.degree = int(degree)
        self.eta_min = float(eta_min)
        self.eta_max = float(eta_max)
        self.clamp = bool(clamp)
        if self.k < self.degree + 1:
            raise ValueError("link_k must be >= degree + 1")
        if not (self.eta_min < 0.0 < self.eta_max):
            raise ValueError("link_eta_min must be < 0 and link_eta_max > 0")
        self._knots = self._build_knots()
        self._basis = []
        self._basis_prime = []
        for i in range(self.k):
            coeff = np.zeros(self.k, dtype=float)
            coeff[i] = 1.0
            spline = BSpline(self._knots, coeff, self.degree, extrapolate=False)
            self._basis.append(spline)
            self._basis_prime.append(spline.derivative(1))
        self._b0 = np.array([float(b(0.0)) for b in self._basis], dtype=float)
        self._b0_prime = np.array([float(b(0.0)) for b in self._basis_prime], dtype=float)

    def _build_knots(self) -> np.ndarray:
        n_internal = self.k - self.degree - 1
        if n_internal < 0:
            raise ValueError("link_k is too small for the spline degree")
        if n_internal == 0:
            internal = np.array([], dtype=float)
        else:
            internal = np.linspace(self.eta_min, self.eta_max, n_internal + 2)[1:-1]
        left = np.repeat(self.eta_min, self.degree + 1)
        right = np.repeat(self.eta_max, self.degree + 1)
        return np.concatenate([left, internal, right]).astype(float)

    def basis_values(self, eta: float):
        eta_eval = float(eta)
        clamped = False
        if self.clamp:
            if eta_eval < self.eta_min:
                eta_eval = self.eta_min
                clamped = True
            elif eta_eval > self.eta_max:
                eta_eval = self.eta_max
                clamped = True
        b_vals = np.array([float(b(eta_eval)) for b in self._basis], dtype=float)
        b_prime = np.array([float(b(eta_eval)) for b in self._basis_prime], dtype=float)
        phi = b_vals - self._b0 - eta_eval * self._b0_prime
        if clamped:
            phi_prime = np.zeros_like(phi)
        else:
            phi_prime = b_prime - self._b0_prime
        return phi, phi_prime

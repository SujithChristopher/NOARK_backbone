
import math

class NoarkKinematics:
    
    def internal_tangent_slopes(self, r1, r2, L, t, tol=1e-12):
        """
        Compute slopes (m) of the internal tangents
        Returns the two slopes (m) of the internal tangents between:
        C1 at (0,0), radius r1
        C2 at (L,t), radius r2
        """
        R = r1 + r2
        denom = L*L - R*R
        inside = L*L + t*t - R*R

        if inside < -tol:
            raise ValueError("No real internal tangents exist.")
        inside = max(inside, 0.0)

        if abs(denom) < tol:
            raise ValueError("Degenerate case: vertical tangent only.")

        sqrt_term = math.sqrt(inside)

        m_plus  = (L*t + R*sqrt_term) / denom
        m_minus = (L*t - R*sqrt_term) / denom

        return m_plus, m_minus

    def compute_b(self, m, r1, r2, L, t):
        """
        Compute intercept b for a given m
        b = r1 * (t - mL) / (r1 + r2)
        """
        return r1 * (t - m * L) / (r1 + r2)


    def tangent_point(self, xc, yc, r, m, b):
        """
        Compute tangent point for circle at (xc, yc), radius r
        Returns tangent point (xp, yp) for circle (xc, yc, r)
        and line y = m x + b.
        """
        k = math.sqrt(1 + m*m)

        # Sign chosen so distance = +r or -r depending on which side
        # General formula:
        xp = xc - m * (m*xc - yc + b) / (1 + m*m)
        yp = yc + (m*xc - yc + b) / (1 + m*m)

        return xp, yp

    def internal_tangents(self, r1, r2, L, t):
        """
        Returns both internal tangent lines and tangent points.
        Wrapper to compute the two complete tangent solutions
        """
        m1, m2 = self.internal_tangent_slopes(r1, r2, L, t)
        solutions = []

        for m in (m1, m2):
            b = self.compute_b(m, r1, r2, L, t)
            # Tangent point on circle C1 = (0,0)
            t1 = self.tangent_point(0, 0, r1, m, b)

            # Tangent point on circle C2 = (L, t)
            t2 = self.tangent_point(L, t, r2, m, b)

            solutions.append({
                "m": m,
                "b": b,
                "tC1": t1,
                "tC2": t2
            })

        return solutions       
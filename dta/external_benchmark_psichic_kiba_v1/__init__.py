"""PSICHIC KIBA adapter preflight.

The full PSICHIC architecture requires per-residue ESM representations and an
ESM-predicted contact graph.  The existing v21 ESM LMDB intentionally stores
pooled sequence embeddings and therefore is not silently treated as PSICHIC
input.  This package audits that boundary and reports the exact cache needed
before a formal PSICHIC run is allowed.
"""


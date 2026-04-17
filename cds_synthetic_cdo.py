"""
CDS panel loading, EDA-style plots, dependence / PCA helpers,
and synthetic CDO pricing (Gaussian copula, Mounfield-style tranche analytics).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.integrate import quad
from scipy.optimize import brentq

# --- Paths & config -----------------------------------------------------------


@dataclass(frozen=True)
class ProjectPaths:
    """Project root and standard filenames."""

    base: Path

    @property
    def cds_csv(self) -> Path:
        return self.base / "cds.csv"


@dataclass
class DependencePlotParams:
    """Tham số cho `CDSPanel.plot_dependence_quadrant`."""

    min_obs_frac: float = 0.75
    n_heatmap_cols: int = 35
    rolling_window: int = 60
    seed: int = 0
    copula_rho: float = 0.45
    copula_n_paths: int = 5000
    copula_seed: int = 1


@dataclass
class CdoDashboardPlotParams:
    """Tham số cho `SyntheticCDOEngine.plot_cdo_dashboard`."""

    rho_min: float = 0.05
    rho_max: float = 0.45
    rho_n: int = 9
    mc_paths: int = 15_000
    equity_attach: float = 0.0
    equity_detach: float = 0.03
    hist_bins: int = 60
    tranche_hist_bins: int = 40


@dataclass
class MounfieldPlotParams:
    """Tham số cho `SyntheticCDOEngine.mounfield_figures` (median PD, LHP, base-corr demo)."""

    rho_demo: float = 0.28
    d_test: float = 0.03
    rho_curve_target: float = 0.30
    K_min: float = 0.03
    K_max: float = 0.18
    K_n: int = 10


@dataclass
class SyntheticCDOParams:
    recovery: float = 0.40
    val_date: str = "2015-06-15"
    pricing_tenor_years: float = 5.0
    pool_size: int = 80
    rng_seed: int = 42
    n_mc: int = 40_000
    rho: float = 0.22
    tranches: Tuple[Tuple[str, float, float], ...] = (
        ("Equity", 0.0, 0.03),
        ("Mezz", 0.03, 0.07),
        ("Senior", 0.07, 0.10),
    )
    dependence_plot: DependencePlotParams = field(default_factory=DependencePlotParams)
    dashboard_plot: CdoDashboardPlotParams = field(default_factory=CdoDashboardPlotParams)
    mounfield_plot: MounfieldPlotParams = field(default_factory=MounfieldPlotParams)

    @property
    def px_column(self) -> str:
        return f"PX{int(self.pricing_tenor_years)}"


# --- Credit curve & copula (stateless) ----------------------------------------


class CreditCurve:
    """Spread → hazard / PD helpers (piecewise constant hazard between tenors)."""

    TENORS: np.ndarray = np.arange(1, 11, dtype=float)
    PX_COLUMNS: Tuple[str, ...] = tuple(f"PX{i}" for i in range(1, 11))

    @staticmethod
    def spread_to_hazard_continuous(spread_bp: np.ndarray, recovery: float) -> np.ndarray:
        s = np.asarray(spread_bp, dtype=float) / 10_000.0
        return s / np.maximum(1e-12, (1.0 - recovery))

    @classmethod
    def default_prob_piecewise(
        cls,
        spread_bp_row: np.ndarray,
        tenors: np.ndarray,
        T: float,
        recovery: float,
    ) -> float:
        lam = cls.spread_to_hazard_continuous(spread_bp_row, recovery)
        S = 1.0
        t_prev = 0.0
        for i, tenor in enumerate(tenors):
            dt = min(tenor, T) - t_prev
            if dt <= 0:
                break
            S *= np.exp(-lam[i] * dt)
            t_prev = min(tenor, T)
            if t_prev >= T:
                break
        if T > tenors[-1]:
            S *= np.exp(-lam[-1] * (T - tenors[-1]))
        return float(1.0 - S)

    @classmethod
    def homogeneous_pd_from_single_spread(
        cls, spread_bp: np.ndarray, T: float, recovery: float
    ) -> np.ndarray:
        lam = cls.spread_to_hazard_continuous(spread_bp, recovery)
        return 1.0 - np.exp(-lam * T)


class GaussianCopulaCDO:
    """One-factor Gaussian copula; portfolio loss & tranche losses."""

    @staticmethod
    def simulate_portfolio_loss_fraction(
        p: np.ndarray,
        rho: float,
        recovery: float,
        n_mc: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        n = len(p)
        p = np.clip(p, 1e-6, 1 - 1e-6)
        c = stats.norm.ppf(p)
        y = rng.standard_normal(n_mc)
        z = (c - np.sqrt(rho) * y[:, None]) / np.sqrt(max(1e-12, 1.0 - rho))
        p_cond = stats.norm.cdf(z)
        u = rng.random((n_mc, n))
        defaults = (u < p_cond).sum(axis=1)
        return (1.0 - recovery) * defaults / n

    @staticmethod
    def tranche_loss_fraction(L: np.ndarray, attach: float, detach: float) -> np.ndarray:
        width = max(1e-12, detach - attach)
        return np.minimum(np.maximum(L - attach, 0.0), detach - attach) / width


# --- Mounfield-style integrals ------------------------------------------------


class MounfieldTranche:
    """Equity tranche EL via integration; LHP; implied ρ (base correlation demo)."""

    @staticmethod
    def p_cond_homogeneous(pd: float, rho: float, y: float) -> float:
        c = stats.norm.ppf(np.clip(pd, 1e-6, 1 - 1e-6))
        return float(
            stats.norm.cdf((c - np.sqrt(rho) * y) / np.sqrt(max(1e-12, 1.0 - rho)))
        )

    @classmethod
    def equity_min_loss_given_z(cls, pz: float, n: int, recovery: float, d: float) -> float:
        s = 0.0
        for k in range(n + 1):
            prob = stats.binom.pmf(k, n, pz)
            Lk = (1.0 - recovery) * k / n
            s += prob * min(Lk, d)
        return s

    @classmethod
    def equity_el_ratio_finite_homogeneous(
        cls, pd: float, rho: float, d: float, n: int, recovery: float
    ) -> float:

        def integrand(y: float) -> float:
            pz = cls.p_cond_homogeneous(pd, rho, y)
            return cls.equity_min_loss_given_z(pz, n, recovery, d) * stats.norm.pdf(y)

        num, _ = quad(integrand, -10.0, 10.0, limit=120)
        return num / max(1e-12, d)

    @classmethod
    def equity_el_ratio_lhp(cls, pd: float, rho: float, d: float, recovery: float) -> float:
        c = stats.norm.ppf(np.clip(pd, 1e-6, 1 - 1e-6))

        def integrand(y: float) -> float:
            pz = stats.norm.cdf((c - np.sqrt(rho) * y) / np.sqrt(max(1e-12, 1.0 - rho)))
            L = (1.0 - recovery) * pz
            return min(L, d) * stats.norm.pdf(y)

        num, _ = quad(integrand, -10.0, 10.0, limit=80)
        return num / max(1e-12, d)

    @classmethod
    def implied_rho_equity_tranche(
        cls,
        pd: float,
        d: float,
        n: int,
        recovery: float,
        target_el_ratio: float,
        lo: float = 0.03,
        hi: float = 0.75,
    ) -> float:

        def gap(rho: float) -> float:
            return cls.equity_el_ratio_finite_homogeneous(pd, rho, d, n, recovery) - target_el_ratio

        return brentq(gap, lo, hi)


# --- CDS panel ----------------------------------------------------------------


class CDSPanel:
    """Long-format CDS table: Date × Ticker × PX1..PX10."""

    PX_COLUMNS: Tuple[str, ...] = tuple(f"PX{i}" for i in range(1, 11))

    def __init__(self, frame: pd.DataFrame):
        self._df = frame.copy()
        if "Date" in self._df.columns:
            self._df["Date"] = pd.to_datetime(self._df["Date"])

    @classmethod
    def load(cls, csv_path: Path | str) -> CDSPanel:
        return cls(pd.read_csv(csv_path))

    @property
    def frame(self) -> pd.DataFrame:
        return self._df

    def wide_tenor(self, tenor_years: int) -> pd.DataFrame:
        col = f"PX{tenor_years}"
        return self._df.pivot_table(index="Date", columns="Ticker", values=col)

    def plot_eda_summary(self) -> plt.Figure:
        df = self._df
        px_cols = list(self.PX_COLUMNS)
        snap = df["Date"].value_counts().sort_index()
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        ax = axes[0, 0]
        ax.plot(snap.index, snap.values, lw=0.8, color="steelblue")
        ax.set_title("Số ticker có dữ liệu theo ngày")
        ax.set_ylabel("Count")

        ax = axes[0, 1]
        s5 = df["PX5"].dropna()
        ax.hist(s5, bins=80, color="coral", edgecolor="white", alpha=0.9)
        ax.set_title("Phân phối PX5 (5Y spread, bp) — toàn bộ quan sát")
        ax.set_xlabel("bp")

        ax = axes[1, 0]
        for k, c in zip([1, 5, 10], ["#333", "#1f77b4", "#2ca02c"]):
            col = f"PX{k}"
            ax.hist(df[col].dropna(), bins=60, alpha=0.35, label=f"{col} ({k}Y)", color=c)
        ax.legend()
        ax.set_title("So sánh phân phối PX1 / PX5 / PX10")
        ax.set_xlabel("bp")

        ax = axes[1, 1]
        cs = df.groupby("Date")["PX5"].mean()
        cs_med = df.groupby("Date")["PX5"].median()
        ax.plot(cs.index, cs.values, label="Mean PX5", lw=1)
        ax.plot(cs_med.index, cs_med.values, label="Median PX5", lw=1, alpha=0.8)
        ax.set_title("Spread 5Y trung bình / trung vị theo thời gian (mặt cắt)")
        ax.legend()
        ax.set_ylabel("bp")

        plt.tight_layout()
        return fig

    def describe_px(self) -> pd.DataFrame:
        return self._df[list(self.PX_COLUMNS)].describe(percentiles=[0.05, 0.5, 0.95]).T.round(2)

    def plot_dependence_quadrant(
        self,
        cfg: Optional[DependencePlotParams] = None,
    ) -> plt.Figure:
        cfg = cfg or DependencePlotParams()
        wide5 = self.wide_tenor(5)
        th = int(cfg.min_obs_frac * len(wide5))
        ok = wide5.columns[wide5.notna().sum() >= th]
        wide5_ok = wide5[ok].dropna(how="all")
        X = wide5_ok.diff().dropna()
        if X.shape[1] < 2:
            raise ValueError("Không đủ cột sau lọc — giảm `dependence_plot.min_obs_frac`.")

        rng = np.random.default_rng(cfg.seed)
        n_show = min(cfg.n_heatmap_cols, X.shape[1])
        cols_show = list(rng.choice(X.columns, size=n_show, replace=False))
        Xsub = X[cols_show]
        C = Xsub.corr()

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax = axes[0, 0]
        im = ax.imshow(C.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(C.columns)))
        ax.set_xticklabels(C.columns, rotation=90, fontsize=6)
        ax.set_yticks(range(len(C.columns)))
        ax.set_yticklabels(C.columns, fontsize=6)
        ax.set_title(f"Corr(ΔPX5) — {n_show} ticker")
        plt.colorbar(im, ax=ax, fraction=0.046)

        ax = axes[0, 1]
        win = cfg.rolling_window
        if len(X) > win + 5:
            pair_mean = []
            for t in range(win, len(X)):
                sub = X.iloc[t - win : t]
                corr = sub.corr().values
                m = np.triu_indices_from(corr, k=1)
                pair_mean.append(np.nanmean(np.abs(corr[m])))
            ax.plot(X.index[win:], pair_mean, color="darkgreen", lw=1.2)
            ax.set_title(f"Trung bình |corr| cặp (rolling {win} ngày, ΔPX5)")
        else:
            ax.text(0.5, 0.5, "Chuỗi quá ngắn cho rolling", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Rolling |corr| (không đủ dữ liệu)")
        ax.set_ylabel("Mean |ρ|")

        if "AAPL" in wide5.columns and "MSFT" in wide5.columns:
            a, b = "AAPL", "MSFT"
        else:
            a, b = cols_show[0], cols_show[1]
        ax = axes[1, 0]
        s = wide5[[a, b]].dropna()
        ax.scatter(s[a], s[b], s=8, alpha=0.35, c="steelblue")
        ax.set_xlabel(f"{a} PX5 (bp)")
        ax.set_ylabel(f"{b} PX5 (bp)")
        ax.set_title(f"Spread 5Y: {a} vs {b}")

        rho_c = cfg.copula_rho
        n = cfg.copula_n_paths
        rng2 = np.random.default_rng(cfg.copula_seed)
        Y = rng2.standard_normal(n)
        e1 = rng2.standard_normal(n)
        e2 = rng2.standard_normal(n)
        Z1 = np.sqrt(rho_c) * Y + np.sqrt(1 - rho_c) * e1
        Z2 = np.sqrt(rho_c) * Y + np.sqrt(1 - rho_c) * e2
        ax = axes[1, 1]
        ax.scatter(Z1, Z2, s=4, alpha=0.25, c="purple")
        ax.set_title(f"Copula Gauss (minh họa): corr ≈ {rho_c:.2f}")
        ax.set_xlabel("Z1")
        ax.set_ylabel("Z2")
        ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()
        return fig

    def dependence_summary(self, cfg: Optional[DependencePlotParams] = None) -> None:
        cfg = cfg or DependencePlotParams()
        wide5 = self.wide_tenor(5)
        th = int(cfg.min_obs_frac * len(wide5))
        ok = wide5.columns[wide5.notna().sum() >= th]
        wide5_ok = wide5[ok].dropna(how="all")
        X = wide5_ok.diff().dropna()
        rng = np.random.default_rng(cfg.seed)
        n_show = min(cfg.n_heatmap_cols, X.shape[1])
        cols_show = list(rng.choice(X.columns, size=n_show, replace=False))
        if "AAPL" in wide5.columns and "MSFT" in wide5.columns:
            a, b = "AAPL", "MSFT"
        else:
            a, b = cols_show[0], cols_show[1]
        print(f"Corr mức spread {a} vs {b} (PX5): {wide5[a].corr(wide5[b]):.2f}")
        print(f"Corr mức thay đổi ΔPX5: {X[a].corr(X[b]):.2f}")

    def plot_pca_pc1(self, min_obs_frac: float = 0.8) -> Optional[plt.Figure]:
        from numpy.linalg import eigh

        wide5 = self.wide_tenor(5)
        th = int(min_obs_frac * len(wide5))
        cols_ok = wide5.columns[wide5.notna().sum() >= th]
        Xraw = wide5[cols_ok].diff().dropna()
        Xz = (Xraw - Xraw.mean()) / Xraw.std(ddof=0).replace(0, np.nan)
        Xz = Xz.dropna(axis=1, how="any")
        if Xz.shape[1] < 5:
            print("Không đủ cột sau chuẩn hóa — tăng ngưỡng hoặc bỏ qua PCA.")
            return None
        Z = Xz.values
        C = np.cov(Z, rowvar=False, bias=True)
        w, V = eigh(C)
        idx = np.argsort(w)[::-1]
        w, V = w[idx], V[:, idx]
        pc1 = Z @ V[:, 0]
        var_ratio = w[0] / w.sum()
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(Xz.index, pc1, color="teal", lw=1)
        ax.set_title(f"PC1 của ΔPX5 (chuẩn hóa) — tỷ lệ phương sai ≈ {var_ratio:.1%}")
        ax.set_ylabel("PC1")
        plt.tight_layout()
        print("Số ticker dùng PCA:", Xz.shape[1], "| số ngày:", len(Xz))
        return fig


# --- End-to-end CDO engine ----------------------------------------------------


class SyntheticCDOEngine:
    """Select pool from panel, build PD vector, run copula MC, tranche metrics."""

    def __init__(
        self,
        panel: CDSPanel,
        params: Optional[SyntheticCDOParams] = None,
        **kwargs: Any,
    ):
        """
        `params`: cấu hình đầy đủ; hoặc bỏ `params` và truyền keyword giống `SyntheticCDOParams`
        (vd. `rho=0.25`, `mounfield_plot=MounfieldPlotParams(rho_demo=0.35)`).
        """
        if params is not None and kwargs:
            raise TypeError("Chỉ dùng `params` hoặc keyword, không trộn cả hai.")
        self.panel = panel
        self.params = params if params is not None else SyntheticCDOParams(**kwargs)
        self._rng: Optional[np.random.Generator] = None
        self._pool: Optional[pd.DataFrame] = None
        self._val_date: Optional[str] = None
        self._p_vec: Optional[np.ndarray] = None
        self._L_paths: Optional[np.ndarray] = None

    def select_pool(self) -> SyntheticCDOEngine:
        p = self.params
        df = self.panel.frame
        px_col = p.px_column
        sub = df[df["Date"] == p.val_date].dropna(subset=[px_col])
        if len(sub) < p.pool_size:
            counts = df.groupby("Date")[px_col].apply(lambda s: s.notna().sum())
            good = counts[counts >= p.pool_size].index.sort_values()
            if len(good) == 0:
                raise ValueError("Không đủ dữ liệu PX cho pool.")
            p.val_date = str(good[len(good) // 2].date())
            sub = df[df["Date"] == p.val_date].dropna(subset=[px_col])
        self._val_date = p.val_date
        sub = sub[sub[px_col] > 0]
        self._pool = sub.sample(n=min(p.pool_size, len(sub)), random_state=p.rng_seed)
        np.random.seed(p.rng_seed)
        self._rng = np.random.default_rng(p.rng_seed)
        return self

    @property
    def pool(self) -> pd.DataFrame:
        if self._pool is None:
            raise RuntimeError("Gọi select_pool() trước.")
        return self._pool

    @property
    def rng(self) -> np.random.Generator:
        if self._rng is None:
            raise RuntimeError("Gọi select_pool() trước.")
        return self._rng

    def build_p_vector(self) -> np.ndarray:
        tenors = CreditCurve.TENORS
        T = self.params.pricing_tenor_years
        r = self.params.recovery
        rows = []
        for _, row in self.pool.iterrows():
            px = row[[f"PX{i}" for i in range(1, 11)]].values.astype(float)
            rows.append(CreditCurve.default_prob_piecewise(px, tenors, T, r))
        return np.array(rows)

    def run_monte_carlo(self) -> SyntheticCDOEngine:
        self._p_vec = self.build_p_vector()
        self._L_paths = GaussianCopulaCDO.simulate_portfolio_loss_fraction(
            self._p_vec,
            self.params.rho,
            self.params.recovery,
            self.params.n_mc,
            self.rng,
        )
        return self

    @property
    def loss_paths(self) -> np.ndarray:
        if self._L_paths is None:
            raise RuntimeError("Gọi run_monte_carlo() trước.")
        return self._L_paths

    @property
    def p_vec(self) -> np.ndarray:
        if self._p_vec is None:
            raise RuntimeError("Gọi run_monte_carlo() trước.")
        return self._p_vec

    def tranche_table(self) -> pd.DataFrame:
        L = self.loss_paths
        p = self.params
        rows = []
        for label, a, d in p.tranches:
            tl = GaussianCopulaCDO.tranche_loss_fraction(L, a, d)
            el_t = float(tl.mean())
            width = d - a
            spr = 10_000 * el_t / max(1e-12, width) / p.pricing_tenor_years
            rows.append(
                {"tranche": label, "attach": a, "detach": d, "EL": el_t, "spread_approx_bp": spr}
            )
        return pd.DataFrame(rows)

    def print_pool_summary(self) -> None:
        px = self.params.px_column
        print(f"Ngày định giá: {self._val_date} | Pool: {len(self.pool)} tên | Cột: {px}")
        print("Ví dụ tickers:", self.pool["Ticker"].tolist()[:12], "...")

    def plot_cdo_dashboard(self) -> Tuple[plt.Figure, pd.DataFrame]:
        p = self.params
        dp = p.dashboard_plot
        L_paths = self.loss_paths
        p_vec = self.p_vec
        names = self.pool["Ticker"].tolist()
        px_col = p.px_column
        spreads_bp = self.pool[px_col].values.astype(float)
        rng = self.rng

        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        ax = axes[0, 0]
        ax.hist(
            L_paths,
            bins=dp.hist_bins,
            density=True,
            color="steelblue",
            alpha=0.85,
            edgecolor="white",
        )
        ax.set_title("Phân phối tổn thất danh mục (fraction notional)")
        ax.set_xlabel("Portfolio loss fraction")
        ax.set_ylabel("Mật độ")

        ax = axes[0, 1]
        rhos = np.linspace(dp.rho_min, dp.rho_max, dp.rho_n)
        el_eq = []
        for r in rhos:
            L2 = GaussianCopulaCDO.simulate_portfolio_loss_fraction(
                p_vec, r, p.recovery, dp.mc_paths, rng
            )
            el_eq.append(
                GaussianCopulaCDO.tranche_loss_fraction(
                    L2, dp.equity_attach, dp.equity_detach
                ).mean()
            )
        ax.plot(rhos, el_eq, "o-", color="darkred")
        ax.set_title(
            f"Equity tranche [{dp.equity_attach:.0%}, {dp.equity_detach:.0%}]: EL vs ρ"
        )
        ax.set_xlabel("Asset correlation ρ")
        ax.set_ylabel("Expected tranche loss")

        ax = axes[1, 0]
        x = np.arange(len(names))
        ax.bar(x, spreads_bp, color="gray", alpha=0.75)
        ax.set_title(f"Spread CDS ({p.px_column}) — pool")
        ax.set_xlabel("Thứ tự tên")
        ax.set_ylabel("Spread (bp)")

        ax = axes[1, 1]
        for label, a, d in p.tranches:
            tl = GaussianCopulaCDO.tranche_loss_fraction(L_paths, a, d)
            ax.hist(tl, bins=dp.tranche_hist_bins, density=True, alpha=0.5, label=label)
        ax.set_title("Phân phối tổn thất theo tranche")
        ax.legend()
        ax.set_xlabel("Tranche loss fraction")

        plt.tight_layout()
        pd_flat = CreditCurve.homogeneous_pd_from_single_spread(
            spreads_bp, p.pricing_tenor_years, p.recovery
        )
        cmp = pd.DataFrame({"piecewise_curve": p_vec, "px5_only_flat": pd_flat})
        return fig, cmp.describe()

    def mounfield_figures(self) -> plt.Figure:
        """Base-correlation style diagnostic plots (homogeneous median PD)."""
        p = self.params
        mp = p.mounfield_plot
        pd_hom = float(np.median(self.p_vec))
        n_pool = len(self.p_vec)
        rho_demo = mp.rho_demo
        print("--- Pool đồng nhất (median PD) ---")
        print(f"PD (median) = {pd_hom:.4f}")
        print(f"E[L] toàn danh mục (lý thuyết) = (1-R)*PD = {(1 - p.recovery) * pd_hom:.4%}")

        d_test = mp.d_test
        el_fin = MounfieldTranche.equity_el_ratio_finite_homogeneous(
            pd_hom, rho_demo, d_test, n_pool, p.recovery
        )
        el_inf = MounfieldTranche.equity_el_ratio_lhp(pd_hom, rho_demo, d_test, p.recovery)
        print(f"\nTranche equity [0, {d_test:.0%}]: EL/d — finite n={n_pool}: {el_fin:.4f} | LHP: {el_inf:.4f}")
        rho_rec = MounfieldTranche.implied_rho_equity_tranche(
            pd_hom, d_test, n_pool, p.recovery, el_fin
        )
        print(f"Khôi phục ρ từ target EL: {rho_rec:.4f} (mục tiêu ρ = {rho_demo})")

        rho_true_curve = mp.rho_curve_target
        Ks = np.linspace(mp.K_min, mp.K_max, mp.K_n)
        rho_implied = []
        targets = []
        for K in Ks:
            tgt = MounfieldTranche.equity_el_ratio_finite_homogeneous(
                pd_hom, rho_true_curve, K, n_pool, p.recovery
            )
            targets.append(tgt)
            rho_implied.append(
                MounfieldTranche.implied_rho_equity_tranche(pd_hom, K, n_pool, p.recovery, tgt)
            )

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        ax = axes[0]
        ax.plot(Ks, rho_implied, "o-", color="darkgreen", label="ρ_base(K) khôi phục")
        ax.axhline(rho_true_curve, color="gray", ls="--", label=f"ρ tạo target = {rho_true_curve}")
        ax.set_xlabel("Detachment K")
        ax.set_ylabel("Implied ρ")
        ax.set_title("Minh họa: khớp ngược ρ (gần phẳng)")
        ax.legend(fontsize=8)

        ax = axes[1]
        ax.plot(Ks, np.array(targets) * 100, "s-", color="steelblue")
        ax.set_xlabel("K")
        ax.set_ylabel("EL equity [0,K] / notional tranche (%)")
        ax.set_title(f"Target EL (%) theo K (ρ = {rho_true_curve:.2f})")
        plt.tight_layout()
        return fig


def print_eda_text_summary(panel: CDSPanel) -> None:
    df = panel.frame
    px_cols = list(CDSPanel.PX_COLUMNS)
    print("=== Cấu trúc ===")
    print(df.shape)
    print(df.dtypes)
    print("\n=== Phạm vi ngày ===")
    print(df["Date"].min(), "→", df["Date"].max())
    print("Số ticker khác nhau:", df["Ticker"].nunique())
    print("Số quan sát / ngày (median):", df.groupby("Date").size().median())
    na_pct = df[px_cols].isna().mean().sort_values(ascending=False)
    print("\n=== Tỷ lệ thiếu theo cột PX (top) ===")
    print((na_pct * 100).round(2).head(6).to_string())

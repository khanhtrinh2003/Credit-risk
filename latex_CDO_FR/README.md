# CDO Synthétique — Rapport LaTeX (français)

Documentation et code source LaTeX du rapport de pricing et de gestion des risques pour un CDO synthétique, compilé à partir du notebook `CDO_Pipeline_Pro.ipynb`.

## Structure

```
latex_CDO_FR/
├── main.tex          # Source LaTeX principal (français)
├── main.pdf          # PDF compilé (1,1 Mo, 18 pages)
├── README.md         # Ce fichier
└── images/           # 13 figures référencées dans main.tex
    ├── fig01_capital_structure.png       — § I.2  Structure de capital
    ├── fig02_bootstrap_repricing.png     — § II.3 Validation bootstrap ISDA
    ├── fig03_hazard_termstruct.png       — § II.3 Structure par terme du hasard
    ├── fig04_loss_distrib_corr.png       — § III  Distribution des pertes vs ρ
    ├── fig05_etl_paths_spreads.png       — § IV.1 Trajectoires ETL & spreads
    ├── fig06_antithetic_var.png          — § IV.2 Réduction de variance
    ├── fig07_lhp_vs_mc.png               — § IV.3 LHP vs Monte Carlo
    ├── fig08_bc_curve.png                — § V.1  Courbe de corrélation de base (FIXÉE)
    ├── fig09_bespoke_surface.png         — § V.2  Surface de spread bespoke (FIXÉE)
    ├── fig10_greeks.png                  — § VI.1 Greeks par tranche
    ├── fig11_hedge_concentration.png     — § VI.2 Concentration des CS01
    ├── fig12_stress_heatmap.png          — § VI.3 Carte de chaleur stress
    └── fig13_historical_etl.png          — § VII  ETL historiques 2015-2021
```

## Compilation

Le document est compilé avec **pdfLaTeX** :

```bash
cd latex_CDO_FR
pdflatex main.tex
pdflatex main.tex   # 2e passe pour la table des matières et les références
```

Pour une compilation propre (sans fichiers auxiliaires temporaires) :

```bash
latexmk -pdf main.tex
latexmk -c          # nettoyer les fichiers .aux, .log, .out, .toc
```

## Dépendances

Le préambule prévoit des **fallbacks gracieux** si certains paquets ne sont pas installés :

| Paquet | Rôle | Fallback |
|--------|------|----------|
| `babel` (french) | Typographie française | Bascule vers anglais |
| `siunitx` | Unités scientifiques | Macros minimales `\SI`, `\si` |
| `tcolorbox` | Encadrés colorés | Requis (pas de fallback) |
| `hyperref` | Liens internes/externes | Requis (pas de fallback) |
| `microtype`, `booktabs`, `caption`, `subcaption`, `titlesec` | Mise en page | Tous requis |

Pour une installation complète sur Debian/Ubuntu :

```bash
sudo apt-get install texlive-latex-extra texlive-fonts-recommended \
                     texlive-lang-french texlive-science
```

Sur macOS (MacTeX) ou Windows (MiKTeX), tous les paquets sont disponibles par défaut.

## Plan du document

1. **§ I — Cadre théorique et données de marché** — Term-sheet, structure de capital, données CDS, taux et récupération.
2. **§ II — Courbes de crédit marginales** — Hasard plat de référence, bootstrap ISDA par morceaux constants, validation.
3. **§ III — Modèle de dépendance** — Copule gaussienne à un facteur, limite LHP de Vasicek (1987).
4. **§ IV — Pricing par Monte Carlo** — Simulateur de temps de défaut, ETL trimestriel, spread équitable, antithétique, benchmark LHP.
5. **§ V — Calibration de la corrélation de base** — Calibrateur BC corrigé, surface de pricing pour tranches bespoke.
6. **§ VI — Analyse des risques** — Greeks de tranche (CS01, Rho01, DV01, JtD), concentration de la couverture, matrice de stress.
7. **§ VII — Analyse de portefeuille** — Série temporelle ETL 2015-2021, levier COVID-19.
8. **§ Synthèse** — Rapport de risque final, liste de validation, recommandations production.

## Notes sur le correctif 04/2026

Le notebook source (`CDO_Pipeline_Pro.ipynb`) contenait deux bugs combinés dans la fonction `lhp_etl` :

1. **Mauvaise convention de Gauss-Hermite** — utilisation de `nodes*sqrt(2), wts/sqrt(pi)` (poids physiciste $e^{-x^2}$) avec `hermegauss` du module `hermite_e` (poids probabiliste $e^{-x^2/2}$), gonflant la variance du facteur latent d'un facteur $\sqrt{2}$.
2. **Formule interne erronée** — application de `lhp_cond_cdf(ℓ, p_c(m), ρ)` à l'intérieur de la boucle Hermite, double-comptant le facteur commun $M$.

La correction, appliquée le 26/04/2026, restaure la formule de Vasicek standard $L|m = \mathrm{LGD} \cdot p_c(m)$ avec quadrature correcte. La calibration BC restitue désormais exactement les cotations marché à $10^{-7}$ près. Les figures `fig08_bc_curve.png` et `fig09_bespoke_surface.png` reflètent les résultats corrigés.

## Auteur et références

- **Auteur :** TOM (Nguyen Duc Khanh Trinh) — M2 Sciences Actuarielles, ISFA Lyon, 2025-2026.
- **Notebook source :** `CDO_Pipeline_Pro.ipynb` (snapshot données 10/09/2021).
- **Bibliographie :** Mounfield (2009), Hull (2018), JP Morgan (2004), Vasicek (1987), Li (2000), O'Kane (2008), ISDA (2003).

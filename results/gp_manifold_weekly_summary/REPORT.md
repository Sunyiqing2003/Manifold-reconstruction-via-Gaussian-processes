# Weekly research progress: GP-derived uncertainty for noisy manifold reconstruction

## Research objective and current position

We observe

\[
Y_i=X_i+\varepsilon_i,\qquad X_i\in M\subset\mathbb R^D,
\qquad \varepsilon_i\sim N(0,\sigma^2 I_D),
\]

and seek a frequentist confidence tube for the true latent manifold \(M\). Last
week's work began from the GP contraction in the teacher's
`Manifold_fitting_notes`. We implemented that construction faithfully and tested
its point estimation and uncertainty separately. The experiments do not show a
systematic reconstruction improvement from replacing cylinder averaging by GP
prediction at the cylinder axis. They do show that GP-derived uncertainty is
more promising than the narrow posterior uncertainty produced by the current
MrGap benchmark, although the GP posterior bands remain geometry dependent.

The evidence points to a target mismatch caused by Gaussian errors in variables
(EIV). A GP trained on noisy local coordinates estimates a noisy-coordinate
regression target, which need not equal the latent-manifold displacement. The
main research direction is therefore still the notes-faithful GP contraction,
with a future frequentist radius that combines GP stochastic uncertainty and a
principled allowance for EIV and geometric target bias. Bootstrap calculations
and exact-MF analytic calculations were used only to diagnose error sources.

The project now contains two GP routes:

- **Route A, the primary direction:** the notes-faithful GP replaces the Yao
  cylinder average and supplies both the contracted point and its uncertainty.
- **Route B, a later empirical alternative:** the final point-estimator center
  is kept at an MF pilot, while a local residual GP supplies an uncertain normal
  displacement used in the tube radius.

The MF analytic/bootstrap experiment is not a proposed final method. A local
EIV likelihood is also only a possible tool for understanding or estimating the
geometric target bias; it has not been implemented here.

## Route A: the notes-faithful GP contraction

### Local construction

For a query point \(z\) near \(M\), let \(z^*\in M\) denote its target latent
manifold point. Let \(F(z)\) be the empirical local ball mean used by Manifold
Fitting. Define the rank-one contraction direction and projector

\[
u_z=\frac{F(z)-z}{\|F(z)-z\|},\qquad U_z=u_z u_z^\top.
\]

The Yao cylinder \(V_z\) is aligned with \(u_z\). For every observation
\(Y_i\in V_z\), define

\[
s_i=u_z^\top(Y_i-z),\qquad
q_i=(I_D-U_z)(Y_i-z).
\]

Then

\[
Y_i-z=q_i+u_zs_i,\qquad u_z^\top q_i=0,
\]

and the local GP training sample is

\[
\mathcal D_z=\{(q_i,s_i):Y_i\in V_z\}.
\]

Route A models the axial coordinate as a function of the projected ambient
coordinate:

\[
s_i=f_z(q_i)+e_i,\qquad f_z\sim\operatorname{GP}(m_z,k_z).
\]

A schematic projected-distance kernel is

\[
k_z(Y_i,Y_j)
=A\exp\left\{-\frac{\|(I_D-U_z)(Y_i-Y_j)\|^2}{\rho}\right\}.
\]

Thus the GP input is \(q_i\), the GP output is \(s_i\), and prediction is made
at the cylinder axis \(q=0\):

\[
f_z(0)\mid\mathcal D_z\sim N\{\mu_z(0),v_z(0)\}.
\]

The Route A contracted point is

\[
G_{\rm GP}(z)=z+u_z\mu_z(0),
\]

with conditional posterior covariance \(v_z(0)u_zu_z^\top\). For a
\(d\)-dimensional query manifold \(\Gamma\), its image estimator is

\[
\widehat M_{\rm GP}=\{G_{\rm GP}(z):z\in\Gamma\}.
\]

This is the teacher-notes construction. It is not “run full MF, then fit a GP
to residuals”; that later construction is Route B.

### Why predict at \(q=0\)?

The original cylinder contraction is approximately based on
\(\mathbb E[s\mid Y\in V_z]\). Route A instead targets
\(\mathbb E[s\mid q=0]\). For the local quadratic heuristic

\[
s=a+cq^2,
\]

cylinder averaging has the additional transverse-window term

\[
\mathbb E[s]-s(0)=c\,\mathbb E[q^2].
\]

Modeling \(q\mapsto s\) and evaluating at \(q=0\) may remove this particular
averaging contribution. This motivation does not imply that the GP must improve
the fitted manifold: it can add sampling variance, regularization error, and
hyperparameter sensitivity. We therefore tested the hypothesis rather than
assuming a point-estimation gain.

## Route A experiment 1: GP prediction versus the shared-cylinder average

The notes-faithful diagnostic used \(n=3000\), \(\sigma=0.06\), and 20 Monte
Carlo replicates for each curve. The independent first split supplied only the
closed query scaffold. Within each replicate, the average and GP used the same
estimated \(u_z\), the same cylinder \(V_z\), and the same observations. Their
only difference was cylinder averaging versus GP prediction at \(q=0\).

| Geometry | Mean \(H_{\rm avg}\) | Mean \(H_{\rm GP}\) | Fraction GP better |
|---|---:|---:|---:|
| Circle | 0.02652 | 0.03032 | 0.30 |
| Ellipse | 0.03027 | 0.03172 | 0.35 |

The ellipse's top-curvature quartile had a mean local GP improvement of about
0.00083. This is weak local evidence that evaluation at \(q=0\) may reduce a
transverse curvature-averaging contribution. Under the current narrow Yao
cylinders, however, the improvement was too small to improve whole-manifold
Hausdorff error. Point-estimation improvement should not be the main claim for
Route A.

The same experiment produced the following finite-grid conditional UQ
diagnostics:

| Geometry | GP posterior coverage | Frequentist same-GP-mean coverage | Mean \(s_{\rm post}/s_F\) |
|---|---:|---:|---:|
| Circle | 0.95 | 0.95 | 1.113 |
| Ellipse | 0.95 | 0.85 | 1.119 |

Here \(s_F\) is the frequentist sampling scale of the same GP posterior mean.
The ratios are approximately 1.11–1.12. These are conditional finite-grid
diagnostics, not a proved confidence statement for the true manifold.

## Route A experiment 2: paper-facing comparison with MF and MrGap

The main setting used \(n=3000\), \(\sigma=0.06\), and 100 Monte Carlo
replicates for circle and ellipse. All methods received shared noisy replicates.

### Point estimation

| Geometry | Manifold Fitting | MrGap | Route A GP |
|---|---:|---:|---:|
| Circle | 0.01463 | 0.01818 | 0.03093 |
| Ellipse | 0.04729 | 0.02001 | 0.03379 |

These entries are mean Hausdorff errors. No method dominates uniformly. MF is
best on the circle, while MrGap is best on this ellipse implementation and
Route A lies between MF and MrGap there. In particular, Route A is not a
uniformly improved manifold point estimator.

### Finite-grid 95% uncertainty diagnostics

| Geometry | Method and scale | Coverage | Mean half-width / \(\sigma\) |
|---|---|---:|---:|
| Circle | MrGap posterior | 0.62 | 0.319 |
| Circle | Route A frequentist GP mean | 0.86 | 0.619 |
| Circle | Route A GP posterior | 0.95 | 0.690 |
| Ellipse | MrGap posterior | 0.46 | 0.336 |
| Ellipse | Route A frequentist GP mean | 0.64 | 0.644 |
| Ellipse | Route A GP posterior | 0.78 | 0.722 |

The MrGap posterior bands are considerably narrower but strongly undercover
the true latent manifold in these experiments. Route A posterior uncertainty
is better calibrated empirically, but the ellipse coverage of 0.78 shows that
GP posterior variance alone is insufficient for latent-manifold UQ. The result
is geometry dependent.

This MrGap comparison has a concrete limitation. The public implementation
does not expose its empirical-Bayes optimizer. The planar curves therefore use
the documented frozen first-round Cassini parameter tuple. The numbers should
not be read as a fully retuned comparison with an oracle or recovered MrGap
optimizer.

## Central theoretical issue: EIV target mismatch

The GP is trained on noisy local coordinates. Let \(U\) be a latent transverse
coordinate and let \(Q\) be its observed counterpart:

\[
Q=U+\varepsilon_T.
\]

A schematic local curved-manifold model is

\[
S=\delta_z+\beta_z U+\frac12\kappa_zU^2+\varepsilon_N,
\]

where \(\delta_z\) is the desired latent displacement along the contraction
direction. Even with unlimited data, regression on the noisy coordinate targets

\[
m_{\sigma,z}(q)=\mathbb E[S\mid Q=q],
\]

so \(m_{\sigma,z}(0)\) need not equal \(\delta_z\). A useful population-to-sample
chain is

\[
\delta_z\longrightarrow m_{\sigma,z}(0)
\longrightarrow f_{\lambda,z}(0)
\longrightarrow \widehat f_z(0),
\]

where \(f_{\lambda,z}\) denotes the regularized or kernel-approximation target.
The total local error decomposes as

\[
\widehat f_z(0)-\delta_z
=\underbrace{\widehat f_z(0)-f_{\lambda,z}(0)}_{\text{I. GP stochastic estimation}}
+\underbrace{f_{\lambda,z}(0)-m_{\sigma,z}(0)}_{\text{II. regularization/approximation}}
+\underbrace{m_{\sigma,z}(0)-\delta_z}_{\text{III. EIV/geometric target bias}}.
\]

This decomposition is the main theoretical roadmap. Computing a GP posterior
variance addresses only part of term I. It does not automatically control terms
II and III, nor the geometry of the image set.

For a local circle or quadratic approximation, term III is heuristically of
order

\[
\sigma^2\kappa_z.
\]

For a circle of radius \(R\), \(\sigma^2/(2R)\) is a natural special-case
reference. The factor \(1/2\) is not asserted to be universal. At present,
\(O(\sigma^2\kappa)\) is a locally motivated and empirically supported scale,
not a general theorem.

## GP posterior variance and frequentist GP-mean variance

The two variance quantities used in the experiments answer different
questions. If the posterior mean at \(q=0\) is linear in the local axial
observations,

\[
\widehat f_z(0)=a_z^\top s,
\]

then under a simplified homoskedastic conditional noise model its frequentist
sampling variance is

\[
\operatorname{Var}_F\{\widehat f_z(0)\}
=\sigma^2\|a_z\|^2.
\]

The GP posterior variance is \(v_{{\rm post},z}(0)\). In general,

\[
v_{{\rm post},z}(0)\ne\sigma^2\|a_z\|^2.
\]

The first is repeated-sampling variability of the fitted GP mean under a noise
model; the second is posterior uncertainty conditional on the chosen GP model
and design. A frequentist theorem must calibrate their relationship or derive a
suitable stochastic radius directly. A posterior credible band should not be
called a frequentist confidence band without that calibration.

## Intended Route A confidence tube

The desired result has not been proved. Conceptually, a Route A radius should
contain four components:

\[
r_A(z)=q_\alpha s_{\rm GP}(z)
+B_{\rm reg}(z)+B_{\rm EIV}(z)+B_{\rm geom}(z),
\]

representing GP stochastic uncertainty, regularization or approximation error,
EIV/geometric target bias, and chart/image-set geometric error. The target tube
would have the form

\[
\mathcal C_{1-\alpha}
=\bigcup_{z\in\Gamma}
\left\{G_{\rm GP}(z)+t u_z:|t|\le r_A(z)\right\}.
\]

This is the intended GP-derived band estimator. Bootstrap is not part of this
definition.

## Auxiliary error-source diagnostic — not the proposed method

The exact-MF analytic/bootstrap experiment was designed to ask whether
unmodeled local averaging variability, direction estimation, or persistent
population geometry could explain the observed UQ behavior. It used \(n=3000\),
\(\sigma=0.06\), and 100 Monte Carlo replicates per geometry.

The sampling-only exact-MF diagnostic attained coverage 0.95 for both circle and
ellipse, with mean half-widths \(0.481\sigma\) and \(0.512\sigma\), respectively.
The analytic direction delta term contains

\[
\frac{1}{\|\widehat F(z)-z\|},
\]

and became unstable at small direction signals. Its maximum additive width
reached about \(13.1\sigma\) for circle and \(10.1\sigma\) for ellipse. In the
oracle-direction exact-MF ablation, ellipse geometric error changed only from
0.02565 to 0.02493, while coverage remained approximately 0.95. Direction
uncertainty is therefore not generally the dominant missing term for this exact
MF estimator.

The strongest result from this auxiliary experiment concerns persistent
geometry. For the circle, the mean absolute residual population displacement
was 0.00292, between the reference scales

\[
\frac{\sigma^2}{2R}=0.00180,
\qquad
\frac{\sigma^2}{R}=0.00360.
\]

For the ellipse, the correlation between residual-bias magnitude and curvature
was 0.953. These observations are empirically consistent with an
\(O(\sigma^2\kappa)\) population term. They do not establish its constant or a
general expansion.

The full-algorithm bootstrap reran the ball step, direction estimation,
cylinder selection, contraction, and final smoothing. Its role was to benchmark
algorithmic stochastic variability and expose missing error sources. The
proposed project is not an MF-plus-bootstrap confidence band.

## Route B: MF-centered GP-derived residual uncertainty

Route B was developed after the notes-faithful experiments and is not the
teacher's original proposal. It first obtains an MF pilot or fitted manifold.
At a point \(z\) on that fit, a local GP models residual normal displacement.
Writing its residual posterior mean and SD as
\(\widehat m_{\rm GP}(z)\) and \(s_{\rm GP}(z)\), the diagnostic radius was

\[
r_B(z)=|\widehat m_{\rm GP}(z)|+q_\alpha s_{\rm GP}(z).
\]

The reported point-estimator center remains the MF center. The residual GP mean
is treated as an uncertain latent displacement included in the radius rather
than necessarily being forced into the point estimate. Route B is GP-derived;
it is not a bootstrap construction.

The frozen Route B experiment used \(n=3000\), \(\sigma=0.06\),
\(h=1.5\sigma\), and 20 Monte Carlo replicates.

| Geometry | MF Hausdorff | GP-refined Hausdorff | Fraction MF better | MF-centered GP coverage | Strictly inside \(1.96\sigma\) band | Mean max width / noise half-width |
|---|---:|---:|---:|---:|---:|---:|
| Circle | 0.0165 | 0.0257 | 1.00 | 1.00 | 1.00 | 0.572 |
| Ellipse | 0.0498 | 0.0269 | 0.00 | 1.00 | 0.00 | 0.775 |

On the circle, inserting the GP correction into the center degraded an already
good MF fit. On the ellipse, the residual GP correction improved a
geometrically biased MF center. The 20/20 MF-centered tube coverage is
encouraging but remains a diagnostic: validity of
\(|\widehat m_{\rm GP}|+q s_{\rm GP}\) as a latent-manifold confidence radius
has not been proved. On the ellipse, a smaller width than the raw noise width
did not imply strict geometric containment because the MF center itself was
displaced.

The numerical MF results across this report should not be treated as one common
benchmark. The notes-faithful comparison uses a split scaffold and matched
cylinders; the paper-facing experiment converts method outputs to a common
query/image representation; Route B constructs a data-driven MF pilot on a
60-point grid; and the auxiliary diagnostic follows every point in the exact
final smoothed MF cloud. These settings answer different questions.

## Route A and Route B

| Feature | Route A: notes-faithful primary route | Route B: later alternative |
|---|---|---|
| Origin | Teacher's `Manifold_fitting_notes` | Subsequent empirical construction |
| Center | \(G_{\rm GP}(z)=z+u_z\mu_z(0)\) | Original MF center |
| GP role | Replaces cylinder average; supplies point estimate and uncertainty | Models residual latent displacement and uncertainty |
| Geometry | Yao ball direction and cylinder | Local normal charts around MF fit |
| Empirical status | No systematic fitting gain; promising but geometry-dependent UQ | Encouraging diagnostic coverage; not theoretically calibrated |
| Main motivation | Direct notes-faithful GP-derived manifold and band | Preserve strong MF point estimation when GP correction is unstable |
| Main theory problem | GP stochastic error plus EIV target bias and image geometry | Validity of a residual-aware radius around a random MF center |

Route A remains the primary research direction. Route B is retained as an
experimentally motivated alternative if preserving the MF point estimator
becomes important.

## Possible next step: local EIV likelihood

One possible way to study or estimate the curvature-dependent target mismatch is
a local latent-coordinate model

\[
Q_i=U_i+\varepsilon_{Ti},
\qquad
S_i=\delta_z+\beta_zU_i+\frac12\kappa_zU_i^2+\varepsilon_{Ni}.
\]

A local EIV likelihood could estimate \(\delta_z\), \(\beta_z\), and
\(\kappa_z\). Its potential role is to quantify or estimate the EIV curvature
bias, rather than to replace the GP-derived stochastic band. A future method
might combine GP stochastic uncertainty with an analytic or likelihood-based
EIV bias calibration. No such local EIV-MLE has yet been implemented or
validated in the reported experiments.

## Desired-theory roadmap

The following items distinguish the desired theorem from current heuristics and
empirical evidence.

1. **T1 — Population target.** Characterize
   
   \[
   m_{\sigma,z}(0)=\mathbb E[S\mid Q=0,\text{ local selection}]
   \]
   
   relative to the latent displacement \(\delta_z\). This is an open theory
   problem.

2. **T2 — EIV expansion or envelope.** Establish dependence on \(\sigma\),
   curvature, reach, local density, and cylinder scale. The currently observed
   \(\sigma^2\kappa\) behavior is a heuristic supported by simulations; the
   desired result is a rigorous expansion or bound in an explicit regime.

3. **T3 — GP stochastic theory.** Obtain uniform frequentist control of
   \(\widehat f_z(0)-f_{\lambda,z}(0)\) over the query manifold.

4. **T4 — Variance calibration.** Relate GP posterior variance to the
   frequentist variance of the GP posterior mean, or construct a direct
   frequentist GP stochastic radius.

5. **T5 — Query and image geometry.** Give conditions under which
   
   \[
   \widehat M_{\rm GP}=\{G_{\rm GP}(z):z\in\Gamma\}
   \]
   
   remains a \(d\)-dimensional embedded manifold. A rank-one contraction of a
   full-dimensional query domain does not automatically produce a
   \(d\)-manifold.

6. **T6 — Global simultaneous band.** Combine GP stochastic uncertainty, GP
   approximation error, EIV/geometric bias, and image-manifold stability into a
   frequentist confidence tube for the true latent manifold. This is the desired
   theorem, not a result already established by the experiments.

## What we learned last week

1. We implemented the GP contraction proposed in the teacher's notes with GP
   input \(q_i\), output \(s_i\), and evaluation at \(q=0\).
2. In the current narrow-cylinder regime, GP prediction at \(q=0\) did not
   systematically improve reconstruction over the shared-cylinder average.
3. Point-estimation improvement should therefore not be the main selling point
   of Route A.
4. Route A GP posterior uncertainty was empirically much better calibrated than
   the narrow MrGap posterior bands, but it still undercovered on the ellipse in
   the paper-facing benchmark.
5. The strongest auxiliary finding was the association between persistent
   systematic error and curvature, empirically consistent with an
   \(O(\sigma^2\kappa)\) EIV/geometric scale.
6. Bootstrap was useful for diagnosing full-algorithm stochastic variability;
   it is not the proposed final confidence-band method.
7. The main target remains a GP-derived frequentist tube that combines GP
   stochastic uncertainty with principled EIV/geometric bias calibration.
8. Route B remains a GP-derived alternative if preserving the MF point
   estimator becomes desirable.

## Source reports

All numerical values above were checked directly against the frozen reports:

- `results/notes_gp_contraction_demo/REPORT.md`;
- `results/benchmark_mf_gp_uq_vs_mrgap/REPORT.md`;
- `results/manifold_fitting_frequentist_uq/REPORT.md`;
- `results/manifold_fitting_gp_mfcenter_demo/REPORT.md`.

No experiment or result file was changed for this weekly summary.

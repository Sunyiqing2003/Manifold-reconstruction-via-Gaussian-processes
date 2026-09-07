# Weekly research progress: GP contraction for noisy manifold reconstruction

## 1. Objective and current conclusion

We observe

$$
Y_i=X_i+\varepsilon_i,\qquad
X_i\in M\subset\mathbb R^D,\qquad
\varepsilon_i\sim N(0,\sigma^2 I_D),
$$

and seek a frequentist confidence tube for the latent manifold $M$. The work
last week implemented and tested the GP contraction described in the teacher's
Manifold Fitting notes. For a query point near the manifold, the method estimates
the Manifold Fitting contraction direction, constructs the corresponding Yao
cylinder, regresses axial displacement on transverse displacement, and evaluates
the regression at the cylinder axis.

The current experiments give two distinct conclusions.

1. GP prediction at the cylinder axis does not systematically improve
   reconstruction over averaging the same cylinder observations.
2. GP-derived uncertainty is empirically more promising than the narrow MrGap
   posterior uncertainty, but its coverage is geometry dependent and is not yet
   a frequentist confidence guarantee.

The main research question is therefore not whether a GP improves point
estimation. It is whether the local GP error can be decomposed and controlled
uniformly relative to the latent manifold target. In particular, the regression
is performed in noisy, data-dependent coordinates and after local selection.
The gap between that population regression target and the desired latent point
must be characterized rather than assigned in advance to a single mechanism.

## 2. GP contraction algorithm

### 2.1 Local direction and cylinder

Let $z$ be a query point near $M$, and let $z^*\in M$ denote the latent point
that the local contraction is intended to recover. Let $\widehat F(z)$ be the
empirical mean of observations in the local ball around $z$. Define

$$
\widehat u_z
=\frac{\widehat F(z)-z}
       {\|\widehat F(z)-z\|},
\qquad
\widehat U_z=\widehat u_z\widehat u_z^\top.
$$

The Yao cylinder $V_z$ is centered at $z$ and aligned with $\widehat u_z$.
For every $Y_i\in V_z$, define

$$
s_i=\widehat u_z^\top(Y_i-z),
\qquad
q_i=(I_D-\widehat U_z)(Y_i-z).
$$

Then

$$
Y_i-z=q_i+\widehat u_zs_i,
\qquad
\widehat u_z^\top q_i=0.
$$

The local training data are

$$
\mathcal D_z=\{(q_i,s_i):Y_i\in V_z\}.
$$

The GP input is the projected ambient vector $q_i\in\widehat u_z^\perp$.
The GP output is the scalar axial coordinate $s_i$. The method does not replace
$\widehat u_z$ by a complete tangent or normal basis.

### 2.2 Local GP and contraction point

The working local model is

$$
s_i=f_z(q_i)+e_i,\qquad
f_z\sim\operatorname{GP}(m_z,k_z).
$$

The implemented squared-exponential covariance is

$$
k_z(q_i,q_j)
=A\exp\left(
  -\frac{\|q_i-q_j\|^2}{2\ell^2}
\right).
$$

With covariance matrix

$$
K_z(i,j)=k_z(q_i,q_j)+\sigma^2\mathbf 1\{i=j\},
$$

the GP is evaluated at $q=0$. The unknown constant mean is handled by universal
kriging. If $\mathbf 1$ is the vector of ones and

$$
k_0(i)=k_z(q_i,0),
$$

the implemented prediction weights are

$$
a_z
=K_z^{-1}k_0
+\frac{K_z^{-1}\mathbf 1}
       {\mathbf 1^\top K_z^{-1}\mathbf 1}
 \left(1-\mathbf 1^\top K_z^{-1}k_0\right).
$$

Thus

$$
\widehat\mu_z(0)=a_z^\top s,
\qquad
\widehat G_{\rm GP}(z)
=z+\widehat u_z\,\widehat\mu_z(0).
$$

For a one-dimensional closed query scaffold $\Gamma$, the fitted image is

$$
\widehat M_{\rm GP}
=\{\widehat G_{\rm GP}(z):z\in\Gamma\}.
$$

The experiment uses a one-dimensional query scaffold because a contraction map
applied to an arbitrary full-dimensional query domain is not automatically a
one-dimensional embedded manifold.

### 2.3 Why evaluate at $q=0$?

The matched cylinder average uses

$$
\overline s_z
=\frac{1}{|V_z|}\sum_{Y_i\in V_z}s_i,
$$

which estimates a cylinder-averaged axial displacement. The GP instead estimates
the regression value at the cylinder axis. For the local heuristic

$$
s=a+cq^2,
$$

the average has the transverse-window contribution

$$
\mathbb E[s]-s(0)
=c\,\mathbb E[q^2].
$$

This motivates estimating the function $q\mapsto s$ and evaluating at zero.
It does not imply that the GP has lower total error. GP fitting can add sampling
variation, kernel approximation error, and sensitivity to covariance
parameters. The matched-cylinder experiment was designed to test this tradeoff
directly.

## 3. Experimental design and parameter choices

This section records the actual frozen choices used by the notes-faithful
experiment and the paper-facing benchmark. No parameter below was selected by
looking at the truth in each replicate.

### 3.1 Data and query scaffold

The main circle and ellipse models are

$$
M_{\rm circle}
=\{(\cos\theta,\sin\theta):0\leq\theta<2\pi\},
$$

and

$$
M_{\rm ellipse}
=\{(1.4\cos\theta,0.8\sin\theta):0\leq\theta<2\pi\}.
$$

Circle parameters are sampled uniformly in angle. Ellipse parameters are
sampled approximately uniformly in arc length by rejection sampling with
acceptance probability proportional to curve speed. Independent isotropic
Gaussian noise with $\sigma=0.06$ is then added in the two ambient coordinates.

The notes-faithful experiment uses $n=3000$ observations and randomly splits
them into:

- 1500 observations for a data-only closed radial scaffold;
- 1500 independent observations for the ball, cylinder, and GP contraction.

The scaffold has 60 equally spaced polar-angle locations. Its radial smoother
uses Gaussian angular weights

$$
w_{ji}
=\exp\left\{
 -\frac{\Delta(\theta_i,\phi_j)^2}{2h_\phi^2}
\right\},
\qquad h_\phi=0.16\ {\rm radians},
$$

where $\Delta$ is periodic angular distance. This $h_\phi$ constructs only the
query scaffold; it is not the GP regression length scale. The query points are
offset from the scaffold by

$$
z_j=\widetilde z_j+\sigma\,\widehat n_j.
$$

The offset coefficient was frozen at one. Its purpose is to place queries in a
controlled neighborhood where the ball mean produces a nonzero contraction
signal. The scaffold normal is used only for this offset and is not passed to
the cylinder or GP, whose direction is recomputed from the independent
contraction sample.

### 3.2 Ball and cylinder scales

Let $n_c$ be the contraction-split size. The implementation follows the
bandwidth scaling in the repository's Yao Manifold Fitting port:

$$
r=\frac{5\sigma}{\log_{10}(n_c)},
\qquad
r_0=2r,
\qquad
R=\frac{10\sigma\sqrt{\log(1/\sigma)}}{\log_{10}(n_c)}.
$$

Here:

- $r_0$ is the ball radius used to estimate $\widehat F(z)$;
- $r$ is the transverse cylinder radius;
- $R$ is the axial cylinder half-length.

At $n_c=1500$ and $\sigma=0.06$, these are approximately

$$
r=0.09446,\qquad
r_0=0.18891,\qquad
R=0.31687.
$$

The multiplicative bandwidth factor was fixed at one. These formulas were
inherited from the existing Manifold Fitting implementation; they were not
optimized for the GP experiment. This choice keeps the GP-versus-average
comparison within the same local Yao geometry, but it does not establish that
the scales are theoretically optimal for GP regression.

At least five observations are requested for the ball step. If the radius ball
contains fewer than five, the five nearest observations are used. At least 11
observations are requested for the cylinder GP. If the cylinder contains fewer
than 11, the 11 closest observations under the scaled cylinder metric are used.
In the frozen main experiment, the mean median cylinder sizes were 46.8 for the
circle and 41.9 for the ellipse; the fallback fractions were zero.

### 3.3 GP regression parameters

The GP covariance parameters were frozen as

$$
A=\sigma^2=0.0036,
\qquad
\ell=r.
$$

The observation-noise variance in $K_z$ was also fixed at the known simulation
value $\sigma^2$. The mean was an unknown constant estimated separately in each
local fit through universal kriging. Numerically, the Cholesky system used a
jitter of

$$
10^{-10}\max(1,A,\sigma^2),
$$

with a $10^{-7}$ fallback if the first factorization failed.

The scale $A=\sigma^2$ gives the latent local axial function the same variance
order as the ambient observation noise. The choice $\ell=r$ ties GP smoothness
to the transverse window over which $q$ is observed. These are transparent
scale-matching rules rather than estimated hyperparameters. The experiment did
not maximize marginal likelihood, cross-validate, or tune $A$, $\ell$, the
nugget, the query offset, or the Yao bandwidth multiplier. Consequently, the
results describe this frozen GP specification and do not establish optimality
or robustness of the covariance parameters.

There is therefore no additional GP bandwidth called $h$ in the current
notes-faithful algorithm. The quantities that can otherwise be conflated as
“bandwidth” are:

| Quantity | Value or rule | Role |
|---|---:|---|
| $h_\phi$ | 0.16 radians | radial smoothing for the query scaffold |
| $r_0$ | $2r$ | ball mean and direction |
| $r$ | $5\sigma/\log_{10}(n_c)$ | transverse cylinder radius |
| $R$ | $10\sigma\sqrt{\log(1/\sigma)}/\log_{10}(n_c)$ | axial cylinder half-length |
| $\ell$ | $r$ | GP covariance length scale in $q$ |

The value $h=1.5\sigma$ belongs to a later MF-centered residual-GP experiment.
It is not part of the main notes-faithful contraction analyzed in this report.

### 3.4 Uncertainty quantities and simultaneous multiplier

The universal-kriging posterior variance at zero is

$$
\widehat v_{{\rm post},z}(0)
=A-k_0^\top K_z^{-1}k_0
+\frac{
 \left(1-\mathbf 1^\top K_z^{-1}k_0\right)^2
}{
 \mathbf 1^\top K_z^{-1}\mathbf 1
}.
$$

The reported conditional posterior SD is

$$
s_{{\rm post},z}
=\sqrt{\widehat v_{{\rm post},z}(0)}.
$$

Because $\widehat\mu_z(0)=a_z^\top s$, the simplified conditional frequentist
variance of the same GP mean, holding the design and weights fixed, is

$$
\widehat v_{F,z}(0)
=\sigma^2\|a_z\|^2.
$$

These are different uncertainty quantities. The first is conditional GP
posterior uncertainty under the covariance model. The second is repeated-noise
variability of the fitted GP mean under a homoskedastic working model. Neither
currently includes randomness from the scaffold, ball direction, cylinder
membership, or the population-target gap.

For the 60-point query scaffold and nominal level 0.95, the finite-grid
simultaneous multiplier is the Bonferroni value

$$
q_{0.95}
=\Phi^{-1}\left(1-\frac{0.05}{2\cdot60}\right)
\approx 3.34.
$$

The displayed pointwise half-width is $q_{0.95}s_z$. Coverage is evaluated
geometrically against a dense truth grid rather than only at corresponding
query parameters. The notes-faithful experiment uses 2400 truth points; the
paper-facing benchmark uses 4800. These bands are finite-grid diagnostics and
are not presented as proved frequentist confidence tubes.

### 3.5 Paper-facing benchmark settings

The paper-facing circle and ellipse benchmark retains the same GP-contraction
construction:

- half-sample scaffold and half-sample contraction;
- 60 query points;
- $h_\phi=0.16$;
- query offset $z=\widetilde z+\sigma\widehat n$;
- the same Yao formulas with multiplier one;
- minimum ball and cylinder sizes 5 and 11;
- $A=\sigma^2$, $\ell=r$, nugget $\sigma^2$;
- constant-mean universal kriging.

Its primary setting has $n=3000$, $\sigma=0.06$, and 100 Monte Carlo replicates.
The broader $n$ and $\sigma$ grid uses only three replicates per non-primary
setting, so those endpoints are sensitivity diagnostics rather than precise
coverage estimates.

MrGap is included only as a documented comparator. Its public release does not
contain the empirical-Bayes optimizer used in the paper. The planar benchmark
therefore freezes the published first-round Cassini tuple

$$
A=0.014,\qquad \rho=0.2,\qquad
\text{noise variance}=0.002.
$$

This limitation prevents interpreting the comparison as a fully optimized
MrGap evaluation.

### 3.6 Parameter adjustment in future experiments

The present numbers should remain the frozen baseline. A future parameter study
should separate geometric localization from GP covariance estimation rather
than adjust all constants against reconstruction error at once.

The geometric parameters can be written as

$$
r=\frac{c_r\sigma}{\log_{10}(n_c)},\qquad
r_0=c_0r,\qquad
R=\frac{c_R\sigma\sqrt{\log(1/\sigma)}}{\log_{10}(n_c)},
$$

and the query offset as

$$
z=\widetilde z+c_{\rm off}\sigma\widehat n.
$$

The current baseline is

$$
c_r=5,\qquad c_0=2,\qquad c_R=10,\qquad c_{\rm off}=1.
$$

These constants control different effects: $r_0$ controls direction signal,
$r$ controls transverse localization and local sample size, $R$ controls axial
truncation, and $c_{\rm off}$ controls the distance from the scaffold to the
target curve. They should first be studied through observable diagnostics such
as ball and cylinder counts, $\|\widehat F(z)-z\|$, fallback frequency, and
stability across data splits. Latent truth error and coverage should remain
evaluation criteria, not tuning objectives.

Conditional on a frozen cylinder construction, GP parameters may be expressed
as

$$
A=c_A\sigma^2,\qquad
\ell=c_\ell r,\qquad
\tau^2=c_\tau\sigma^2,
$$

where $\tau^2$ is the GP working noise variance. The current baseline has

$$
c_A=c_\ell=c_\tau=1.
$$

Possible data-driven choices include pooled local marginal likelihood across
query points, held-out conditional predictive likelihood within cylinders, or
a predeclared empirical-Bayes rule. Any such procedure must be fitted using
observed data only, frozen before coverage evaluation, and repeated inside the
sampling analysis because estimated covariance parameters add uncertainty.
Local unconstrained maximum likelihood may be weakly identified when cylinders
contain only 40–50 points, so a practical rule would need compact parameter
ranges or partial pooling across nearby queries.

The following order would keep interpretation clear:

1. retain the current frozen specification as the reference;
2. vary geometric constants while keeping GP ratios fixed;
3. freeze the resulting observable-data rule;
4. compare fixed GP ratios with a predeclared likelihood-based rule;
5. rerun coverage evaluation only after all selection rules are frozen.

This is a proposed adjustment protocol, not a tuning experiment already
completed. The theory must ultimately state admissible sequences for $r_0$,
$r$, $R$, $A$, $\ell$, and $\tau^2$ rather than rely only on a finite-sample
selection procedure.

## 4. Experimental results

### 4.1 Matched-cylinder mechanism test

This is the cleanest test of replacing the cylinder average by prediction at
$q=0$. Both methods use the same estimated direction, cylinder, and cylinder
observations.

| Geometry | Mean $H_{\rm avg}$ | Mean $H_{\rm GP}$ | GP better fraction |
|---|---:|---:|---:|
| Circle | 0.02652 | 0.03032 | 0.30 |
| Ellipse | 0.03027 | 0.03172 | 0.35 |

The ellipse's top-curvature quartile has a mean local GP improvement of about
0.00083. This is weak local evidence for the transverse-window motivation, but
the effect is too small to improve whole-curve Hausdorff error in the current
narrow-cylinder regime.

The corresponding finite-grid conditional uncertainty results are:

| Geometry | Posterior coverage | Frequentist same-GP-mean coverage | Mean $s_{\rm post}/s_F$ |
|---|---:|---:|---:|
| Circle | 0.95 | 0.95 | 1.113 |
| Ellipse | 0.95 | 0.85 | 1.119 |

The posterior and frequentist GP-mean SDs are numerically close but not equal.
The coverage values are based on 20 Monte Carlo replicates and should be read as
diagnostics.

### 4.2 Paper-facing point estimation

At $n=3000$ and $\sigma=0.06$, the mean Hausdorff errors over 100 replicates
are:

| Geometry | Manifold Fitting | MrGap | GP contraction |
|---|---:|---:|---:|
| Circle | 0.01463 | 0.01818 | 0.03093 |
| Ellipse | 0.04729 | 0.02001 | 0.03379 |

No method dominates both geometries. The GP contraction is not a uniformly
improved point estimator. These MF values also should not be numerically merged
with the matched-cylinder comparison: the paper-facing benchmark converts
method outputs to a common query/image representation, whereas the mechanism
test compares two contractions on a shared scaffold and shared cylinders.

### 4.3 Paper-facing 95% uncertainty

| Geometry | Method and scale | Coverage | Mean half-width / $\sigma$ |
|---|---|---:|---:|
| Circle | MrGap posterior | 0.62 | 0.319 |
| Circle | GP-mean frequentist scale | 0.86 | 0.619 |
| Circle | GP posterior scale | 0.95 | 0.690 |
| Ellipse | MrGap posterior | 0.46 | 0.336 |
| Ellipse | GP-mean frequentist scale | 0.64 | 0.644 |
| Ellipse | GP posterior scale | 0.78 | 0.722 |

MrGap's posterior bands are much narrower but strongly undercover the latent
curve in this frozen comparison. The GP posterior scale improves empirical
coverage, but the ellipse coverage remains 0.78. Thus conditional GP posterior
variance alone does not account for all errors relevant to the latent-manifold
target.

## 5. Reconsidered theory setting

### 5.1 What the GP actually targets

It is premature to state that the remaining error is caused by EIV. A more
accurate starting point is to define the population regression induced by the
entire local construction.

First consider an idealized fixed query $z$, fixed unit direction $u$, and fixed
cylinder $V(z,u)$. Define the observed local coordinates

$$
Q=(I_D-uu^\top)(Y-z),
\qquad
S=u^\top(Y-z).
$$

The population regression target of the local GP is

$$
m_{z,u,V}(q)
=\mathbb E[S\mid Q=q,\ Y\in V(z,u)].
$$

The desired contraction displacement is a latent geometric quantity,
schematically

$$
\delta_{z,u}
=u^\top(z^*-z).
$$

Even an oracle nonparametric regression estimator converges to
$m_{z,u,V}(0)$, not automatically to $\delta_{z,u}$. The population gap is

$$
b_{\rm target}(z,u,V)
=m_{z,u,V}(0)-\delta_{z,u}.
$$

This definition is neutral about its cause. Several effects can contribute:

1. measurement error in the transverse coordinate;
2. measurement error in the axial coordinate together with selection;
3. curvature and higher-order local geometry;
4. nonuniform latent sampling density;
5. truncation by the finite ball and cylinder;
6. displacement of the query point;
7. estimation error in $\widehat u_z$ and the resulting change of cylinder
   membership.

Calling the whole gap “EIV bias” would combine these mechanisms without
identification.

### 5.2 EIV as one candidate mechanism

In an ideal fixed frame, let $T$ be a latent tangent coordinate and suppose

$$
Q=T+\varepsilon_T,
$$

and

$$
S=\delta+\beta T+\frac12\kappa T^2+\varepsilon_N.
$$

Then

$$
\mathbb E[S\mid Q=0]
-\delta
=\beta\,\mathbb E[T\mid Q=0]
+\frac12\kappa\,\mathbb E[T^2\mid Q=0]
+\mathbb E[\varepsilon_N\mid Q=0].
$$

Only under additional conditions—such as a fixed correct frame, local
symmetry, weak boundary effects, a regular latent density, and independent
isotropic noise—do the first and third terms vanish and the second term have
scale approximately $\sigma^2\kappa/2$. With cylinder selection, estimated
direction, and a displaced query, the conditional moments can change.

Therefore the observed curvature association is compatible with a
noisy-coordinate or EIV mechanism, but it does not identify that mechanism.
The exact-MF diagnostic found a circle residual magnitude 0.00292, between
$\sigma^2/(2R)=0.00180$ and $\sigma^2/R=0.00360$, and an ellipse correlation
0.953 between residual magnitude and curvature. These are useful clues for the
order of a target-gap term. They are not evidence that all remaining GP error
is caused by EIV, and they do not determine a universal coefficient.

### 5.3 Error decomposition for the GP contraction

Let $m_z(0)$ denote the population selected-regression target in the actual
data-dependent local coordinate system, and let $f_{\lambda,z}(0)$ denote the
kernel-regularized target. A useful decomposition is

$$
\widehat\mu_z(0)-\delta_z
=
\underbrace{
  \widehat\mu_z(0)-f_{\lambda,z}(0)
}_{\text{GP stochastic estimation}}
+
\underbrace{
  f_{\lambda,z}(0)-m_z(0)
}_{\text{kernel approximation or regularization}}
+
\underbrace{
  m_z(0)-\delta_z
}_{\text{population target gap}}.
$$

For a global manifold statement, this local scalar decomposition is still
incomplete. It must be augmented by errors from the estimated query scaffold
and direction, dependence between overlapping local fits, and stability of the
image $z\mapsto\widehat G_{\rm GP}(z)$.

The phrase “population target gap” should be used until its components are
derived under explicit assumptions. EIV may become one term in a later
expansion rather than the label for the entire residual.

## 6. Theory outline

The theory should proceed from an idealized local target to the full
data-dependent image instead of starting from a presumed
$\sigma^2\kappa$ correction.

### T0. Define the estimand and query geometry

Specify:

- how $z^*$ is associated with $z$, for example by the unique metric projection
  within the reach tube;
- whether $\Gamma$ is deterministic, estimated from an independent split, or
  estimated from the same data;
- the dimension and regularity required of $\Gamma$;
- the exact cylinder selection event.

Without this step, “coverage of the manifold” and “local displacement” are not
unambiguous.

### T1. Fixed-frame population regression

For deterministic $(z,u,V)$, characterize

$$
m_{z,u,V}(0)
=\mathbb E[S\mid Q=0,\ Y\in V(z,u)].
$$

Derive its dependence on latent density, curvature, higher fundamental forms,
noise level, $r$, and $R$. This stage should separate measurement-error
convolution from finite-window selection.

### T2. Local expansion of the population target gap

Under explicit symmetry, density, reach, and scale assumptions, seek an
expansion or envelope for

$$
m_{z,u,V}(0)-\delta_{z,u}.
$$

A term of order $\sigma^2\kappa$ is a candidate in special regimes. Other terms
may involve density gradients, $r^2\kappa$, higher-order geometry, query
displacement, and cylinder truncation. The expansion should determine when each
term is leading instead of assuming the curvature term always dominates.

### T3. Estimated direction and random cylinder

Control

$$
\widehat u_z-u_z
$$

and the induced symmetric difference between $V(z,\widehat u_z)$ and
$V(z,u_z)$. This is not a standard smooth delta-method problem because cylinder
membership changes discretely. Sample splitting removes some dependence but
does not remove randomness of the direction or local design.

### T4. GP approximation and stochastic error

For the selected local design, control

$$
\widehat\mu_z(0)-f_{\lambda,z}(0)
$$

uniformly over $z\in\Gamma$, and separately control

$$
f_{\lambda,z}(0)-m_z(0).
$$

The result must state how $A$, $\ell$, the noise variance, local sample size,
and the cylinder scales may depend on $n$ and $\sigma$. The current choices
$A=\sigma^2$ and $\ell=r$ are experimental scale rules, not theoretical
sequences derived from an optimality result.

### T5. Variance calibration

Determine whether the GP posterior SD

$$
s_{{\rm post},z}
$$

upper-bounds, approximates, or must be rescaled relative to the frequentist
sampling distribution of $\widehat\mu_z(0)$. This requires accounting for
estimated mean, random design, covariance-parameter choice, and local
selection. The empirical ratio near 1.11 in the matched-cylinder experiment is
evidence for one frozen regime, not a general calibration result.

### T6. Simultaneous control over the query scaffold

Replace the finite-grid Bonferroni diagnostic with uniform control over
$z\in\Gamma$. This requires regularity of the local GP process, dependence
between overlapping cylinders, and discretization error between a finite query
grid and the continuous scaffold.

### T7. Stability of the image manifold

Establish conditions under which

$$
\widehat M_{\rm GP}
=\{\widehat G_{\rm GP}(z):z\in\Gamma\}
$$

is an embedded manifold of the intended dimension and is geometrically close
to $M$. Local scalar coverage along $\widehat u_z$ does not by itself imply
Hausdorff coverage of the complete image.

### T8. Candidate confidence radius

Only after T0–T7 should the radius be assembled. A schematic form is

$$
r(z)
=q_\alpha s_{\rm GP}(z)
+B_{\rm approx}(z)
+B_{\rm target}(z)
+B_{\rm frame}(z)
+B_{\rm image}(z).
$$

Here $B_{\rm target}$ is a bound for the full population target gap. It should
not be renamed an EIV term unless a derivation shows which part is due to
measurement error. The desired tube is

$$
\mathcal C_{1-\alpha}
=\bigcup_{z\in\Gamma}
\left\{
  \widehat G_{\rm GP}(z)+t\widehat u_z:
  |t|\leq r(z)
\right\}.
$$

This is a target for future theory, not a band already validated by the current
simulations.

## 7. Immediate next steps

1. Write the fixed-frame population target $m_{z,u,V}(0)$ for the circle under
   Gaussian noise and the actual finite-cylinder selection rule.
2. Compare analytically the contributions of transverse measurement error,
   finite transverse width $r$, axial truncation $R$, and query offset.
3. Determine whether the circle coefficient approaches $1/2$ under a clearly
   stated limiting regime or depends materially on selection and latent density.
4. Extend the calculation to a general local quadratic graph and identify all
   terms of the same order as $\sigma^2\kappa$.
5. Treat $A$, $\ell$, and the nugget as theoretical sequences and determine
   conditions for uniform GP-mean control; only then decide whether estimation
   by marginal likelihood is compatible with the proof.
6. Use the existing simulation outputs to check derived signs and orders, while
   keeping truth-based quantities out of the estimator and parameter selection.

The present evidence supports continuing with the teacher-notes GP contraction.
It does not yet justify attributing the observed residual to EIV alone, adding a
fixed curvature correction, or calling the conditional GP posterior band a
frequentist confidence tube.

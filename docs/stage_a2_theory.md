# Stage A2 — difference-aware certificates, certified routing, online budget allocation

This note states and proves every mathematical claim made by
`stage_ab/vinterval.py`, `stage_ab/stage_a2.py` and `stage_ab/budget.py`.
Measured evidence is in [`reports/stage_a2_validation.md`](../reports/stage_a2_validation.md).
The Stage A contracts of [`stage_ab_theory.md`](stage_ab_theory.md) remain valid; this note
extends them.

**Target of every certificate.** All error bounds are relative to the
*same-grid, exact-arithmetic Godunov map* $S$ with the binary64 time-step ratio.
None of them bounds the error with respect to the entropy solution of the PDE,
and no speedup is claimed (Section 10).

## 0. Results at a glance

| # | Statement | Type | Where |
|---|---|---|---|
| 4.1 | One-step bound for **any** neural/Godunov trust pattern, including all floating-point effects | soundness | `cell_costs`, `mixed_step`, `rigorous_step_cost` |
| 4.2 | Global bound for any adaptive routing rule; the rollout never exceeds its total budget | soundness | `certified_rollout_a2` |
| 5.3 | Difference enclosure: $D(y)-D(x)$ for the real ReLU flux minus Godunov, via interval forward-mode AD | soundness | `difference_enclosure`, `difference_bounds` |
| 5.5 | The difference certificate is gauge-invariant; face-separable certificates are not | structure | — |
| 6.1 | Every face-separable certificate is $\Theta(1)$ as $h\to 0$, while the true error is $O(h)$ | lower bound | `mode="separable"` |
| 6.2 | The table difference certificate is $O(h\,\mathrm{TV})$ | upper bound | `mode="table"` |
| 6.3 | The local-box certificate exceeds the exact error by at most $d_i^2+\text{(FP)}$ on regular cells: asymptotically exact | exactness gap | `mode="local"`, `local_regularity` |
| 7.1 | Exact trust frontier $C(k)$ for all $k$ by a DP on the cycle in $O(N^2)$ | algorithm | `trust_frontier` |
| 8.1 | Threshold policy for the global budget: $\mathrm{LP}\le e^{\alpha\hat w}(\alpha\,\mathrm{ALG}+BL/e)$, $\alpha=1+\ln(U/L)$ | competitive | `ThresholdPolicy` |
| 8.3 | Robustification of any budget heuristic by budget splitting | competitive | `RobustifiedPolicy` |
| 8.4, 8.5 | Greedy is no better than $\theta$-competitive; uniform pacing no better than $\max(\theta,T)$ | lower bounds | E4 |

Theorem 8.1 is a self-contained variant of the threshold algorithm of
Zhou, Chakrabarty and Lukose (2008); Theorem 8.3 is the standard "combine two
algorithms by splitting the resource" argument. They are included because the
certified budget is new as an *application* of online knapsack, not because the
arguments are new. Novelty claims about Sections 4–7 have **not** been checked
by a literature search in this note (Section 10).

## 1. Setting, notation and assumptions

* $N\ge 2$ periodic cells, indices mod $N$, $h=1/N$, $\|w\|:=h\sum_i|w_i|$.
* $\lambda$ is the real value of the binary64 number `lam`; $M$ the binary64 envelope; $K=[-M,M]^N$.
* $f(u)=u^2/2$, $p(a)=f(a^+)=\tfrac12\max(a,0)^2$, $q(b)=f(b^-)=\tfrac12\min(b,0)^2$,
  $G(a,b)=\max(p(a),q(b))$.
* Face $f$ carries the input $x_f=(v_f,v_{f+1})$; cell $i$ lies between faces $i-1$ and $i$.
  $S(v)_i=v_i-\lambda\,(G(x_i)-G(x_{i-1}))$.
* $F$ is the real-arithmetic ReLU network, $\hat F$ its binary64 implementation
  (`FrozenFlux.predict`: bias first, then products added input by input), $\hat G$ the
  binary64 `godunov_flux`. $D:=F-G$ and $\phi:=\hat F-F$.
* $\boldsymbol u=2^{-53}$, $\eta=2^{-1075}$ (half the least subnormal); $\mathrm{fl}$ is round-to-nearest-even;
  $\mathrm{pred},\mathrm{succ}$ are `nextafter` towards $\mp\infty$.
* $\mathrm{TV}(v)=\sum_i|v_{i+1}-v_i|$.

**Assumptions.**

* **H1** IEEE-754 binary64, round-to-nearest-even, gradual underflow. Any nonfinite intermediate raises (fail closed).
* **H2** NumPy element-wise ufuncs (`+ - * abs maximum minimum clip nextafter` and comparisons) are single correctly rounded operations, without FMA contraction; `np.sum` of $n$ terms evaluates some binary tree of correctly rounded additions. This is a platform assumption, tested but not proved.
* **H3** $\lambda M\le 1$ (checked in exact rationals).
* **H4** $v_0\in K$ with binary64 entries.

## 2. Facts about the scheme

**Lemma 2.1 (Godunov flux).** $G$ is the Godunov flux of $f$:
$G(a,b)=\min_{[a,b]}f$ for $a\le b$ and $\max_{[b,a]}f$ for $a>b$. The functions $p,q$
are $C^1$ with $p'(a)=a^+$, $q'(b)=b^-$; $p$ is nondecreasing and $q$ nonincreasing.
Hence $G$ is nondecreasing in $a$, nonincreasing in $b$, locally Lipschitz, and $G(u,u)=f(u)$.

*Proof.* $f$ is convex with minimum at $0$. If $a\le b$: for $0\le a$ the minimum is $f(a)=p(a)$ and $q(b)=0$;
for $b\le 0$ it is $f(b)=q(b)$ and $p(a)=0$; for $a<0<b$ it is $0=p(a)=q(b)$. If $a>b$ the maximum
of the convex $f$ on $[b,a]$ is $\max(f(a),f(b))$; for $a\ge0\ge b$ this is $\max(p(a),q(b))$, for $a>b\ge 0$ it is
$f(a)=p(a)$ with $q(b)=0$, and for $0\ge a>b$ it is $f(b)=q(b)$ with $p(a)=0$. The function $a\mapsto (a^+)^2/2$ has
one-sided derivatives $0$ at $0$ on both sides, so $p\in C^1$ with $p'=a^+$; likewise $q$. $\square$

**Lemma 2.2 (monotone scheme, invariant box).** Under H3, $S$ is nondecreasing in each argument on $K$ and $S(K)\subseteq K$.

*Proof.* $S_i$ depends on $(v_{i-1},v_i,v_{i+1})$. It is nondecreasing in $v_{i-1}$ because $+\lambda G(v_{i-1},v_i)$
is, and in $v_{i+1}$ because $-\lambda G(v_i,v_{i+1})$ is (Lemma 2.1). For the middle argument let
$\varphi(t)=t-\lambda G(t,c)+\lambda G(c',t)$ and $s<t$ in $[-M,M]$ with $s,t$ on the same side of $0$.
Because $\max$ is 1-Lipschitz in each slot, $p'(r)=r^+$ is nondecreasing and $-q'(r)=|r^-|$ is nonincreasing,
$G(t,c)-G(s,c)\le p(t)-p(s)\le (t-s)\,t^+$ and $G(c',s)-G(c',t)\le q(s)-q(t)\le (t-s)\,|s^-|$. Hence
$\varphi(t)-\varphi(s)\ge (t-s)(1-\lambda(t^++|s^-|))\ge (t-s)(1-\lambda M)\ge 0$, since on one side of $0$
one of $t^+,|s^-|$ vanishes and the other is at most $M$. For $s<0<t$ write
$\varphi(t)-\varphi(s)=[\varphi(t)-\varphi(0)]+[\varphi(0)-\varphi(s)]\ge 0$. Monotonicity in one coordinate at a time,
with intermediate vectors in $K$, gives $v\le w\Rightarrow S(v)\le S(w)$ on $K$. Constant vectors are fixed points
(all face fluxes equal), so $(\min v)\mathbf 1\le v\le(\max v)\mathbf 1$ implies $\min v\le S(v)_i\le\max v$. $\square$

**Lemma 2.3 (Crandall–Tartar $L^1$ contraction).** For $a,b\in K$: $\|S(a)-S(b)\|\le\|a-b\|$.

*Proof.* Let $m=a\wedge b$, $q=a\vee b$ (componentwise); both lie in $K$ and $m\le a,b\le q$. By Lemma 2.2
$S(m)\le S(a),S(b)\le S(q)$, so $|S(a)_i-S(b)_i|\le S(q)_i-S(m)_i$. The periodic flux differences telescope,
$\sum_iS(w)_i=\sum_iw_i$, hence $\sum_i|S(a)_i-S(b)_i|\le\sum_i(q_i-m_i)=\sum_i|a_i-b_i|$. $\square$

**Lemma 2.4 (projection).** Let $\Pi$ clip componentwise to $[-M,M]$. For $s\in K$ and any $p$:
$|\Pi(p)_i-s_i|\le|p_i-s_i|$. Floating `np.clip` is exact.

*Proof.* $\Pi$ is 1-Lipschitz in each coordinate and fixes $s_i$. `np.clip` returns one of $p_i,-M,M$. $\square$

This fixes review item A-1: the original rollout did not keep fallback states in $K$, which Lemma 2.3 needs.
Both the new rollout and the original `certified_rollout` now clip.

## 3. Floating-point facts

**Lemma 3.1 (outward rounding).** For binary64 $x,y$ and $\circ\in\{+,-,\times\}$ with finite
$c=\mathrm{fl}(x\circ y)$: $\mathrm{pred}(c)\le x\circ y\le\mathrm{succ}(c)$. Consequently the operations of `VI`
return intervals that contain the exact result whenever the operands' intervals contain their exact operands.

*Proof.* $c$ is a nearest float to $z=x\circ y$. If $z<\mathrm{pred}(c)$, then $\mathrm{pred}(c)$ would be strictly nearer to $z$
than $c$; similarly above. For intervals: the exact result of $+,-$ is monotone in each operand, and a product
$xy$ over a box attains its extremes at corners; each exact corner value is $\ge\mathrm{pred}(\mathrm{fl}(\text{corner}))\ge
\mathrm{pred}(\min\mathrm{fl}(\text{corners}))$, because `pred` is nondecreasing. `mul_point` is the special case with a
degenerate second operand. $\square$

**Lemma 3.2 (network enclosure).** `network_value_enclosure` evaluates the network on a box with the *same
operation sequence* as $\hat F$, each operation outward rounded. Its result contains both $F(x)$ and $\hat F(x)$ for
every $x$ in the box.

*Proof.* Induction over operations. Real values: Lemma 3.1. Floating values: if $\hat x\in X$, $\hat y\in Y$ then,
$\mathrm{fl}$ being monotone, $\mathrm{fl}(\hat x+\hat y)\in[\mathrm{fl}(x_{lo}+y_{lo}),\mathrm{fl}(x_{hi}+y_{hi})]\subseteq
[\mathrm{pred}(\cdot),\mathrm{succ}(\cdot)]$, and $\mathrm{fl}(\hat x w)$ lies between $\mathrm{fl}(x_{lo}w)$ and $\mathrm{fl}(x_{hi}w)$.
ReLU is exact and monotone. $\square$

**Lemma 3.3 (face bounds).** (a) *Table.* For every $x$ in the closed bin $B_{pq}$,
$|\hat F(x)-G(x)|\le\delta_{pq}$. (b) *Point.* `face_bounds(mode="local")` returns
$\delta^{pt}_f\ge|\hat F(x_f)-G(x_f)|$.

*Proof.* On a box $\beta=[a_{lo},a_{hi}]\times[b_{lo},b_{hi}]$, Lemma 3.2 gives $\hat F(x)\in[\nu_{lo},\nu_{hi}]$ and
Lemma 2.1 gives $G(\beta)=[G(a_{lo},b_{hi}),G(a_{hi},b_{lo})]$; `godunov_value_enclosure` rounds these two corner values
outward. Then $|\hat F(x)-G(x)|\le\max(\nu_{hi}-g_{lo},\,g_{hi}-\nu_{lo})$, rounded up. `sub_edges` returns sub-bin edges
whose first and last entries are the bin edges and in which consecutive closed sub-intervals share the *same* float
endpoint (a running maximum enforces order), so the sub-boxes cover the bin exactly. `bin_index` sends every value to
a closed bin containing it. (b) is (a) with the degenerate box $\{x_f\}$. $\square$

**Lemma 3.4 (floating Godunov flux).** For $|a|,|b|\le M$:
$|\hat G(a,b)-G(a,b)|\le\varepsilon_G:=\boldsymbol uM^2/2+(3M+2)\eta$.

*Proof.* `godunov_flux` selects a branch with exact comparisons, except that shocks ($a>b$) test
$s=\mathrm{fl}(0.5\cdot\mathrm{fl}(a+b))\ge 0$. A sum of two floats is an integer multiple of $2^{-1074}$, so
$\mathrm{fl}(a+b)$ has the sign of $a+b$ and vanishes only if $a+b=0$; halving can only produce a signed zero when
$|a+b|=2^{-1074}$. The test `-0.0 >= 0.0` is true, so the only wrong branch is $a+b=-2^{-1074}$, where it returns $f(a)$
instead of $f(b)$ and $|f(a)-f(b)|=|a-b||a+b|/2\le 2M\eta$. The value $\mathrm{fl}(\mathrm{fl}(0.5x)\,x)$: halving is exact
unless the result is subnormal, so $t=x/2+\tau$ with $|\tau|\le\eta$; then $\mathrm{fl}(tx)=tx(1+\theta)+\eta_2$ with
$|\theta|\le\boldsymbol u$, $|\eta_2|\le\eta$, whence $|\mathrm{fl}(tx)-x^2/2|\le\boldsymbol u x^2/2+(1+\boldsymbol u)|x|\eta+\eta$.
With $|x|\le M$ and the branch term the claim follows. $\square$

**Lemma 3.5 (update defect).** For face fluxes $\Phi$ (floats), let $dF=\mathrm{fl}(\Phi_i-\Phi_{i-1})$,
$m=\mathrm{fl}(\lambda\,dF)$, $p=\mathrm{fl}(v_i-m)$ and $\tilde P_i=v_i-\lambda(\Phi_i-\Phi_{i-1})$ (exact). Then
$$|p-\tilde P_i|\le \tfrac{\boldsymbol u}{1-\boldsymbol u}|p|+\lambda\tfrac{2\boldsymbol u+\boldsymbol u^2}{1-\boldsymbol u}|dF|+\eta .$$

*Proof.* $dF=(\Phi_i-\Phi_{i-1})(1+\theta_1)$, $m=\lambda\,dF(1+\theta_2)+\eta_2$, $p=(v_i-m)(1+\theta_3)$ with
$|\theta_k|\le\boldsymbol u$, $|\eta_2|\le\eta$ (addition and subtraction are exact in the subnormal range). Then
$p-\tilde P_i=\theta_3(v_i-m)+(\lambda(\Phi_i-\Phi_{i-1})-m)$ with
$|\lambda(\Phi_i-\Phi_{i-1})-m|\le\lambda|\Phi_i-\Phi_{i-1}|(2\boldsymbol u+\boldsymbol u^2)+\eta$,
$|\Phi_i-\Phi_{i-1}|\le|dF|/(1-\boldsymbol u)$ and $|v_i-m|\le|p|/(1-\boldsymbol u)$. $\square$

`FPConstants.update_defect` evaluates the right-hand side with constants rounded up and the absolute term
replaced by $2^{-1072}=8\eta$. With nonnegative operands a product can lose at most $\eta$ to underflow, so the
computed term satisfies $\hat t\ge(1-\boldsymbol u)^3(\text{exact right-hand side})$; Lemma 3.6 handles the sum.
`update_defect_apriori` bounds the *returned value* for every state in $K$ with $|\Phi|\le\Phi_{\max}$, using
$|dF|\le 2\Phi_{\max}(1+\boldsymbol u)$, $|p|\le(M+\lambda|dF|(1+\boldsymbol u)+\eta)(1+\boldsymbol u)$ and
$(1+\boldsymbol u)^k\le 1/(1-k\boldsymbol u)$.

**Lemma 3.6 (sums of nonnegative floats).** If $\hat s$ is any binary-tree sum of $n$ nonnegative floats $t_j$ in
round-to-nearest, then $(1-\boldsymbol u)^{n-1}\sum t_j\le\hat s\le(1+\boldsymbol u)^{n-1}\sum t_j$. Hence
$\sum t_j\le\hat s/(1-(n-1)\boldsymbol u)$ (`rigorous_nonneg_sum`, finished in exact rationals).

*Proof.* Each addition of nonnegative numbers has relative error at most $\boldsymbol u$ (exact when subnormal), and
every leaf passes through at most $n-1$ additions. Bernoulli: $(1-\boldsymbol u)^{n-1}\ge 1-(n-1)\boldsymbol u$. $\square$

## 4. The mixed (face-selective) certificate

Let $s\in\{0,1\}^N$ mark neural faces. The step evaluates $\Phi_f=\hat F(x_f)$ if $s_f=1$ and $\Phi_f=\hat G(x_f)$
otherwise, forms $p$ as in Lemma 3.5 and returns $v^+=\Pi(p)$. Face errors are $\tilde e_f=\Phi_f-G(x_f)$.
`cell_costs` returns, rounded up,
$$\kappa_i(0,0)=2\varepsilon_G,\quad\kappa_i(1,0)=\delta_{i-1}+\varepsilon_G,\quad\kappa_i(0,1)=\varepsilon_G+\delta_i,\quad
\kappa_i(1,1)=\min(\delta_{i-1}+\delta_i,\ \Delta_i),$$
where $\delta_f$ are face bounds (Lemma 3.3) and $\Delta_i$ is the difference bound of Section 5
(`mode="separable"` omits $\Delta_i$ and reproduces the original Stage A certificate).

**Theorem 4.1 (one step, any trust pattern).** For $v\in K$ and every $s$,
$$\|v^+-S(v)\|\le\lambda h\sum_i\kappa_i(s_{i-1},s_i)+\rho_{upd},$$
where $\rho_{upd}$ is `update_defect`. `rigorous_step_cost` returns an upper bound of the first term.

*Proof.* $S(v)\in K$ (Lemma 2.2), so by Lemma 2.4 $\|v^+-S(v)\|\le\|p-S(v)\|\le\|p-\tilde P\|+\|\tilde P-S(v)\|$.
The first term is at most $\rho_{upd}$ (Lemmas 3.5, 3.6). Since $\tilde P_i-S(v)_i=-\lambda(\tilde e_i-\tilde e_{i-1})$,
the second is $\lambda h\sum_i|\tilde e_i-\tilde e_{i-1}|$. Per cell: a Godunov face has $|\tilde e|\le\varepsilon_G$
(Lemma 3.4), a neural face $|\tilde e|\le\delta$ (Lemma 3.3); if both faces are neural the triangle inequality and
Corollary 5.4 give $\min(\delta_{i-1}+\delta_i,\Delta_i)$. All additions in `cell_costs` round up; the final sum uses
Lemma 3.6 and exact rationals for $\lambda$ and $h=1/N$. $\square$

**Theorem 4.2 (global bound and budget).** Let $v_{n+1}$ be produced from $v_n$ with *any* trust pattern $s^n$
chosen by any rule depending on the past, and $u_{n+1}=S(u_n)$, $u_0=v_0$. With $\eta_n$ the step certificate of
Theorem 4.1, $\|v_n-u_n\|\le\sum_{m<n}\eta_m$. In `certified_rollout_a2` moreover $\sum_{m<T}\eta_m\le\varepsilon_{tot}$.

*Proof.* $v_n,u_n\in K$ by Lemmas 2.2 and 2.4. $\|v_{n+1}-u_{n+1}\|\le\|v_{n+1}-S(v_n)\|+\|S(v_n)-S(u_n)\|\le\eta_n+\|v_n-u_n\|$
by Theorem 4.1 and Lemma 2.3; induct.
*Budget.* Let $\bar c_0$ be the a priori bound on the all-Godunov frontier value and $\bar\rho$ the a priori bound of
Lemma 3.5 (`per_step_reserve`), $B=\varepsilon_{tot}-T(\bar c_0+\bar\rho)\ge 0$ (checked). The option with $k$ neural faces
is charged $w(k)\ge\bar C(k)-\bar c_0$ (rounded up), where $\bar C(k)$ = `frontier_upper` bounds the real cost of the
pattern the DP returns (Lemma 3.6 for the DP's sequential sums); $w(0)=0$ because $\bar C(0)\le\bar c_0$ (asserted at run
time and true a priori). The rollout enforces $\sum_n w(k_n)\le B$ in exact rationals: a proposal that would exceed $B$
is replaced by the zero option, whatever the policy's own floating bookkeeping says (tested with a policy that ignores
the budget). Hence
$\eta_n\le\bar C(k_n)+\rho_n\le\bar c_0+w(k_n)+\bar\rho$ and $\sum_n\eta_n\le T(\bar c_0+\bar\rho)+B=\varepsilon_{tot}$.
The rollout also re-checks the invariant $\text{spent}_n\le\varepsilon_{tot}-(T-n-1)(\bar c_0+\bar\rho)$ in exact rationals. $\square$

The routing rule may be any function of the past — including a learned one — without affecting soundness.
This separation (soundness from Theorems 4.1–4.2, performance from Sections 7–8) is the design principle.

## 5. The difference-aware certificate

The original table bounds $|\tilde e_{i-1}|$ and $|\tilde e_i|$ separately and adds them. But the state only sees
$\tilde e_i-\tilde e_{i-1}$, and adjacent inputs $x_{i-1}=(v_{i-1},v_i)$, $x_i=(v_i,v_{i+1})$ are close wherever the
solution is smooth. Write $d=x_i-x_{i-1}=(v_i-v_{i-1},\,v_{i+1}-v_i)$.

**Lemma 5.1 (a ReLU network along a segment).** Let $\Omega$ be a box containing the segment
$\{x+td:t\in[0,1]\}$ and let $([g_a],[g_b])$ be the output of `network_value_and_gradient` on $\Omega$. Then
$\varphi(t)=F(x+td)$ is continuous and piecewise affine with finitely many pieces, and on the interior of each piece
$\varphi'(t)=\gamma_a d_a+\gamma_b d_b$ for some $\gamma_a\in[g_a]$, $\gamma_b\in[g_b]$.

*Proof.* By induction over layers every pre-activation $z_j(t)$ and activation $y_j(t)$ is continuous piecewise affine
with finitely many pieces: affine maps preserve this, and on each affine piece of $z_j$ either $z_j$ keeps one sign,
crosses $0$ once (one new breakpoint), or vanishes identically. Refine to a common finite partition. On the interior of a
piece each $y_j'=s_jz_j'$ with $s_j=1$ if $z_j>0$, $s_j=0$ if $z_j<0$, and $s_j$ arbitrary if $z_j\equiv0$ (then $z_j'=0$).
Hence $\varphi'(t)=W_LS_{L-1}W_{L-1}\cdots S_1W_1d=\gamma\cdot d$. Because $x+td\in\Omega$ and the value part of
`network_value_and_gradient` is the outward-rounded recursion of Lemma 3.2, $z_j(t)$ lies in the computed pre-activation
interval; `relu_slope` returns $[1,1]$ only if that interval is in $(0,\infty)$ (forcing $s_j=1$),
$[0,0]$ only if it is in $(-\infty,0)$, and $[0,1]$ otherwise. So every admissible $s_j$ lies in the slope interval. The
forward recursion computes, with outward rounding, the same sums and products with point weights and slope intervals, so
by inclusion monotonicity (Lemma 3.1) $\gamma\in([g_a],[g_b])$. $\square$

**Lemma 5.2 (Godunov along a segment).** With $\Omega$ as above, let `godunov_gradient_cases` return
$(\mathrm{has}_p,P_a,\mathrm{has}_q,Q_b)$. Then $\psi(t)=G(x+td)$ is continuous and piecewise $C^1$ with finitely many
pieces, and at every non-breakpoint $\psi'(t)=a(t)^+d_a$ (branch $p$) or $\psi'(t)=b(t)^-d_b$ (branch $q$), with
$a(t)^+\in P_a$, $b(t)^-\in Q_b$; branch $q$ does not occur if $\mathrm{has}_q$ is false, and likewise for $p$.

*Proof.* $P(t)=p(a(t))$, $Q(t)=q(b(t))$ are $C^1$ (Lemma 2.1). $P-Q$ is piecewise polynomial of degree $\le 2$ with at
most three pieces (sign changes of the affine $a(t)$, $b(t)$), so $\{P=Q\}$ is a finite union of points and closed
intervals. Where $P>Q$, $\psi'=P'=a^+d_a$; where $Q>P$, $\psi'=b^-d_b$; on an interval where $P\equiv Q$ both expressions
coincide. $\mathrm{has}_q$ is false only if the outward lower bound of $p$ on $\Omega$ exceeds the outward upper bound of $q$,
in which case $P>Q$ throughout. Monotonicity of $a\mapsto a^+$, $b\mapsto b^-$ gives the enclosures. $\square$

**Theorem 5.3 (difference enclosure).** With $[d]\ni d$ and
$J_p=([g_a]-P_a)[d_a]+[g_b][d_b]$, $J_q=[g_a][d_a]+([g_b]-Q_b)[d_b]$ and $J$ the hull of the flagged $J$'s (all interval
operations outward rounded), $D(x+d)-D(x)\in J$.

*Proof.* $\chi=\varphi-\psi$ is continuous and piecewise $C^1$ on a common finite partition, so
$D(x+d)-D(x)=\chi(1)-\chi(0)=\int_0^1\chi'(t)\,dt$. At every non-breakpoint, Lemmas 5.1 and 5.2 give
$\chi'(t)=(\gamma_a-a^+)d_a+\gamma_bd_b\in J_p$ or $\chi'(t)=\gamma_ad_a+(\gamma_b-b^-)d_b\in J_q$ (inclusion
monotonicity), hence $\chi'(t)\in J$. An integrable function with values in a closed interval has its average over $[0,1]$
in that interval. $\square$

No Clarke calculus is needed: restricting to the segment makes every function one-dimensional and piecewise smooth.

**Corollary 5.4 (difference bounds).** With $\varepsilon_{fp}(x)\ge|\hat F(x)-F(x)|$ (per-bin table computed by the
exact-rational `fp_roundoff_bound`),
$$|\tilde e_i-\tilde e_{i-1}|\le\Delta_i:=|J|_{\uparrow}+\varepsilon_{fp}(x_{i-1})+\varepsilon_{fp}(x_i)$$
whenever both faces are neural, where $J$ is computed either on the **triple box**
$\Omega_{pqr}=\mathrm{hull}(B_p\cup B_q)\times\mathrm{hull}(B_q\cup B_r)$ from offline gradient tables ($p,q,r$ the bins of
$v_{i-1},v_i,v_{i+1}$; `mode="table"`), or on the **local box** $\Omega_i=\mathrm{hull}(x_{i-1},x_i)$ online
(`mode="local"`). Both boxes contain $x_{i-1}$ and $x_i$.

*Proof.* $\tilde e_i-\tilde e_{i-1}=D(x_i)-D(x_{i-1})+\phi(x_i)-\phi(x_{i-1})$; apply Theorem 5.3 and the triangle
inequality. $[d]$ is formed by outward-rounded subtraction of the float states. $\square$

**Proposition 5.5 (gauge invariance).** Replacing $F$ by $F+c$ (a change of the output bias) leaves the state update, the
exact error and $J$ unchanged, while any sound face-separable certificate must grow by up to $2\lambda|c|$.

*Proof.* Flux differences cancel $c$; $J$ uses only gradients. A face-separable bound must dominate $|\tilde e_f|$, which
contains $c$. $\square$ (Up to rounding of $\hat F$, which $\varepsilon_{fp}$ covers.)

## 6. Resolution behaviour

Let $A(v)=\lambda h\sum_i|\hat e_i-\hat e_{i-1}|$ be the exact all-neural error before update rounding
($\hat e_f=\hat F(x_f)-G(x_f)$), $E(u)=F(u,u)-f(u)$ the network's consistency defect, and $\bar\varepsilon=\max\varepsilon_{fp}$.
$\mathrm{Lip}_a,\mathrm{Lip}_b$ are Lipschitz constants of $D$ on $[-M,M]^2$ in each argument.

**Theorem 6.1 (face-separable certificates do not vanish).** For every per-face bound
$\beta(x)\ge|\hat F(x)-G(x)|$ and $B_{sep}(v)=\lambda h\sum_i(\beta(x_{i-1})+\beta(x_i))$,
$$B_{sep}(v)\ \ge\ 2\lambda\,h\sum_i|E(v_i)|-2\lambda\,\mathrm{Lip}_b\,h\,\mathrm{TV}(v)-2\lambda\bar\varepsilon,\qquad
A(v)\ \le\ \lambda(\mathrm{Lip}_a+\mathrm{Lip}_b)\,h\,\mathrm{TV}(v)+2\lambda\bar\varepsilon .$$
Hence if $v^h_i=v(x_i)$ samples a piecewise-continuous $v$ of bounded variation, then
$\liminf_{h\to0}B_{sep}(v^h)\ge 2\lambda\int_0^1|E(v(x))|\,dx-2\lambda\bar\varepsilon$ while $A(v^h)=O(h)+2\lambda\bar\varepsilon$:
unless the network is consistent along the solution ($E\circ v\approx 0$), the overestimation factor grows like $1/h$.

*Proof.* $\sum_i(\beta(x_{i-1})+\beta(x_i))=2\sum_i\beta(x_i)$ (periodic reindexing) and
$\beta(x_i)\ge|D(x_i)|-\varepsilon_{fp}(x_i)\ge|D(v_i,v_i)|-\mathrm{Lip}_b|v_{i+1}-v_i|-\bar\varepsilon$ with $D(u,u)=E(u)$
because $G(u,u)=f(u)$. The upper bound on $A$ follows from
$|\hat e_i-\hat e_{i-1}|\le|D(x_i)-D(x_{i-1})|+2\bar\varepsilon$ and $|d_a|=|v_i-v_{i-1}|$, $|d_b|=|v_{i+1}-v_i|$. The limit is
a Riemann sum; $\mathrm{TV}(v^h)\le\mathrm{TV}(v)$. $\square$

The measured factor for the original certificate grows from 177× ($N=64$) to 2,258× ($N=1024$) on smooth data
(median over 12 initial conditions × 3 training seeds; report, E1) — exactly this $1/h$ law.

**Theorem 6.2 (table difference certificate is $O(h)$).** Let $\Gamma_a,\Gamma_b$ be the largest magnitudes of the
interval coefficients of $[d_a]$, $[d_b]$ over the triple table and both branches. Then, up to terms of order
$\boldsymbol u$ and $2^{-1074}$,
$$A(v)\le B_{table}(v)\le\lambda(\Gamma_a+\Gamma_b)\,h\,\mathrm{TV}(v)+2\lambda\bar\varepsilon .$$

*Proof.* $B_{table}\ge A$ by Theorem 4.1. $\kappa_i(1,1)\le\Delta_i\le\Gamma_a|d_a|+\Gamma_b|d_b|+2\bar\varepsilon$ plus
outward-rounding terms; sum and use $\sum_i|d_{a,i}|=\sum_i|d_{b,i}|=\mathrm{TV}(v)$. $\square$

The constant does not shrink with $h$ because the triple box has the fixed width of two bins; the measured factor is
flat in $N$: 14–31× with 16 bins and 8–16× with 32 bins on smooth and shocked data, 50–110× on Riemann data.

**Definition (regular cell).** Cell $i$ is *regular* if on $\Omega_i=\mathrm{hull}(x_{i-1},x_i)$ every hidden
pre-activation interval is strictly signed and exactly one Godunov branch is flagged (`local_regularity`). Let
$\beta_i\in\{p,q\}$ be that branch and $d_{i,p}=v_i-v_{i-1}$, $d_{i,q}=v_{i+1}-v_i$.

**Theorem 6.3 (exactness gap of the local certificate).** For every $v\in K$ and every cell,
$0\le\kappa^{loc}_i(1,1)-|\hat e_i-\hat e_{i-1}|$, and on regular cells
$$\kappa^{loc}_i(1,1)-|\hat e_i-\hat e_{i-1}|\ \le\ d_{i,\beta_i}^2+2\big(\varepsilon_{fp}(x_{i-1})+\varepsilon_{fp}(x_i)\big)+r_i,$$
where $r_i$ is the widening caused by outward rounding, $r_i\le c_{net}\,\boldsymbol u\,(|d_{i,p}|+|d_{i,q}|)$ with
$c_{net}$ depending only on the weights and $M$. Consequently
$$0\le B_{loc}(v)-A(v)\le\lambda h\sum_{i\ \mathrm{regular}}\big(d_{i,\beta_i}^2+r_i\big)+\lambda h\sum_{i\ \mathrm{irregular}}\kappa_i^{loc}(1,1)+4\lambda\bar\varepsilon .$$

*Proof.* Lower bound: Theorem 4.1. On a regular cell the activation pattern is constant on $\Omega_i$, so $F$ is affine
there with gradient $\gamma$, and the forward recursion multiplies point weights by point slopes: $[g]\subseteq\gamma\pm$
(rounding). Say $\beta_i=p$; then $G=p(a)$ on $\Omega_i$ and
$D(x_i)-D(x_{i-1})=(\gamma_a-\bar a)d_a+\gamma_bd_b$ with $\bar a=\int_0^1(v_{i-1}+td_a)^+dt\in P_a$. The exact-arithmetic
value set $\{(\gamma_a-c)d_a+\gamma_bd_b:\,c\in P_a\}$ is an interval of length $|P_a||d_a|$ containing the true value, and
$|P_a|=a_{hi}^+-a_{lo}^+\le a_{hi}-a_{lo}=|d_a|$ because $\Omega_i$ is exactly the hull in $a$. Hence
$|J|_\uparrow\le|D(x_i)-D(x_{i-1})|+d_a^2+r_i$, and with $\kappa\le\Delta_i$ and
$|\hat e_i-\hat e_{i-1}|\ge|D(x_i)-D(x_{i-1})|-\varepsilon_{fp}(x_{i-1})-\varepsilon_{fp}(x_i)$ the per-cell bound follows;
branch $q$ is symmetric. Summation gives the second display. $\square$

**Corollary 6.4 (asymptotic exactness).** If $v^h$ samples an $L_v$-Lipschitz $v$, at most $m$ cells are irregular, and
$\Gamma_{loc}$ bounds the magnitudes of the interval coefficients of $[d_a],[d_b]$ in the local enclosure (e.g. the
network's Lipschitz constant plus $M$), then $B_{loc}-A\le\lambda\big(L_v^2+2m\Gamma_{loc}L_v\big)h^2+O(\bar\varepsilon+\boldsymbol u)$. Whenever $A(v^h)\ge c\,h$,
$B_{loc}/A=1+O(h)$.

The irregular count is bounded in $h$ when the curve $x\mapsto(v(x),v(x))$ crosses the activation boundaries of $F$ and
the sonic set $\{v=0\}$ finitely often: each crossing makes one or two cells irregular. This is a hypothesis about $v$ and
$F$, not a theorem, and the constant is not small. Measured (E1, median): 43 → 96 irregular cells on smooth data as $N$ goes
64 → 1024, flat from $N=512$ on (a width-16 network with two hidden layers has 32 hidden units, and a sine crosses each
boundary about twice). At $N=1024$ the irregular cells carry ~87% of the gap ($7.3\cdot10^{-6}$ of $8.4\cdot10^{-6}$ against an
exact error of $5.4\cdot10^{-5}$), and every regular cell satisfied the per-cell inequality of Theorem 6.3 without using $r_i$
(180 runs). Measured factors: 1.60 → 1.14 on smooth data, 1.61 → 1.20 after shock formation. On Riemann data the factor
stays at 1.36: the whole error sits in four irregular jump cells, which Theorem 6.3 does not make exact.

## 7. Routing: the exact trust frontier

**Theorem 7.1 (DP on the cycle).** For costs $\kappa\in\mathbb R_{\ge0}^{N\times2\times2}$, `trust_frontier` returns
$C(k)=\min\{\sum_i\kappa_i(s_{i-1},s_i):s\in\{0,1\}^N,\ |s|=k\}$ for every $k=0,\dots,N$, and `select(k)` returns a
minimizer, in $O(N^2)$ time and memory.

*Proof.* Fix $\sigma=s_{N-1}$, the left face of cell 0. For $f=0,\dots,N-1$ let $V_f(t,k)$ be the minimum of
$\sum_{i=0}^f\kappa_i(s_{i-1},s_i)$ over prefixes $(s_0,\dots,s_f)$ with $s_f=t$, $\sum_{j\le f}s_j=k$ and $s_{-1}:=\sigma$.
Then $V_0(t,t)=\kappa_0(\sigma,t)$ and $V_f(t,k)=\min_{t'}V_{f-1}(t',k-t)+\kappa_f(t',t)$, because the only coupling between
the prefix and cell $f$ is $s_{f-1}$ (principle of optimality). Patterns with $s_{N-1}=\sigma$ have total cost
$V_{N-1}(\sigma,k)$ at their best, and cell 0 is consistent with $s_{-1}=s_{N-1}=\sigma$. Minimizing over $\sigma\in\{0,1\}$
gives $C(k)$. Back-pointers store the minimizing $t'$; the traceback verifies the count and the closing condition. Each of
the $N$ stages does $O(N)$ work. $\square$

**Proposition 7.2 (floating DP).** With round-to-nearest sums, the DP returns for each $k$ a pattern minimizing the
*sequentially computed* float sum among all patterns with $|s|=k$ (since $x\mapsto\mathrm{fl}(x+c)$ is nondecreasing, the
minimum commutes with the stage update), so its real cost is within a factor
$(1+(N-1)\boldsymbol u)/(1-(N-1)\boldsymbol u)$ of optimal. Soundness never depends on this: the executed pattern is
re-certified (Theorem 4.1) and charged an upper bound (Theorem 4.2).

**Proposition 7.3 (routing needs no solver call).** The pattern and its certificate are functions of the current state,
the offline tables and the network weights only. The Godunov flux is evaluated only on faces with $s_f=0$ of the executed
step, and no exact oracle is called (tested with a monkeypatched oracle). In `table` mode the verifier uses table
look-ups only; in `local` mode it evaluates interval extensions of the network at every face (cost measured in E5).

**Remark (why partial trust is cheap where it helps).** Under the local certificate, the cost of a pattern is
$\sum_{\text{interior of neural blocks}}\Delta_i+\sum_{\text{block interfaces}}(\delta_f+\varepsilon_G)$: every
neural/classical interface pays one face error. Excluding a region is worth it exactly when its difference errors exceed
the two interface payments. Shocks and sonic points, where $G$ is nonsmooth and $\Delta_i$ is large, are therefore the
faces the frontier gives back to Godunov first (report, E2 trust map). Holding the temporal rule fixed, face-selective
routing trusts 2.6× (budget 30%) and 1.6× (budget 60%) as many faces as the all-or-nothing rule (E3).

## 8. Online allocation of the global budget

**Model.** At step $n$ a menu $\{(v_{nj},w_{nj})\}_{j=0}^{m_n}$ is revealed (adaptively; it may depend on earlier choices),
with $(v_{n0},w_{n0})=(0,0)$, $v,w\ge 0$, and $w_{nj}=0\Rightarrow v_{nj}=0$. An online rule picks one option per step with
$\sum_nw_{nj_n}\le B$. In the rollout $v=k$ (neural faces) and $w$ the charged excess cost. The benchmark is the hindsight
fractional optimum on the realized menus,
$$\mathrm{LP}(B)=\max\Big\{\sum_{n,j}x_{nj}v_{nj}:\ \sum_jx_{nj}\le1,\ \sum_{n,j}x_{nj}w_{nj}\le B,\ x\ge0\Big\}\ \ge\ \mathrm{OPT}(B).$$
(The guarantee is instance-wise. It does not compare against a counterfactual rollout whose menus would differ.)

**Threshold policy.** Fix $0<L\le U$ with every density $v/w\le U$; $\theta=U/L$, $\alpha=1+\ln\theta$,
$\Psi(z)=\tfrac Le e^{\alpha z}=\tfrac Le(\theta e)^z$ on $z\in[0,1]$ (the threshold function of Zhou–Chakrabarty–Lukose).
With utilization $z_n=(\text{charged so far})/B$, choose among options with $z_n+w/B\le1$ one maximizing
$$\pi_n(j)=v_{nj}-B\int_{z_n}^{z_n+w_{nj}/B}\Psi(u)\,du .$$

**Theorem 8.1.** Let $\hat w=\max_{n,j}w_{nj}/B$. The threshold policy never exceeds $B$, and
$$\mathrm{LP}(B)\ \le\ e^{\alpha\hat w}\big(\alpha\,\mathrm{ALG}+BL/e\big).$$

*Proof.* Feasibility is enforced. $\int_{z_0}^{z_1}\Psi=(\Psi(z_1)-\Psi(z_0))/\alpha$ and $\Psi(z+\hat w)=\Psi(z)e^{\alpha\hat w}$.
Let $z^*$ be the final utilization and $j_n$ the chosen options.
*(i)* $\pi_n(0)=0$, so $\pi_n(j_n)\ge0$ and $\mathrm{ALG}=\sum_nv_{nj_n}\ge B\int_0^{z^*}\Psi=B(\Psi(z^*)-L/e)/\alpha$.
*(ii)* Let $\bar\Psi=\Psi(\min(1,z^*+\hat w))$. For any option $j$ at step $n$: if feasible then, $\Psi$ being increasing,
$v_{nj}\le\pi_n(j_n)+B\int_{z_n}^{z_n+w/B}\Psi\le\pi_n(j_n)+w_{nj}\bar\Psi$, because $z_n+w/B\le\min(1,z^*+\hat w)$; if
infeasible then $z^*\ge z_n>1-\hat w$, so $\bar\Psi=\Psi(1)=U$ and $v_{nj}\le Uw_{nj}\le\pi_n(j_n)+w_{nj}\bar\Psi$.
*(iii)* For a feasible fractional $x$: $\sum_jx_{nj}v_{nj}\le\pi_n(j_n)+\bar\Psi\sum_jx_{nj}w_{nj}$ (use $\sum_jx_{nj}\le1$,
$\pi\ge0$). Summing, $\mathrm{LP}\le\sum_n\pi_n(j_n)+B\bar\Psi=\mathrm{ALG}-B\int_0^{z^*}\Psi+B\bar\Psi$.
*(iv)* Insert $\int_0^{z^*}\Psi=(\Psi(z^*)-L/e)/\alpha$ and $\bar\Psi\le\Psi(z^*)e^{\alpha\hat w}$:
$\mathrm{LP}\le\mathrm{ALG}+B\Psi(z^*)(e^{\alpha\hat w}-1/\alpha)+BL/(e\alpha)$. By (i), $B\Psi(z^*)\le\alpha\mathrm{ALG}+BL/e$, and
$e^{\alpha\hat w}-1/\alpha>0$, so $\mathrm{LP}\le\mathrm{ALG}+(\alpha\mathrm{ALG}+BL/e)(e^{\alpha\hat w}-1/\alpha)+BL/(e\alpha)
=e^{\alpha\hat w}(\alpha\,\mathrm{ALG}+BL/e)$. $\square$

$L$ is a free parameter: it trades the multiplicative $\alpha$ against the additive $BL/e$. The implementation evaluates
$\pi_n$ and the utilization in floating point; the theorem is stated for exact arithmetic, and the budget itself is
enforced exactly by the rollout (Theorem 4.2), so rounding in the policy can only change which option is chosen.
No lower bound on densities is assumed. Zhou, Chakrabarty and Lukose prove a purely multiplicative $\ln\theta+2$ for multiple-choice knapsack under
$L\le v/w\le U$ and infinitesimal weights, and $\ln\theta+1$ as a lower bound for every (even randomized) online
algorithm for plain online knapsack; Theorem 8.1 is the variant used here, with an explicit finite-weight factor.

**Corollary 8.2 (charge floor).** Densities in the rollout are unbounded (a well-certified face can cost ~$10^{-18}$).
Charging $w'(k)=w(k)+k\,c_{\min}$ with $c_{\min}=\beta B/(NT)$ gives $U=1/c_{\min}$, and
$\mathrm{LP}_{w}((1-\beta)B)\le\mathrm{LP}_{w'}(B)\le e^{\alpha\hat w}(\alpha\,\mathrm{ALG}+BL/e)$.

*Proof.* A fractional solution feasible for $w$ at budget $(1-\beta)B$ pays at most $c_{\min}\sum_{n,j}x_{nj}k\le c_{\min}NT=\beta B$
extra under $w'$. $\square$

**Theorem 8.3 (robustifying any heuristic).** Run the threshold policy on budget $\gamma B$ and any policy $\mathcal A$ on
$(1-\gamma)B$, each charged for its own proposal; execute the proposal of larger value. Then the executed weight is at most $B$,
$\mathrm{ALG}\ge\max\{\mathcal A((1-\gamma)B),\ \mathrm{THR}(\gamma B)\}$, and
$\gamma\,\mathrm{LP}(B)\le e^{\alpha\hat w/\gamma}\big(\alpha\,\mathrm{ALG}+\gamma BL/e\big)$.

*Proof.* The executed option is one of the two proposals, so its weight is at most the sum of both charges; each
sub-policy respects its own budget. Per step the executed value is the maximum of the two proposals, so the sum dominates
each sub-policy's total. Scaling a feasible fractional solution by $\gamma$ shows $\mathrm{LP}(\gamma B)\ge\gamma\mathrm{LP}(B)$;
apply Theorem 8.1 at budget $\gamma B$ (so $\hat w$ becomes $\hat w/\gamma$). $\square$

The sub-policies must be charged for proposals that are not executed; that slack is the price of the guarantee.
Measured (E3/E4): robustified pacing with $\gamma=0.1$ loses 5% against pure pacing on Burgers rollouts, and its worst ratio
over the adversarial instances was $1/\gamma=10$, where pure pacing reached $\theta$ (1,005 at $\theta=1000$).

**Proposition 8.4 (greedy).** "Take the most valuable option that fits" (with or without a per-step cap $\ge B/m$) has ratio at
least $\theta$ even as $\hat w\to0$.

*Proof.* $m$ steps each offering $(LB/m,B/m)$, then $m$ steps each offering $(UB/m,B/m)$. Greedy fills $B$ with the first
block ($\mathrm{ALG}=LB$) while $\mathrm{OPT}=UB$. $\square$

**Proposition 8.5 (pacing).** Adaptive uniform pacing (allowance = remaining budget / steps left) has ratio at least $T$ on a
single-burst instance and at least $\theta$ on the high-then-low instance.

*Proof.* Burst: step 1 offers options $(UjB/m,\,jB/m)$, $j\le m$, later steps offer nothing; pacing can afford weight at
most $B/T$ at step 1, so $\mathrm{ALG}\le UB/T$ versus $\mathrm{OPT}=UB$. High-then-low: $m$ steps of $(UB/m,B/m)$ then $m$
steps of $(LB/m,B/m)$ over $T=2m$; the allowance $B/(2m)$ admits nothing until the last $m$ steps. $\square$

## 9. What the measurements say (summary)

See the report for tables and figures. In brief: the $1/h$ law of Theorem 6.1 and the flat or decreasing factors of
Theorems 6.2–6.3 are visible across $N=64\dots1024$ and three training seeds; every audited one-step and global certificate
held against the exact-rational oracle; on Burgers rollouts adaptive pacing is within 0.3% (median) of the hindsight LP
bound, the worst-case-oriented threshold policy is 1.39–1.62× off, and greedy 1.43–2.07×; on the lower-bound instances
greedy and pacing fail by $\theta$ and $T$ while the threshold policy stays within 1.39×; the Python prototype's local
certificate costs ~300× a Godunov step at $N=128$, so there is no speed claim.

## 10. Claim boundaries, limitations and positioning

* **Target.** Same-grid exact-arithmetic Godunov, not the PDE. First-order Godunov has its own $O(\sqrt h)$ $L^1$ error
  across shocks, not included.
* **Arithmetic.** H1–H2 are platform assumptions; the code is tested, not formally verified. $c_{net}$ in Theorem 6.3 is
  not computed; $r_i$ is observed to be at the $10^{-16}$ relative level.
* **Scalar, periodic, first order, $H=1$.** The contraction argument (Lemma 2.3) is scalar. For systems (Euler) there is no
  general $L^1$ contraction, so Theorem 4.2 does not transfer; Theorem 4.1 and Section 5 do (they are one-step statements).
* **Budget policies** are competitive against the hindsight optimum on the realized menus; the menus depend on earlier
  decisions and the counterfactual optimum is not bounded. Threshold guarantees are worst-case and loose on benign data.
* **Cost.** No acceleration is claimed. In Burgers the solver is the cheapest component. The abstract saving is "solver
  face evaluations avoided"; whether it pays depends on the ratio of solver cost to network-plus-verifier cost.
* **Literature.** Section 8's algorithms are known (Zhou–Chakrabarty–Lukose 2008; budget-splitting combinations are
  standard in learning-augmented algorithms). Interval bound propagation and Lipschitz bounds for ReLU networks are standard
  in neural-network verification. Whether *difference-aware* (telescoping-aware) certification of neural numerical fluxes
  and certified face-wise routing have precedent has **not** been checked here and must be before any novelty claim.

## 11. Map from statements to code and tests

| Statement | Code | Test (`tests/stage_ab/test_stage_a2.py`) |
|---|---|---|
| 3.1 | `vinterval.VI` | `test_vector_intervals_enclose_exact_arithmetic` |
| 3.2, 3.3 | `network_value_enclosure`, `CertificateTables.build`, `face_bounds`, `sub_edges` | `test_network_value_gradient_and_difference_enclosures`, `test_sub_edges_cover_bins_exactly`, `test_difference_bound_dominates_actual_face_differences` |
| 3.4 | `FPConstants.eps_godunov` | `test_godunov_fp_error_bound_including_adversarial_inputs` |
| 3.5, 3.6 | `update_defect`, `update_defect_apriori`, `rigorous_nonneg_sum` | `test_update_defect_and_apriori_bound`, `test_rigorous_nonneg_sum_dominates_exact_sum` |
| 4.1 | `cell_costs`, `mixed_step`, `rigorous_step_cost` | `test_one_step_mixed_certificate_is_sound` (all modes, random patterns) |
| 4.2 | `certified_rollout_a2`, `per_step_reserve` | `test_certified_rollout_bounds_and_budget`, `test_rollout_enforces_budget_even_for_a_rogue_policy`, `test_rollout_without_audit_never_calls_exact_oracle` |
| 5.1–5.4 | `network_value_and_gradient`, `godunov_gradient_cases`, `difference_enclosure`, `difference_bounds` | `test_network_value_gradient_and_difference_enclosures`, `test_godunov_gradient_branch_cases` |
| 6.1–6.3 | `mode=...`, `local_regularity` | `test_separable_certificate_does_not_vanish_but_difference_does`; E1 (`r3_decomposition`) |
| 7.1, 7.2 | `trust_frontier`, `Frontier.select`, `frontier_upper` | `test_trust_frontier_matches_brute_force`, `test_frontier_upper_bounds_rigorous_cost` |
| 8.1, 8.3 | `ThresholdPolicy`, `RobustifiedPolicy` | `test_threshold_policy_competitive_bound_against_exact_lp`, `test_robustified_policy_budget_and_best_of_both` |
| 8.4, 8.5 | — | `test_greedy_and_pacing_lower_bound_instances`; E4 |

### References

* M. G. Crandall, L. Tartar (1980), Some relations between nonexpansive and order preserving mappings, *Proc. AMS* 78.
* M. G. Crandall, A. Majda (1980), Monotone difference approximations for scalar conservation laws, *Math. Comp.* 34.
* Y. Zhou, D. Chakrabarty, R. Lukose (2008), Budget constrained bidding in keyword auctions and online knapsack problems,
  WINE 2008, LNCS 5385, 566–576; full version D. Chakrabarty, Y. Zhou, R. Lukose, *Online knapsack problems*. Threshold
  $\Psi(z)=(Ue/L)^z(L/e)$; $\ln(U/L)+1$ upper and (randomized) lower bound for online knapsack; $\ln(U/L)+2$ for the
  multiple-choice variant; both under $L\le v/w\le U$ and infinitesimal weights.
* N. J. Higham (2002), *Accuracy and Stability of Numerical Algorithms*, 2nd ed., Ch. 2–4 (rounding models, summation).

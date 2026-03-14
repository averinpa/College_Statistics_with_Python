"""
College Statistics with Python - Interactive Dashboard
A Streamlit dashboard for exploring college-level statistics concepts with Python.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import (
    bernoulli, binom, poisson, geom, norm, t, f, chi2
)

st.set_page_config(
    page_title="College Statistics with Python",
    page_icon="📊",
    layout="wide",
)

# --- Sidebar Navigation ---
st.sidebar.title("College Statistics with Python")
st.sidebar.markdown(
    "Use Python to get intuition on complex concepts, "
    "empirically test theoretical proofs, or build algorithms from scratch."
)

topics = [
    "Home",
    "Bernoulli & Binomial",
    "Geometric & Poisson",
    "Sampling Distributions",
    "Confidence Intervals",
    "Hypothesis Testing",
    "Two-Sample Inference",
    "ANOVA",
    "Categorical Data (Chi-Square)",
    "Regression Analysis",
]
topic = st.sidebar.radio("Select a topic", topics)


# ======== Helper ========
def plot_distribution(x, y, title, xlabel, ylabel, color="steelblue", kind="bar"):
    fig, ax = plt.subplots(figsize=(8, 4))
    if kind == "bar":
        ax.bar(x, y, color=color, edgecolor="white")
    else:
        ax.plot(x, y, color=color, linewidth=2)
        ax.fill_between(x, y, alpha=0.3, color=color)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)


# ======== Pages ========

if topic == "Home":
    st.title("College Statistics with Python")
    st.markdown(
        """
        Welcome to the **College Statistics with Python** interactive dashboard!

        This app lets you explore core statistics concepts visually and interactively.
        Use the sidebar to navigate between topics.

        ### Topics covered
        | # | Topic | Key concepts |
        |---|-------|-------------|
        | 1 | **Bernoulli & Binomial** | Discrete probability, PMF, CDF |
        | 2 | **Geometric & Poisson** | Waiting time, rare events |
        | 3 | **Sampling Distributions** | Central Limit Theorem, standard error |
        | 4 | **Confidence Intervals** | Margin of error, t-distribution |
        | 5 | **Hypothesis Testing** | z-test, t-test, p-values, Type I/II errors |
        | 6 | **Two-Sample Inference** | Comparing proportions & means |
        | 7 | **ANOVA** | F-statistic, between/within group variance |
        | 8 | **Categorical Data** | Chi-square goodness-of-fit & independence |
        | 9 | **Regression Analysis** | Slope inference, nonlinear regression |

        ---
        *Built with [Streamlit](https://streamlit.io) for the
        [College Statistics with Python](https://github.com/enrolle/College_Statistics_with_Python) course.*
        """
    )

elif topic == "Bernoulli & Binomial":
    st.header("Bernoulli & Binomial Random Variables")

    tab1, tab2 = st.tabs(["Bernoulli", "Binomial"])

    with tab1:
        st.subheader("Bernoulli Distribution")
        p = st.slider("Probability of success (p)", 0.0, 1.0, 0.3, 0.01, key="bern_p")
        n_samples = st.slider("Number of samples", 100, 10_000, 1000, 100, key="bern_n")

        samples = bernoulli.rvs(p, size=n_samples)
        counts = [np.sum(samples == 0), np.sum(samples == 1)]
        plot_distribution(
            ["Failure (0)", "Success (1)"],
            [c / n_samples for c in counts],
            f"Bernoulli Distribution (p={p})",
            "Outcome", "Relative Frequency",
        )
        st.markdown(f"**Theoretical:** P(0) = {1-p:.3f}, P(1) = {p:.3f}")
        st.markdown(f"**Empirical:** P(0) = {counts[0]/n_samples:.3f}, P(1) = {counts[1]/n_samples:.3f}")

    with tab2:
        st.subheader("Binomial Distribution")
        n = st.slider("Number of trials (n)", 1, 50, 10, key="binom_n")
        p = st.slider("Probability of success (p)", 0.0, 1.0, 0.5, 0.01, key="binom_p")

        x = np.arange(0, n + 1)
        pmf = binom.pmf(x, n, p)
        plot_distribution(x, pmf, f"Binomial PMF (n={n}, p={p})", "k", "P(X=k)")

        st.markdown(f"**Expected value:** E(X) = n·p = {n*p:.2f}")
        st.markdown(f"**Variance:** Var(X) = n·p·(1−p) = {n*p*(1-p):.2f}")

        st.subheader("CDF")
        cdf = binom.cdf(x, n, p)
        plot_distribution(x, cdf, f"Binomial CDF (n={n}, p={p})", "k", "P(X≤k)", kind="line")

elif topic == "Geometric & Poisson":
    st.header("Geometric & Poisson Random Variables")
    tab1, tab2 = st.tabs(["Geometric", "Poisson"])

    with tab1:
        st.subheader("Geometric Distribution")
        p = st.slider("Probability of success (p)", 0.01, 1.0, 0.25, 0.01, key="geom_p")
        max_k = st.slider("Max trials to display", 5, 50, 20, key="geom_k")

        x = np.arange(1, max_k + 1)
        pmf = geom.pmf(x, p)
        plot_distribution(x, pmf, f"Geometric PMF (p={p})", "k (trial of first success)", "P(X=k)", color="darkorange")

        st.markdown(f"**Expected value:** E(X) = 1/p = {1/p:.2f}")

    with tab2:
        st.subheader("Poisson Distribution")
        lam = st.slider("Rate parameter (λ)", 0.1, 20.0, 5.0, 0.1, key="pois_lam")
        max_k = st.slider("Max k to display", 5, 50, 20, key="pois_k")

        x = np.arange(0, max_k + 1)
        pmf = poisson.pmf(x, lam)
        plot_distribution(x, pmf, f"Poisson PMF (λ={lam})", "k", "P(X=k)", color="seagreen")

        st.markdown(f"**E(X) = λ = {lam:.1f}**, Var(X) = λ = {lam:.1f}")

        st.subheader("Binomial → Poisson approximation")
        n_approx = st.slider("n (large)", 50, 5000, 500, 50, key="pois_n")
        p_approx = lam / n_approx
        x2 = np.arange(0, max_k + 1)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(x2 - 0.2, binom.pmf(x2, n_approx, p_approx), width=0.4, label=f"Binomial(n={n_approx}, p={p_approx:.4f})", color="steelblue")
        ax.bar(x2 + 0.2, poisson.pmf(x2, lam), width=0.4, label=f"Poisson(λ={lam})", color="seagreen", alpha=0.7)
        ax.set_title("Binomial → Poisson Approximation")
        ax.set_xlabel("k")
        ax.set_ylabel("P(X=k)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

elif topic == "Sampling Distributions":
    st.header("Sampling Distributions & Central Limit Theorem")

    pop_dist = st.selectbox("Population distribution", ["Normal", "Uniform", "Exponential", "Skewed"])
    pop_size = 100_000
    if pop_dist == "Normal":
        population = np.random.normal(50, 10, pop_size)
    elif pop_dist == "Uniform":
        population = np.random.uniform(0, 100, pop_size)
    elif pop_dist == "Exponential":
        population = np.random.exponential(10, pop_size)
    else:
        population = np.random.exponential(5, pop_size) ** 2

    sample_size = st.slider("Sample size (n)", 2, 200, 30, key="clt_n")
    n_samples = st.slider("Number of samples", 100, 5000, 1000, 100, key="clt_samp")

    sample_means = [np.mean(np.random.choice(population, sample_size)) for _ in range(n_samples)]

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(population, bins=50, color="gray", edgecolor="white", density=True)
        ax.set_title("Population Distribution", fontweight="bold")
        ax.set_xlabel("Value")
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(sample_means, bins=50, color="steelblue", edgecolor="white", density=True)
        ax.axvline(np.mean(population), color="red", linestyle="--", label=f"Pop mean = {np.mean(population):.2f}")
        ax.set_title(f"Sampling Distribution of X̄ (n={sample_size})", fontweight="bold")
        ax.set_xlabel("Sample Mean")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown(f"""
    | Statistic | Value |
    |-----------|-------|
    | Population mean | {np.mean(population):.4f} |
    | Mean of sample means | {np.mean(sample_means):.4f} |
    | Population SD | {np.std(population):.4f} |
    | Std error (σ/√n) | {np.std(population)/np.sqrt(sample_size):.4f} |
    | SD of sample means | {np.std(sample_means):.4f} |
    """)

elif topic == "Confidence Intervals":
    st.header("Confidence Intervals")

    ci_type = st.radio("Interval type", ["Proportion", "Mean (t-interval)"])

    if ci_type == "Proportion":
        true_p = st.slider("True population proportion", 0.01, 0.99, 0.5, 0.01, key="ci_p")
        n = st.slider("Sample size", 10, 1000, 100, key="ci_n")
        confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01, key="ci_conf")
        n_intervals = st.slider("Number of intervals to draw", 10, 100, 40, key="ci_ints")

        z = norm.ppf(1 - (1 - confidence) / 2)
        fig, ax = plt.subplots(figsize=(10, max(6, n_intervals * 0.18)))
        covers = 0
        for i in range(n_intervals):
            sample = bernoulli.rvs(true_p, size=n)
            p_hat = np.mean(sample)
            me = z * np.sqrt(p_hat * (1 - p_hat) / n)
            lo, hi = p_hat - me, p_hat + me
            color = "steelblue" if lo <= true_p <= hi else "red"
            if lo <= true_p <= hi:
                covers += 1
            ax.plot([lo, hi], [i, i], color=color, linewidth=1.5)
            ax.plot(p_hat, i, "o", color=color, markersize=3)

        ax.axvline(true_p, color="black", linestyle="--", label=f"True p = {true_p}")
        ax.set_title(f"{confidence*100:.0f}% Confidence Intervals — {covers}/{n_intervals} cover true p ({covers/n_intervals*100:.0f}%)", fontweight="bold")
        ax.set_xlabel("Proportion")
        ax.set_ylabel("Interval #")
        ax.legend()
        ax.grid(axis="x", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    else:
        true_mu = st.slider("True population mean", 0.0, 100.0, 50.0, 0.5, key="ci_mu")
        true_sigma = st.slider("Population std dev", 1.0, 30.0, 10.0, 0.5, key="ci_sig")
        n = st.slider("Sample size", 5, 200, 30, key="ci_n2")
        confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01, key="ci_conf2")
        n_intervals = st.slider("Number of intervals", 10, 100, 40, key="ci_ints2")

        t_crit = t.ppf(1 - (1 - confidence) / 2, df=n - 1)
        fig, ax = plt.subplots(figsize=(10, max(6, n_intervals * 0.18)))
        covers = 0
        for i in range(n_intervals):
            sample = np.random.normal(true_mu, true_sigma, n)
            x_bar = np.mean(sample)
            s = np.std(sample, ddof=1)
            me = t_crit * s / np.sqrt(n)
            lo, hi = x_bar - me, x_bar + me
            color = "steelblue" if lo <= true_mu <= hi else "red"
            if lo <= true_mu <= hi:
                covers += 1
            ax.plot([lo, hi], [i, i], color=color, linewidth=1.5)
            ax.plot(x_bar, i, "o", color=color, markersize=3)

        ax.axvline(true_mu, color="black", linestyle="--", label=f"True μ = {true_mu}")
        ax.set_title(f"{confidence*100:.0f}% t-Intervals — {covers}/{n_intervals} cover true μ ({covers/n_intervals*100:.0f}%)", fontweight="bold")
        ax.set_xlabel("Mean")
        ax.set_ylabel("Interval #")
        ax.legend()
        ax.grid(axis="x", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

elif topic == "Hypothesis Testing":
    st.header("Hypothesis Testing")

    test_type = st.radio("Test type", ["z-test (proportion)", "t-test (mean)"])

    if test_type == "z-test (proportion)":
        st.subheader("z-test for a Population Proportion")
        p0 = st.slider("Null hypothesis p₀", 0.01, 0.99, 0.5, 0.01, key="ht_p0")
        p_hat = st.slider("Sample proportion (p̂)", 0.01, 0.99, 0.6, 0.01, key="ht_phat")
        n = st.slider("Sample size (n)", 10, 1000, 100, key="ht_n")
        alt = st.selectbox("Alternative hypothesis", ["p ≠ p₀ (two-sided)", "p > p₀", "p < p₀"])

        se = np.sqrt(p0 * (1 - p0) / n)
        z_stat = (p_hat - p0) / se if se > 0 else 0

        if "two" in alt:
            p_value = 2 * norm.sf(abs(z_stat))
        elif ">" in alt:
            p_value = norm.sf(z_stat)
        else:
            p_value = norm.cdf(z_stat)

        col1, col2, col3 = st.columns(3)
        col1.metric("z-statistic", f"{z_stat:.4f}")
        col2.metric("p-value", f"{p_value:.4f}")
        col3.metric("Reject H₀ (α=0.05)?", "Yes" if p_value < 0.05 else "No")

        x_plot = np.linspace(-4, 4, 300)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x_plot, norm.pdf(x_plot), "k-", linewidth=2)
        if "two" in alt:
            ax.fill_between(x_plot, norm.pdf(x_plot), where=(x_plot <= -abs(z_stat)) | (x_plot >= abs(z_stat)), color="red", alpha=0.4, label="p-value region")
        elif ">" in alt:
            ax.fill_between(x_plot, norm.pdf(x_plot), where=(x_plot >= z_stat), color="red", alpha=0.4, label="p-value region")
        else:
            ax.fill_between(x_plot, norm.pdf(x_plot), where=(x_plot <= z_stat), color="red", alpha=0.4, label="p-value region")
        ax.axvline(z_stat, color="blue", linestyle="--", label=f"z = {z_stat:.2f}")
        ax.set_title("Null Distribution (Standard Normal)", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    else:
        st.subheader("t-test for a Population Mean")
        mu0 = st.slider("Null hypothesis μ₀", 0.0, 100.0, 50.0, 0.5, key="ht_mu0")
        x_bar = st.slider("Sample mean (x̄)", 0.0, 100.0, 55.0, 0.5, key="ht_xbar")
        s = st.slider("Sample std dev (s)", 0.1, 30.0, 10.0, 0.1, key="ht_s")
        n = st.slider("Sample size (n)", 5, 500, 30, key="ht_n2")
        alt = st.selectbox("Alternative hypothesis", ["μ ≠ μ₀ (two-sided)", "μ > μ₀", "μ < μ₀"])

        se = s / np.sqrt(n)
        t_stat = (x_bar - mu0) / se
        df = n - 1

        if "two" in alt:
            p_value = 2 * t.sf(abs(t_stat), df)
        elif ">" in alt:
            p_value = t.sf(t_stat, df)
        else:
            p_value = t.cdf(t_stat, df)

        col1, col2, col3 = st.columns(3)
        col1.metric("t-statistic", f"{t_stat:.4f}")
        col2.metric("p-value", f"{p_value:.4f}")
        col3.metric("Reject H₀ (α=0.05)?", "Yes" if p_value < 0.05 else "No")

        x_plot = np.linspace(-4, 4, 300)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x_plot, t.pdf(x_plot, df), "k-", linewidth=2)
        if "two" in alt:
            ax.fill_between(x_plot, t.pdf(x_plot, df), where=(x_plot <= -abs(t_stat)) | (x_plot >= abs(t_stat)), color="red", alpha=0.4, label="p-value region")
        elif ">" in alt:
            ax.fill_between(x_plot, t.pdf(x_plot, df), where=(x_plot >= t_stat), color="red", alpha=0.4, label="p-value region")
        else:
            ax.fill_between(x_plot, t.pdf(x_plot, df), where=(x_plot <= t_stat), color="red", alpha=0.4, label="p-value region")
        ax.axvline(t_stat, color="blue", linestyle="--", label=f"t = {t_stat:.2f}")
        ax.set_title(f"t-distribution (df={df})", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

elif topic == "Two-Sample Inference":
    st.header("Two-Sample Inference")

    st.subheader("Difference in Proportions")
    col1, col2 = st.columns(2)
    with col1:
        p1 = st.slider("Group 1 proportion (p̂₁)", 0.01, 0.99, 0.65, 0.01, key="ts_p1")
        n1 = st.slider("Group 1 size (n₁)", 10, 2000, 1000, 10, key="ts_n1")
    with col2:
        p2 = st.slider("Group 2 proportion (p̂₂)", 0.01, 0.99, 0.58, 0.01, key="ts_p2")
        n2 = st.slider("Group 2 size (n₂)", 10, 2000, 1000, 10, key="ts_n2")

    diff = p1 - p2
    p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se_diff = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z_stat = diff / se_diff if se_diff > 0 else 0
    p_value = 2 * norm.sf(abs(z_stat))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("p̂₁ − p̂₂", f"{diff:.4f}")
    col2.metric("z-statistic", f"{z_stat:.4f}")
    col3.metric("p-value", f"{p_value:.4f}")
    col4.metric("Significant (α=0.05)?", "Yes" if p_value < 0.05 else "No")

    st.subheader("Sampling Distribution of p̂₁ − p̂₂ under H₀")
    x_plot = np.linspace(-0.2, 0.2, 300)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_plot, norm.pdf(x_plot, 0, se_diff), "k-", linewidth=2)
    ax.axvline(diff, color="blue", linestyle="--", label=f"Observed diff = {diff:.4f}")
    ax.fill_between(x_plot, norm.pdf(x_plot, 0, se_diff),
                     where=(x_plot <= -abs(diff)) | (x_plot >= abs(diff)),
                     color="red", alpha=0.4, label="p-value region")
    ax.set_title("Null Distribution of Difference in Proportions", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

elif topic == "ANOVA":
    st.header("Analysis of Variance (ANOVA)")

    st.markdown("Enter data for up to 4 groups (comma-separated values):")
    default_groups = [
        "38, 40, 36, 42, 45, 39, 41",
        "30, 32, 35, 28, 34, 31, 33",
        "25, 27, 29, 26, 30, 28, 24",
    ]
    groups = []
    group_labels = []
    for i in range(4):
        default = default_groups[i] if i < len(default_groups) else ""
        val = st.text_input(f"Group {i+1}", default, key=f"anova_g{i}")
        if val.strip():
            try:
                groups.append(np.array([float(v) for v in val.split(",")]))
                group_labels.append(f"Group {i+1}")
            except ValueError:
                st.warning(f"Could not parse Group {i+1}")

    if len(groups) >= 2:
        k = len(groups)
        all_data = np.concatenate(groups)
        N = len(all_data)
        grand_mean = np.mean(all_data)

        group_means = [np.mean(g) for g in groups]
        group_sizes = [len(g) for g in groups]

        SSB = sum(n_i * (m - grand_mean) ** 2 for n_i, m in zip(group_sizes, group_means))
        SSW = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)
        SST = SSB + SSW

        dfB = k - 1
        dfW = N - k
        MSB = SSB / dfB
        MSW = SSW / dfW if dfW > 0 else 0
        F_stat = MSB / MSW if MSW > 0 else 0
        p_value = f.sf(F_stat, dfB, dfW) if dfW > 0 else 1.0

        st.markdown("### ANOVA Table")
        st.markdown(f"""
        | Source | SS | df | MS | F | p-value |
        |--------|------|------|------|------|---------|
        | Between | {SSB:.4f} | {dfB} | {MSB:.4f} | {F_stat:.4f} | {p_value:.4f} |
        | Within | {SSW:.4f} | {dfW} | {MSW:.4f} | | |
        | Total | {SST:.4f} | {N-1} | | | |
        """)

        st.markdown(f"**Decision (α=0.05):** {'Reject H₀ — at least one group mean differs' if p_value < 0.05 else 'Fail to reject H₀ — no significant difference'}")

        fig, ax = plt.subplots(figsize=(8, 4))
        x_f = np.linspace(0, max(F_stat * 2, 5), 300)
        ax.plot(x_f, f.pdf(x_f, dfB, dfW), "k-", linewidth=2)
        ax.fill_between(x_f, f.pdf(x_f, dfB, dfW), where=(x_f >= F_stat), color="red", alpha=0.4, label="p-value region")
        ax.axvline(F_stat, color="blue", linestyle="--", label=f"F = {F_stat:.2f}")
        ax.set_title(f"F-distribution (df₁={dfB}, df₂={dfW})", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        for i, g in enumerate(groups):
            ax.scatter([i] * len(g), g, alpha=0.7, s=50)
            ax.plot(i, np.mean(g), "D", color="black", markersize=10)
        ax.set_xticks(range(k))
        ax.set_xticklabels(group_labels)
        ax.set_ylabel("Value")
        ax.set_title("Group Data (diamonds = means)", fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Enter at least 2 groups to perform ANOVA.")

elif topic == "Categorical Data (Chi-Square)":
    st.header("Chi-Square Tests")

    tab1, tab2 = st.tabs(["Goodness of Fit", "Test of Independence"])

    with tab1:
        st.subheader("Chi-Square Goodness of Fit")
        st.markdown("Test whether observed frequencies match expected proportions.")

        n_cats = st.slider("Number of categories", 2, 7, 4, key="gof_cats")
        observed = []
        expected_pct = []
        cols = st.columns(n_cats)
        for i in range(n_cats):
            with cols[i]:
                o = st.number_input(f"Obs {i+1}", min_value=0, value=25 + i * 5, key=f"gof_o{i}")
                e = st.number_input(f"Exp% {i+1}", min_value=0.0, value=round(100 / n_cats, 1), key=f"gof_e{i}")
                observed.append(o)
                expected_pct.append(e)

        total_obs = sum(observed)
        total_pct = sum(expected_pct)
        if total_obs > 0 and total_pct > 0:
            expected = [total_obs * (ep / total_pct) for ep in expected_pct]
            chi2_stat = sum((o - e) ** 2 / e for o, e in zip(observed, expected) if e > 0)
            df = n_cats - 1
            p_value = chi2.sf(chi2_stat, df)

            col1, col2, col3 = st.columns(3)
            col1.metric("χ² statistic", f"{chi2_stat:.4f}")
            col2.metric("df", str(df))
            col3.metric("p-value", f"{p_value:.4f}")

            fig, ax = plt.subplots(figsize=(8, 4))
            x = np.arange(n_cats)
            ax.bar(x - 0.2, observed, 0.4, label="Observed", color="steelblue")
            ax.bar(x + 0.2, expected, 0.4, label="Expected", color="darkorange", alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels([f"Cat {i+1}" for i in range(n_cats)])
            ax.set_ylabel("Frequency")
            ax.set_title("Observed vs Expected", fontweight="bold")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)

    with tab2:
        st.subheader("Chi-Square Test of Independence")
        st.markdown("2×2 contingency table:")
        col1, col2 = st.columns(2)
        with col1:
            a = st.number_input("Cell (1,1)", min_value=0, value=30, key="ind_a")
            c = st.number_input("Cell (2,1)", min_value=0, value=20, key="ind_c")
        with col2:
            b_val = st.number_input("Cell (1,2)", min_value=0, value=10, key="ind_b")
            d = st.number_input("Cell (2,2)", min_value=0, value=40, key="ind_d")

        table = np.array([[a, b_val], [c, d]])
        n_total = table.sum()
        if n_total > 0:
            row_sums = table.sum(axis=1)
            col_sums = table.sum(axis=0)
            expected_table = np.outer(row_sums, col_sums) / n_total
            chi2_stat = np.sum((table - expected_table) ** 2 / expected_table)
            p_value = chi2.sf(chi2_stat, 1)

            col1, col2, col3 = st.columns(3)
            col1.metric("χ² statistic", f"{chi2_stat:.4f}")
            col2.metric("df", "1")
            col3.metric("p-value", f"{p_value:.4f}")

            st.markdown(f"**Decision (α=0.05):** {'Reject H₀ — variables are associated' if p_value < 0.05 else 'Fail to reject H₀ — no significant association'}")

elif topic == "Regression Analysis":
    st.header("Regression Analysis")

    st.subheader("Simple Linear Regression")
    st.markdown("Enter data or use the defaults (salary vs years of experience):")

    x_input = st.text_input("X values (comma-separated)", "1, 2, 3, 4, 5, 6, 7, 8, 9, 10", key="reg_x")
    y_input = st.text_input("Y values (comma-separated)", "25, 30, 35, 38, 42, 48, 52, 55, 60, 65", key="reg_y")

    try:
        x_data = np.array([float(v) for v in x_input.split(",")])
        y_data = np.array([float(v) for v in y_input.split(",")])

        if len(x_data) == len(y_data) and len(x_data) >= 3:
            n = len(x_data)
            x_bar = np.mean(x_data)
            y_bar = np.mean(y_data)

            b1 = np.sum((x_data - x_bar) * (y_data - y_bar)) / np.sum((x_data - x_bar) ** 2)
            b0 = y_bar - b1 * x_bar
            y_hat = b0 + b1 * x_data
            residuals = y_data - y_hat
            SSE = np.sum(residuals ** 2)
            SST_reg = np.sum((y_data - y_bar) ** 2)
            r_squared = 1 - SSE / SST_reg if SST_reg > 0 else 0
            se = np.sqrt(SSE / (n - 2)) if n > 2 else 0
            se_b1 = se / np.sqrt(np.sum((x_data - x_bar) ** 2)) if np.sum((x_data - x_bar) ** 2) > 0 else 0
            t_stat = b1 / se_b1 if se_b1 > 0 else 0
            p_value = 2 * t.sf(abs(t_stat), n - 2)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Intercept (b₀)", f"{b0:.4f}")
            col2.metric("Slope (b₁)", f"{b1:.4f}")
            col3.metric("R²", f"{r_squared:.4f}")
            col4.metric("p-value (slope)", f"{p_value:.6f}")

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(x_data, y_data, color="steelblue", s=60, zorder=5)
                x_line = np.linspace(x_data.min() - 0.5, x_data.max() + 0.5, 100)
                ax.plot(x_line, b0 + b1 * x_line, "r-", linewidth=2, label=f"ŷ = {b0:.2f} + {b1:.2f}x")
                ax.set_title("Scatter Plot with Regression Line", fontweight="bold")
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.legend()
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)

            with col2:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(y_hat, residuals, color="darkorange", s=60)
                ax.axhline(0, color="black", linestyle="--")
                ax.set_title("Residual Plot", fontweight="bold")
                ax.set_xlabel("Fitted Values")
                ax.set_ylabel("Residuals")
                ax.grid(alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)

            st.subheader("Confidence Interval for Slope")
            conf_level = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01, key="reg_conf")
            t_crit = t.ppf(1 - (1 - conf_level) / 2, n - 2)
            ci_lo = b1 - t_crit * se_b1
            ci_hi = b1 + t_crit * se_b1
            st.markdown(f"**{conf_level*100:.0f}% CI for slope:** [{ci_lo:.4f}, {ci_hi:.4f}]")
        else:
            st.warning("X and Y must have the same length (≥ 3 points).")
    except ValueError:
        st.warning("Could not parse input values.")

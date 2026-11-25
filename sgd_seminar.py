from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)


def fixed_gradient_descent(
    function,
    gradient,
    theta0,
    learning_rate,
    max_iter=500,
    tolerance=1e-8,
):
    theta = np.array(theta0, dtype=float, copy=True)
    values, norms, steps = [], [], []
    for _ in range(max_iter):
        value = function(theta)
        grad = gradient(theta)
        values.append(value)
        norms.append(np.linalg.norm(grad))
        if np.linalg.norm(grad) <= tolerance:
            break
        theta -= learning_rate * grad
        steps.append(learning_rate)
    values.append(function(theta))
    return theta, values, norms, steps


def backtracking_line_search(
    theta, value, gradient, direction, function, alpha=1.0,
    c1=1e-4, shrink=0.5,
):
    for _ in range(60):
        candidate = theta + alpha * direction
        candidate_value = function(candidate)
        if (
            np.isfinite(candidate_value)
            and candidate_value <= value + c1 * alpha * gradient.dot(direction)
        ):
            return alpha
        alpha *= shrink
    return alpha


def barzilai_borwein_quadratic(seed=42):
    rng = np.random.default_rng(seed)
    a = np.diag([1.0, 3.0, 10.0, 30.0])
    target = rng.normal(size=4)

    def function(theta):
        delta = theta - target
        return 0.5 * delta.dot(a).dot(delta)

    def gradient(theta):
        return a @ (theta - target)

    theta = rng.normal(size=4)
    bb_values = [function(theta)]
    bb_steps = []
    previous_step = np.array(theta, copy=True)
    previous_grad = gradient(theta)
    alpha = 0.1

    for _ in range(100):
        grad = gradient(theta)
        if np.linalg.norm(grad) < 1e-10:
            break
        direction = -grad
        alpha = backtracking_line_search(
            theta, function(theta), grad, direction, function, alpha
        )
        theta_new = theta + alpha * direction
        new_grad = gradient(theta_new)
        s = theta_new - theta
        y = new_grad - grad
        sy = s.dot(y)
        ss = s.dot(s)
        if sy > 1e-12:
            bb_alpha = ss / sy
            bb_alpha = np.clip(bb_alpha, 1e-4, 1.0)
        else:
            bb_alpha = alpha
        theta = theta_new
        bb_steps.append(alpha)
        bb_values.append(function(theta))
        previous_step = theta_new
        previous_grad = new_grad
        alpha = bb_alpha

    fixed_theta, fixed_values, _, fixed_steps = fixed_gradient_descent(
        function, gradient, np.array(theta + 0.0), 0.03, max_iter=100
    )

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    left.semilogy(bb_values, label="BB + backtracking")
    left.semilogy(fixed_values, label="Fixed learning rate")
    left.set(title="Quadratic objective", xlabel="Iteration", ylabel="Function value")
    left.legend()
    right.plot(bb_steps, label="BB + backtracking step")
    right.plot(fixed_steps[:len(bb_steps)], label="Fixed step")
    right.set(title="Step size", xlabel="Iteration", ylabel="Learning rate")
    right.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "01_bb_backtracking.png", dpi=180)
    plt.close(fig)
    return bb_values, bb_steps


def make_logistic_problem(seed=7, n=300, d=12):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    true_theta = rng.normal(size=d)
    logits = np.clip(x @ true_theta, -40.0, 40.0)
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
    y = (probabilities > rng.random(n)).astype(float)

    def loss(theta):
        z = x @ theta
        return np.mean(np.logaddexp(0.0, z) - y * z)

    def full_gradient(theta):
        probabilities = 1 / (1 + np.exp(-np.clip(x @ theta, -40, 40)))
        return x.T @ (probabilities - y) / n

    def sample_gradient(theta, i):
        z = x[i] @ theta
        probability = 1 / (1 + np.exp(-np.clip(z, -40, 40)))
        return (probability - y[i]) * x[i]

    return x, y, loss, full_gradient, sample_gradient


def sag_experiment(seed=7):
    _, _, loss, full_gradient, sample_gradient = make_logistic_problem(seed=seed)
    n = 300

    def run_sag(adaptive=False):
        theta = np.zeros(12)
        stored = np.zeros((n, 12))
        average = np.zeros(12)
        values = []
        for iteration in range(1500):
            if adaptive:
                raw = np.maximum(np.linalg.norm(stored, axis=1), 1e-12)
                probabilities = raw / raw.sum()
                i = np.random.default_rng(seed + iteration).choice(n, p=probabilities)
            else:
                i = np.random.default_rng(seed + iteration).integers(n)
            old = stored[i]
            new = sample_gradient(theta, i)
            average += (new - old) / n
            stored[i] = new
            theta -= 2.0 / n * average
            if iteration % 5 == 0:
                values.append(loss(theta))
        return values

    def run_sgd():
        theta = np.zeros(12)
        values = []
        for iteration in range(1500):
            i = np.random.default_rng(seed + iteration).integers(n)
            grad = sample_gradient(theta, i)
            learning_rate = 0.5 / np.sqrt(1 + iteration)
            theta -= learning_rate * grad
            if iteration % 5 == 0:
                values.append(loss(theta))
        return values

    uniform = run_sag(adaptive=False)
    adaptive = run_sag(adaptive=True)
    sgd = run_sgd()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(uniform, label="Uniform SAG")
    ax.semilogy(adaptive, label="Adaptive SAG")
    ax.semilogy(sgd, label="SGD")
    ax.set(title="SAG and stochastic methods", xlabel="Logged iteration (x5)", ylabel="Logistic loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "02_sag_adaptive_sampling.png", dpi=180)
    plt.close(fig)


def solve_trust_region(grad, hessian, radius):
    try:
        newton = np.linalg.solve(hessian, -grad)
    except np.linalg.LinAlgError:
        newton = -grad
    if np.linalg.norm(newton) <= radius:
        return newton
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    diagonal = np.maximum(eigenvalues, 1e-8)
    inverse_grad = -eigenvectors.T @ grad
    direction = eigenvectors @ (inverse_grad / diagonal)
    norm = np.linalg.norm(direction)
    if norm > 1e-14:
        direction *= min(1.0, radius / norm)
    return direction


def trust_region_experiment(seed=11):
    rng = np.random.default_rng(seed)

    def full_newton_problem(theta):
        x, y = theta
        return 2 * x**2 - 1.5 * x * y + y**2 + 0.3 * x**3 - 0.15 * y**4

    def full_newton_grad(theta):
        x, y = theta
        return np.array([4*x - 1.5*y + 0.9*x*x, -1.5*x + 2*y - 0.6*y**3])

    def full_newton_hessian(theta):
        x, y = theta
        return np.array([[4 + 1.8*x, -1.5], [-1.5, 2 - 1.8*y*y]])

    theta_trust = rng.normal(size=2)
    theta_full = theta_trust.copy()
    radius = 0.5
    trust_values = [full_newton_problem(theta_trust)]
    trust_radii = [radius]
    full_values = [full_newton_problem(theta_full)]
    trust_trajectory = [theta_trust.copy()]

    for _ in range(60):
        grad = full_newton_grad(theta_trust)
        hessian = full_newton_hessian(theta_trust)
        step = solve_trust_region(grad, hessian, radius)
        candidate = theta_trust + step
        actual = full_newton_problem(theta_trust) - full_newton_problem(candidate)
        predicted = -(grad.dot(step) + 0.5 * step.dot(hessian).dot(step))
        ratio = actual / predicted if predicted > 1e-12 else -1.0
        if ratio > 0.1:
            theta_trust = candidate
        if ratio > 0.75:
            radius = min(2.0 * radius, 2.0)
        elif ratio < 0.25:
            radius *= 0.25
        trust_values.append(full_newton_problem(theta_trust))
        trust_radii.append(radius)
        trust_trajectory.append(theta_trust.copy())
        if np.linalg.norm(full_newton_grad(theta_trust)) < 1e-8:
            break

    for _ in range(60):
        grad = full_newton_grad(theta_full)
        hessian = full_newton_hessian(theta_full)
        try:
            step = np.linalg.solve(hessian, -grad)
        except np.linalg.LinAlgError:
            break
        theta_full += step
        full_values.append(full_newton_problem(theta_full))
        if np.linalg.norm(full_newton_grad(theta_full)) < 1e-8:
            break

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    left.plot(trust_values, label="Trust region Newton")
    left.plot(full_values, label="Full Newton")
    left.set(title="Newton comparison", xlabel="Iteration", ylabel="Function value")
    left.legend()
    right.plot(trust_radii, label="Trust region radius")
    right.set(title="Trust region radius", xlabel="Iteration", ylabel="Radius")
    right.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "03_newton_trust_region.png", dpi=180)
    plt.close(fig)
    return np.array(trust_trajectory)


def damped_curvature(s, y, hessian, threshold=1e-8):
    sy = s.dot(y)
    by = hessian @ y
    yby = y.dot(by)
    if sy >= threshold * yby:
        return s, y
    theta = (1.0 - threshold) * sy + threshold * yby
    if theta <= 0:
        theta = threshold
    return s, theta * s / yby + (1 - theta) * y


def bfgs_experiment(seed=13):
    rng = np.random.default_rng(seed)
    a = np.diag([1.0, 2.0, 5.0, 12.0, 25.0])
    target = rng.normal(size=5)

    def function(theta):
        delta = theta - target
        return 0.5 * delta.dot(a).dot(delta)

    def gradient(theta):
        return a @ (theta - target)

    theta = rng.normal(size=5)
    inverse_hessian = np.eye(5)
    bfgs_values = [function(theta)]
    trajectories = [theta.copy()]

    for _ in range(100):
        grad = gradient(theta)
        if np.linalg.norm(grad) < 1e-10:
            break
        direction = -inverse_hessian @ grad
        alpha = backtracking_line_search(
            theta, function(theta), grad, direction, function, alpha=1.0
        )
        theta_new = theta + alpha * direction
        s = theta_new - theta
        y = gradient(theta_new) - grad
        sy = s.dot(y)
        ss = s.dot(s)
        if sy <= 1e-10 * ss:
            theta_y = 0.2 * ss / max(ss - sy, 1e-12)
            y = theta_y * y + (1 - theta_y) * s
            sy = s.dot(y)
        if sy > 1e-12:
            rho = 1 / sy
            identity = np.eye(5)
            inverse_hessian = (
                (identity - rho * np.outer(s, y)) @ inverse_hessian
                @ (identity - rho * np.outer(y, s))
                + rho * np.outer(s, s)
            )
        else:
            inverse_hessian = np.eye(5)
        theta = theta_new
        bfgs_values.append(function(theta))
        trajectories.append(theta.copy())

    gd_theta, gd_values, _, _ = fixed_gradient_descent(
        function, gradient, trajectories[0], 0.025, max_iter=100
    )
    newton_theta = target.copy()
    newton_values = [function(newton_theta)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(bfgs_values, label="Damped/curvature BFGS")
    ax.semilogy(gd_values, label="Fixed gradient descent")
    ax.semilogy(newton_values, label="Newton (one step)")
    ax.set(title="BFGS, gradient descent and Newton", xlabel="Iteration", ylabel="Quadratic value")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "04_bfgs_damping.png", dpi=180)
    plt.close(fig)
    return bfgs_values


def lbfgs_direction(grad, history, m):
    q = grad.copy()
    recent = history[-m:]
    alpha_values = np.zeros(len(recent))
    for i in range(len(recent) - 1, -1, -1):
        s, y, rho = recent[i]
        alpha = rho * s.dot(q)
        alpha_values[i] = alpha
        q -= alpha * y
    if history:
        s, y, _ = history[-1]
        gamma = s.dot(y) / y.dot(y)
    else:
        gamma = 1.0
    r = gamma * q
    for i in range(len(recent)):
        s, y, rho = recent[i]
        alpha = alpha_values[i]
        beta = rho * y.dot(r)
        r += s * (alpha - beta)
    return -r


def lbfgs_experiment(seed=17, d=80):
    rng = np.random.default_rng(seed)
    diagonal = np.linspace(0.5, 50.0, d)
    target = rng.normal(size=d)
    hessian = np.diag(diagonal)

    def function(theta):
        delta = theta - target
        return 0.5 * delta.dot(hessian).dot(delta)

    def gradient(theta):
        return hessian @ (theta - target)

    initial = rng.normal(size=d)

    def run(memory):
        theta = initial.copy()
        history = []
        values = [function(theta)]
        memory_bytes = 0
        for _ in range(300):
            grad = gradient(theta)
            if not np.all(np.isfinite(grad)):
                break
            if np.linalg.norm(grad) < 1e-8:
                break
            direction = lbfgs_direction(grad, history, memory)
            alpha = backtracking_line_search(
                theta, function(theta), grad, direction, function, alpha=1.0
            )
            theta_new = theta + alpha * direction
            s = theta_new - theta
            y = gradient(theta_new) - grad
            sy = s.dot(y)
            if sy > 1e-12:
                history.append((s, y, 1 / sy))
            values.append(function(theta_new))
            theta = theta_new
            memory_bytes = 2 * len(history[-memory:]) * d * np.dtype(float).itemsize
        return values, len(values) - 1, memory_bytes

    results = {}
    for memory in (3, 10, 30):
        results[memory] = run(memory)

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.5))
    for memory, (values, iterations, _) in results.items():
        left.semilogy(values, label=f"m={memory}, {iterations} iterations")
    left.set(title="L-BFGS convergence", xlabel="Iteration", ylabel="Quadratic value")
    left.legend()
    memories = list(results)
    bytes_used = [results[m][2] for m in memories]
    right.bar([str(m) for m in memories], np.array(bytes_used) / 1024)
    right.set(title="Stored S/Y memory", xlabel="m", ylabel="KiB")
    fig.tight_layout()
    fig.savefig(PLOTS / "05_lbfgs_memory.png", dpi=180)
    plt.close(fig)
    return results


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    print("Running BB + backtracking...")
    barzilai_borwein_quadratic()
    print("Running SAG experiments...")
    sag_experiment()
    print("Running Newton trust region...")
    trust_region_experiment()
    print("Running BFGS...")
    bfgs_experiment()
    print("Running L-BFGS...")
    lbfgs_experiment()
    print(f"Plots saved in: {PLOTS}")


if __name__ == "__main__":
    main()

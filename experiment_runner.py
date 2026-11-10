import numpy as np
from concurrent.futures import ProcessPoolExecutor
import copy
import os

def run_realization(
    signal_generator,
    noise_generator,
    unknown_system,
    algorithms,
    n_samples,
):
    filter_length = len(unknown_system)

    # Preenche os regressores antes do início da realização.
    initial_regressor = signal_generator.generate(filter_length)
    x_regressor = initial_regressor.copy()

    for algorithm in algorithms.values():
        algorithm.reset()
        algorithm.initialize_regressor(initial_regressor.copy())

    results = {
        name: {
            "MSD": np.zeros(n_samples),
            "MSE": np.zeros(n_samples),
            "NPEA": np.zeros(n_samples),
        }
        for name in algorithms
    }

    adaptation_counters = {name: 0 for name in algorithms}
    adaptation_intervals = {}

    for name, algorithm in algorithms.items():
        interval = algorithm.get_adaptation_interval()
        if not isinstance(interval, (int, np.integer)):
            raise ValueError("O intervalo de adaptação deve ser inteiro.")
        adaptation_intervals[name] = interval

    for n in range(n_samples):
        x = signal_generator.generate_sample()

        x_regressor[1:] = x_regressor[:-1]
        x_regressor[0] = x

        noise = noise_generator.generate_sample()
        d = unknown_system @ x_regressor + noise

        for name, algorithm in algorithms.items():
            e = algorithm.step(x, d)
            w = algorithm.get_time_coefficients()

            results[name]["MSD"][n] = np.sum((w - unknown_system) ** 2)
            results[name]["MSE"][n] = e ** 2
            results[name]["NPEA"][n] = algorithm.get_npea()

            adaptation_counters[name] += 1
            if adaptation_counters[name] == adaptation_intervals[name]:
                algorithm.AI()
                adaptation_counters[name] = 0

                interval = algorithm.get_adaptation_interval()
                if not isinstance(interval, (int, np.integer)):
                    raise ValueError("O intervalo de adaptação deve ser inteiro.")
                adaptation_intervals[name] = interval

    return results

def run_monte_carlo(
    signal_generator,
    noise_generator,
    unknown_system,
    algorithms,
    n_samples,
    n_realizations,
):
    accumulated_results = {
        name: {
            "MSD": np.zeros(n_samples),
            "MSE": np.zeros(n_samples),
            "NPEA": np.zeros(n_samples),
        }
        for name in algorithms
    }

    for _ in range(n_realizations):
        results = run_realization(
            signal_generator,
            noise_generator,
            unknown_system,
            algorithms,
            n_samples,
        )

        for name in algorithms:
            for metric in accumulated_results[name]:
                accumulated_results[name][metric] += results[name][metric]

    for name in algorithms:
        for metric in accumulated_results[name]:
            accumulated_results[name][metric] /= n_realizations

    return accumulated_results

def _parallel_worker(args):
    (
        signal_generator,
        noise_generator,
        unknown_system,
        algorithms,
        n_samples,
        n_realizations,
    ) = args

    signal_generator = copy.deepcopy(signal_generator)
    noise_generator = copy.deepcopy(noise_generator)
    algorithms = copy.deepcopy(algorithms)

    # Inicializa geradores aleatórios independentes neste processo.
    signal_generator.rng = np.random.default_rng()
    noise_generator.rng = np.random.default_rng()

    accumulated_results = {
        name: {
            "MSD": np.zeros(n_samples),
            "MSE": np.zeros(n_samples),
            "NPEA": np.zeros(n_samples),
        }
        for name in algorithms
    }

    for _ in range(n_realizations):
        results = run_realization(
            signal_generator,
            noise_generator,
            unknown_system,
            algorithms,
            n_samples,
        )

        for name in algorithms:
            for metric in accumulated_results[name]:
                accumulated_results[name][metric] += results[name][metric]

    return accumulated_results

def run_parallel_monte_carlo(
    signal_generator,
    noise_generator,
    unknown_system,
    algorithms,
    n_samples,
    n_realizations,
    n_workers=None,
):
    if n_workers is None:
        n_workers = os.cpu_count()

    n_workers = min(n_workers, n_realizations)

    base = n_realizations // n_workers
    remainder = n_realizations % n_workers

    tasks = []

    for worker in range(n_workers):
        worker_realizations = base + (worker < remainder)

        tasks.append((
            signal_generator,
            noise_generator,
            unknown_system,
            algorithms,
            n_samples,
            worker_realizations,
        ))

    accumulated_results = {
        name: {
            "MSD": np.zeros(n_samples),
            "MSE": np.zeros(n_samples),
            "NPEA": np.zeros(n_samples),
        }
        for name in algorithms
    }

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        worker_results = executor.map(_parallel_worker, tasks)

        for results in worker_results:
            for name in algorithms:
                for metric in accumulated_results[name]:
                    accumulated_results[name][metric] += results[name][metric]

    for name in algorithms:
        for metric in accumulated_results[name]:
            accumulated_results[name][metric] /= n_realizations

    return accumulated_results
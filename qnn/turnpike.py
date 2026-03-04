import random
from itertools import product
from typing import Iterable

import numpy as np
from tqdm import tqdm


def difference_set(s: Iterable[int | float]) -> set:
    return {x - y for x in s for y in s}


def find_value_of_k(s: Iterable[int]) -> int:
    """
    Find the value of K for the turnpike encoding, see Paper.
    """
    delta_s = difference_set(s)
    for K in range(max(delta_s) + 1):
        if K not in delta_s:
            return K - 1
    return max(delta_s)


def candidate_set_generator(d: int) -> set[int]:
    """
    Generate the candidate set for the turnpike encoding, see Paper.
    Args:
        d: The dimension (the size) of the sets.

    Returns: A set of integers.

    """
    for tup in product(range(1, d * (d - 1) // 2 + 1), repeat=d - 1):
        new_candidate = {0}
        value = 0
        for entry in tup:
            new_candidate.add(value + entry)
            value += entry
        yield new_candidate


def compute_solution_relaxed_turnpike(d: int) -> list[set[int]]:
    """
    Compute all solutions for the relaxed turnpike problem of the given dimension
    Args:
        d: The dimension (the size) of the sets.

    Returns: Al ist of all solutions.

    """
    k_max = -np.inf
    s_max_list = []
    for s in (C := tqdm(candidate_set_generator(d), total=(d * (d - 1) // 2) ** (d - 1))):
        k = find_value_of_k(s)
        if k > k_max:
            k_max = k
            s_max_list = [s]
            C.set_description(f"{k_max}, {s}")
        elif k == k_max:
            s_max_list.append(s)
            C.set_description(f"Number of solutions: {len(s_max_list)}, {k_max=}, {s=}")
    return s_max_list


def _set_plus_number(S: set[int], k: int) -> set[int]:
    return {s + k for s in S}


def _negative_set(S: set[int]) -> set[int]:
    return {-s for s in S}


def _random_split(S: set[int]) -> tuple[set[int], int]:
    s = random.choice(list(S))
    A = S.copy()
    A.remove(s)
    return A, s


def _reflected_set(S: set[int]) -> set[int]:
    return _set_plus_number(_negative_set(S), max(S))


def _optimize(s: set[int]) -> set[int]:
    """
    Helper function to compute the solution to the turnpike problem using a greedy search algorithm. A single optimization
    step is performed on the given set s.
    Args:
        s: The set to optimize.

    Returns: The optimized set.

    """
    ka = find_value_of_k(s)
    k = ka + 1
    candidates = _set_plus_number(s, -k) | _set_plus_number(s, k)
    max_k = 0
    max_s = set()
    for s in candidates:
        new_s = s | {s}
        k_new_s = find_value_of_k(new_s)
        if k_new_s > max_k:
            max_k = k_new_s
            max_s = new_s
    max_s = _set_plus_number(max_s, -min(max_s))
    return max_s


def _find_solutions(initial_set: set[int], max_iter=100, verbose: bool = False) -> set[int]:
    """
    Helper function to find solutions to the turnpike problem using a greedy search algorithm. Optimizes a
    single initial set.
    Args:
        initial_set: The initial set to optimize.
        max_iter: The maximum number of iterations.
        verbose: Print the progress.

    Returns:

    """
    S = initial_set.copy()
    for round in range(max_iter):
        if verbose:
            print(f"Round {round}: {S=}, K = {find_value_of_k(S)}")
        S, s = _random_split(S)
        S = _optimize(S)
    return S


def greedy_search_turnpike_solutions(max_iter: int = 100,
                                     initial_sets: Iterable[set[int]] = None) -> set[frozenset[int]]:
    """
    Search for solutions to the turnpike problem using a greedy search algorithm. The algorithm starts with a set
    of starting sets and iteratively optimizes each starting set by replacing one element with another element that
    maximizes the value of K. The algorithm stops after a given number of iterations.

    Example:
    initial_sets = [{0, 1, 2, 3},
                    {0, 1, 2, 4},
                    {0, 1, 3, 4},
                    {0, 2, 3, 4},
                    {1, 2, 3, 4}] # so d = 4
    solutions = greedy_search_turnpike_solutions(max_iter=100, initial_sets=initial_sets)

    Args:
        max_iter: The maximum number of iterations per initial set.
        initial_sets: A list of initial sets. The length of an initial set is the dimension of the turnpike problem.

    Returns: A set of solutions with the maximal value of K found.
    """
    max_k = 0
    solutions = set()
    initial_sets = list(initial_sets) if initial_sets is not None else []
    number_initial_sets = len(initial_sets)
    for i, init_set in (iterator := tqdm(enumerate(initial_sets), total=number_initial_sets)):
        solution = _find_solutions(init_set, max_iter, verbose=False)
        k = find_value_of_k(solution)
        if k > max_k:
            max_k = k
            solutions = {frozenset(solution),
                         frozenset(_reflected_set(solution))
                         }
            iterator.set_description(f"Max K: {k}, number of solutions: {len(solutions)}")
        elif k == max_k:
            solutions.add(frozenset(solution))
            solutions.add(frozenset(_reflected_set(solution)))
            iterator.set_description(f"Max K: {k}, number of solutions: {len(solutions)}")
    return solutions

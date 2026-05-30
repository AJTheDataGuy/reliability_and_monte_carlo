"""Statistical Functions"""

# 3rd Party Imports
import numpy as np


def simple_a_b_test_monte_beta(
    beta_params_1: tuple = (39, 121),
    beta_params_2: tuple = (53, 107),
    num_simulations: int = 100_000,
)->dict:
    """
    Simple A/B test using beta distributions
    and monte carlo simulation

    Adapted from Chapter 15 of
    Bayesian Statistics the Fun Way

    :param beta_params_1: Beta distribution parameters for the first distribution
                            given as a tuple of (alpha,beta)
    :param beta_params_2: Beta distribution parameters for the second distribution
                            given as a tuple of (alpha,beta)
    :param num_simulations: Number of Monte Carlo simulations to perform
    :returns: A dictionary comparing the two distributions with multiple options
                for further analysis. The dictionary contains:
                    1. % trials where Distribution 1 was better (as a float - such as 0.3)
                    2. % trials where Distribution 2 was better (as a float - such as 0.7)
                    3. An array of Dist_1 / Dist_2. Can be used to create a histogram showing
                        performance of Dist_1 relative to Dist_2
                        (for example showing on average that Dist_1 was 1.3 times better than Dist_2)
                    4. An array of Dist_2 / Dist_1. Can be used to create a histogram showing
                        performance of Dist_2 relative to Dist_1
                        (for example showing on average that Dist_2 was 1.5 times better than Dist_1)
    """
    # Step 1: Unpack the tuples
    beta_1_a, beta_1_b = beta_params_1
    beta_2_a, beta_2_b = beta_params_2

    # Step 2: Generate Monte Carlo Simulation
    monte_beta_1_array = np.random.beta(a=beta_1_a, b=beta_1_b, size=num_simulations)
    monte_beta_2_array = np.random.beta(a=beta_2_a, b=beta_2_b, size=num_simulations)

    # Step 3: Calculate overall how much better each simulation did
    percent_dist_1_better_float = (
        np.sum(monte_beta_1_array > monte_beta_2_array) / num_simulations
    )

    percent_dist_2_better_float = (
        np.sum(monte_beta_2_array > monte_beta_1_array) / num_simulations
    )

    # Step 4: Calculate relative individual performance of each simulation
    # Intended to be used with a histogram
    a_relative_to_b_array = monte_beta_1_array / monte_beta_2_array
    b_relative_to_a_array = monte_beta_2_array / monte_beta_1_array

    return {
        "percent_dist_1_better": percent_dist_1_better_float,
        "percent_dist_2_better": percent_dist_2_better_float,
        "array_a_divided_by_b": a_relative_to_b_array,
        "array_b_divided_by_a": b_relative_to_a_array,
    }

import pandas as pd
import numpy as np
from itertools import combinations
from scipy.stats import fisher_exact, chi2_contingency
from collections import Counter


def extract_subsequences(trajectory, max_len=7, min_len=3):
    """Extract all contiguous and non-contiguous subsequences up to max_len."""
    steps = trajectory if isinstance(trajectory, list) else trajectory.split()
    subseqs = set()

    # Contiguous n-grams (most interpretable, start here)
    for n in range(min_len, min(max_len + 1, len(steps) + 1)):
        for i in range(len(steps) - n + 1):
            subseqs.add(tuple(steps[i:i + n]))

    # Optional: skip-grams (non-contiguous but order-preserving)
    # for n in range(2, min(max_len + 1, len(steps) + 1)):
    #     for combo in combinations(range(len(steps)), n):
    #         subseqs.add(tuple(steps[j] for j in combo))

    return subseqs


def remove_redundant_patterns(df, effect_col='effect', tolerance=0.03):
    """
    If a longer pattern has a similar effect to a shorter sub-pattern,
    keep only the shorter one (Occam's razor).
    """
    df = df[df['significant']].copy()
    df['pattern_tuple'] = df['pattern'].apply(lambda x: tuple(x.split(' → ')))

    to_drop = set()
    patterns = df.sort_values('length').to_dict('records')

    for i, long_p in enumerate(patterns):
        if long_p['pattern'] in to_drop:
            continue
        long_t = long_p['pattern_tuple']

        for short_p in patterns:
            if short_p['length'] >= long_p['length']:
                continue
            short_t = short_p['pattern_tuple']

            # Check if short is a contiguous sub-pattern of long
            short_str = ' → '.join(short_t)
            long_str = ' → '.join(long_t)
            if short_str in long_str:
                # If effect is similar, the longer pattern is redundant
                if abs(long_p['effect'] - short_p['effect']) < tolerance:
                    to_drop.add(long_p['pattern'])
                    break

    return df[~df['pattern'].isin(to_drop)].drop(columns=['pattern_tuple'])


def find_predictive_patterns(
        trajectories: list[list[str]],
        labels: list[int],  # 1=success, 0=failure
        min_pattern_len: int = 3,
        max_pattern_len: int = 7,
        min_occurrences: int = 10,
        p_value_threshold: float = 0.05,
):
    """
    Find short patterns that are statistically predictive of success/failure.
    """
    n = len(trajectories)
    base_rate = np.mean(labels)

    # Step 1: Extract all short subsequences per trajectory
    pattern_presence = {}  # pattern -> list of bools (present in each trajectory)

    for i, traj in enumerate(trajectories):
        subseqs = extract_subsequences(traj, max_len=max_pattern_len, min_len=min_pattern_len)
        for s in subseqs:
            if s not in pattern_presence:
                pattern_presence[s] = np.zeros(n, dtype=bool)
            pattern_presence[s][i] = True

    labels_arr = np.array(labels)

    # Step 2: For each pattern, compute stats
    results = []
    for pattern, present in pattern_presence.items():
        count = present.sum()
        if count < min_occurrences or count > n - min_occurrences:
            continue  # skip too rare or too common

        # Success rates
        sr_with = labels_arr[present].mean()
        sr_without = labels_arr[~present].mean()

        # 2x2 contingency table for Fisher's exact test
        #                success  failure
        # pattern present    a       b
        # pattern absent     c       d
        a = (labels_arr[present] == 1).sum()
        b = (labels_arr[present] == 0).sum()
        c = (labels_arr[~present] == 1).sum()
        d = (labels_arr[~present] == 0).sum()

        _, p_value = fisher_exact([[a, b], [c, d]])

        # Effect size: difference in success rate
        effect = sr_with - sr_without

        # Lift
        lift = sr_with / max(base_rate, 1e-9)

        results.append({
            'pattern': ' → '.join(pattern),
            'length': len(pattern),
            'occurrences': int(count),
            'success_rate_with': round(sr_with, 3),
            'success_rate_without': round(sr_without, 3),
            'effect': round(effect, 3),
            'lift': round(lift, 2),
            'p_value': p_value,
        })

    df = pd.DataFrame(results)

    # Step 3: Multiple testing correction (Benjamini-Hochberg)
    if len(df) > 0:
        df = df.sort_values('p_value')
        df['rank'] = range(1, len(df) + 1)
        df['bh_threshold'] = (df['rank'] / len(df)) * p_value_threshold
        df['significant'] = df['p_value'] <= df['bh_threshold']
        df = df.drop(columns=['rank', 'bh_threshold'])

    significant = df[df['significant']].copy()
    minimal_patterns = remove_redundant_patterns(significant)
    base_success_rate = np.mean(labels)
    minimal_patterns.loc[:,"predictive_score"] = minimal_patterns.apply(lambda p: predictive_score(p, base_success_rate), axis=1)
    # Return the full dataframe for more flexible display
    return minimal_patterns


def predictive_score(row, base_success_rate):
    """
    Single number: how much does this pattern shift the
    probability vs. the baseline, as a relative change.

    +60% means "60% more likely to succeed than average"
    -45% means "45% more likely to fail than average"
    """
    sr_with = row['success_rate_with']

    if sr_with >= base_success_rate:
        # Success-predictive: relative increase in success rate
        score = (sr_with - base_success_rate) / max(base_success_rate, 1e-9)
    else:
        # Failure-predictive: relative increase in failure rate
        fr_with = 1 - sr_with
        fr_base = 1 - base_success_rate
        score = -((fr_with - fr_base) / max(fr_base, 1e-9))

    return round(score * 100, 1)

if __name__ == "__main__":
    import os, json
    trajs = []
    labels =[]
    traj_dir = "/Users/lilache/PycharmProjects/CLEAR/src/clear_eval/agentic/output/new_ui/orig_cuga/traj_data"
    for f in os.listdir(traj_dir):
        if f.endswith(".csv"):
            df = pd.read_csv(os.path.join(traj_dir,f))
            traj = list(df["agent_name"])
            trajs.append(traj)
            labels.append(json.loads((list(df["meta_data"])[0]))["trajectory_score"])
    res_df = find_predictive_patterns(trajs, labels, min_pattern_len=3)

    print("=== FAILURE SIGNALS ===")
    print(res_df[res_df['significant'] & (res_df['effect'] < 0)]
          .sort_values('effect')
          .head(10)
          .to_string(index=False))

    # Top success-predictive patterns
    print("\n=== SUCCESS SIGNALS ===")
    print(res_df[res_df['significant'] & (res_df['effect'] > 0)]
          .sort_values('effect', ascending=False)
          .head(10)
          .to_string(index=False))

import argparse
import numpy as np
import pandas as pd

from uehling import (
    V_uehling_v,
    V_uehling_short,
    V_uehling_long,
)


def run_validation(data_path: str):
    # Load Mathematica ground truth
    df_math = pd.read_csv(data_path)

    r_vals = df_math["r_over_compton"].values
    V_math = df_math["V_Uehling_natural_units"].values

    # Compute Python values
    V_py = V_uehling_v(r_vals)

    # Relative error at each point
    rel_error = np.abs((V_py - V_math) / V_math)

    print("=== Mathematica Comparison ===")
    print(f"Max relative error : {rel_error.max():.2e}")
    print(f"Mean relative error: {rel_error.mean():.2e}")

    # Asymptotic checks
    r_short = 0.01
    r_long = 5.0

    print(f"\n=== Short-range check (r={r_short}) ===")
    print(f"Numerical    : {V_uehling_v(r_short):.8e}")
    print(f"Short approx : {V_uehling_short(r_short):.8e}")

    print(f"\n=== Long-range check (r={r_long}) ===")
    print(f"Numerical    : {V_uehling_v(r_long):.8e}")
    print(f"Long approx  : {V_uehling_long(r_long):.8e}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate Python Uehling potential implementation against Mathematica data."
    )

    parser.add_argument(
        "--data",
        default="../data/uehling_data.csv",
        help="Path to CSV file containing Mathematica reference data.",
    )

    args = parser.parse_args()

    run_validation(args.data)


if __name__ == "__main__":
    main()
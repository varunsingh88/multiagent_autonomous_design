import matplotlib.pyplot as plt
import numpy as np
import aerosandbox.numpy as anp
import aerosandbox as asb
import json
import neuralfoil as nf
import pickle
import pandas as pd
import seaborn as sns
import re

def plot_airfoil():

    coords_optimized = np.loadtxt('./data_storage/optimized_airfoil_test.txt')
    airfoil_data_loc = "./data_storage/airfoil_designs_revision3.pkl"
    optimized_design_loc = "../data_storage/optimized_airfoil.txt"
    with open(f"{airfoil_data_loc}", "rb") as data:
        airfoil_data = pickle.load(data)
    coords_orig = airfoil_data["coords"]

    coordinate_airfoil = asb.Airfoil("dae11")  # Initializing a default airfoil shape
    coordinate_airfoil.coordinates = coords_optimized
    coordinate_airfoil.name = f"optimized_design"
    kulfan_airfoil = coordinate_airfoil.to_kulfan_airfoil()

    aero_data = nf.get_aero_from_coordinates(
        coordinates=kulfan_airfoil.coordinates,
        alpha=0,
        Re=5e6,
    )
    print('\n', f'optimized CD:{aero_data["CD"]}, optimized CL: {aero_data["CL"]}')

    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(111)
    line1 = ax1.plot(
        coords_optimized[:, 0], coords_optimized[:, 1], color="red", label="Optimized", linewidth=2
    )
    line2 = ax1.plot(
        coords_orig[:, 0], coords_orig[:, 1], color="blue", label="Initial", linestyle='--', linewidth=2
    )
    # ax1.fill(coords_orig[:, 0], coords_orig[:, 1], color="blue", alpha=0.3)
    # ax1.fill(coords_optimized[:, 0], coords_optimized[:, 1], color="red", alpha=0.6)

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.grid(True, which="major", linestyle="-.", linewidth=1.0, alpha=0.5)
    ax1.minorticks_on()
    ax1.grid(True, which="minor", linestyle=":", linewidth=0.5)
    ax1.legend()
    plt.tight_layout()
    plt.savefig(f"./data_storage/optimized_vs_initial.png", dpi=300)
    # plt.show()
    plt.close()

def plot_optimization_comparison(optim_summary):
    optimization_log = pd.DataFrame(optim_summary)
    random_optim_file = 'data_store/Run13/optimized_airfoil_results_random.pkl'
    optim_file_run9 = './data_store/Run9/optimized_airfoil.pkl'
    optim_file_run10 = './data_store/Run10/optimized_airfoil.pkl'
    optim_file_run11 = './data_store/Run11/optimized_airfoil.pkl'
    optim_file_run12 = './data_store/Run12/optimized_airfoil.pkl'
    optim_file_run13 = './data_store/Run13/optimized_airfoil.pkl'
    with open(f"{random_optim_file}", "rb") as data:
        random_optim_data = pickle.load(data)
    with open(f"{optim_file_run9}", "rb") as data:
        run9_optim_data = pickle.load(data)
    with open(f"{optim_file_run10}", "rb") as data:
        run10_optim_data = pickle.load(data)
    with open(f"{optim_file_run11}", "rb") as data:
        run11_optim_data = pickle.load(data)
    with open(f"{optim_file_run12}", "rb") as data:
        run12_optim_data = pickle.load(data)
    with open(f"{optim_file_run13}", "rb") as data:
        run13_optim_data = pickle.load(data)

    random_optim_results = pd.DataFrame(random_optim_data)
    random_optim_results = pd.concat([optimization_log, random_optim_results], axis=1)
    random_optim_results['cl_cd'] = random_optim_results['Cl_optim'] / random_optim_results['Cd_optim']
    success_df = random_optim_results[random_optim_results['Success'] == 'Success']
    failed_df = random_optim_results[random_optim_results['Success'] != 'Success']

    run9_optim_results = pd.DataFrame(run9_optim_data)
    run10_optim_results = pd.DataFrame(run10_optim_data)
    run11_optim_results = pd.DataFrame(run11_optim_data)
    run12_optim_results = pd.DataFrame(run12_optim_data)
    run13_optim_results = pd.DataFrame(run13_optim_data)

    llm_optim_results = pd.concat([run9_optim_results, run10_optim_results, run11_optim_results, run12_optim_results, run13_optim_results])
    llm_optim_results['ID'] = range(1, 3*len(llm_optim_results)+1, 3)
    llm_optim_results['cl_cd'] = llm_optim_results['Cl_optim']/llm_optim_results['Cd_optim']

    fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=300)

    ax.scatter(success_df['ID'], success_df['cl_cd'],
               color='#1f77b4', marker='o', s=60, edgecolors='k', linewidths=0.8, label='Random design (converged)',alpha=0.8)

    ax.scatter(failed_df['ID'], failed_df['cl_cd'],
               color='#d62728', marker='X', s=80, edgecolors='k', linewidths=0.8, label='Random design (unconverged)', alpha=0.8)
    ax.scatter(llm_optim_results['ID'], llm_optim_results['cl_cd'],
               color='#f39c12', marker='*', s=120, edgecolors='k', linewidths=0.8, label='Agent designs')

    ax.set_xlabel('Simulation Run ID', fontsize=12, fontweight='bold', labelpad=8)
    ax.set_ylabel('Optimized Lift-to-Drag Ratio ($L/D$)', fontsize=12, fontweight='bold', labelpad=8)
    ax.grid(True, linestyle='--', alpha=0.4, which='both')
    ax.legend(frameon=True, facecolor='white', edgecolor='k', framealpha=0.9, loc='upper right')

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig('./data_store/optimized_ld_plot.png', dpi=300)
    # plt.show()
    plt.close()
    t = 1


def parse_simulation_log(log_data_or_file, is_file_path=False):
    """Parses a simulation log to extract run IDs and their success status.

    Args:
        log_data_or_file (str): The log text or path to the file.
        is_file_path (bool): Set to True if log_data_or_file is a file path.

    Returns:
        list of dict: A list containing IDs and statuses.
    """
    # Load content depending on whether a file path or string is provided
    if is_file_path:
        with open(log_data_or_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    else:
        lines = log_data_or_file.strip().split("\n")

    results = []
    run_id = 0
    current_status = None

    for line in lines:
        # Check for the start of a simulation run
        if "Initial CL" in line:
            # If a previous run ended abruptly without capturing status, log it as Unknown
            if run_id > 0 and current_status is None:
                results.append({"ID": run_id, "Success": "Unknown/Aborted"})

            run_id += 1
            current_status = None  # Reset status for the new run

        # Event 1: Success criteria match
        elif "Success!" in line:
            current_status = "Success"

        # Event 2: Solver constraint failure match
        elif "Solver could not satisfy all constraints" in line:
            current_status = "Failed (Constraints)"

        # If a status was recorded for this run block, save it
        if current_status and (len(results) < run_id):
            results.append({"ID": run_id, "Success": current_status})

    # Catch case where the final iteration is cut off or inconclusive
    if run_id > len(results):
        results.append({"ID": run_id, "Success": "Incomplete/Running"})

    return results

def analyze_log():
    # --- Execution Example ---

    # Paste your log chunk here to test locally
    log_file = './data_store/Run13/optimization_log_random_designs.txt'
    with open(log_file, 'r', encoding='utf-8') as file:
        log_content = file.read()

    # Call the function (assuming direct string text usage for this example)
    simulation_summary = parse_simulation_log(log_content, is_file_path=False)

    # Display results in a readable format
    print(f"{'Run ID':<10} | {'Status':<25}")
    print("-" * 40)
    for run in simulation_summary:
        print(f"{run['ID']:<10} | {run['Success']:<25}")

    return simulation_summary



def plot_airfoil_comparison(df, row_index=0, save_name="airfoil_comparison.png"):
    """Plots initial vs optimized shape for a specific run in the DataFrame."""
    # Extract data for the designated row
    init_coords = np.asarray(df.loc[0, "coord_init"][0])[row_index]
    final_coords = np.asarray(df.loc[0, "coord_optim"][0])[row_index]

    # Configure global journal styles (Times/Serif font style match for LaTeX)
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.linewidth"] = 1.0

    # Initialize a wide-aspect ratio figure typical for aerodynamic panels
    fig, ax = plt.subplots(figsize=(8.5, 2.5), dpi=300)

    # 1. Plot Baseline/Initial Geometry (Muted grey dashed profile)
    ax.plot(
        init_coords[:, 0],
        init_coords[:, 1],
        color="#7f7f7f",
        linestyle="--",
        linewidth=1.5,
        label="Initial Baseline",
    )

    # 2. Plot Optimized Profile (Solid high-contrast performance blue)
    ax.plot(
        final_coords[:, 0],
        final_coords[:, 1],
        color="#1f77b4",
        linestyle="-",
        linewidth=2.0,
        label="Optimized Design",
    )

    # Academic Labels using LaTeX notation for physics terms
    ax.set_xlabel(
        "$x/c$", fontsize=11, fontweight="bold", labelpad=4
    )
    ax.set_ylabel(
        "$y/c$",
        fontsize=11,
        fontweight="bold",
        labelpad=4,
    )
    # ax.set_title(
    #     f"Run ID: {row_index}",
    #     fontsize=12,
    #     fontweight="bold",
    #     pad=10,
    # )

    # MANDATORY: Prevent aspect ratio distortion of the physical structure
    # ax.set_aspect("equal", adjustable="box")

    # Clean grids and layout constraints
    ax.grid(True, linestyle=":", alpha=0.5)

    # Place legend cleanly outside of primary chord line contours
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="none",
        loc="best",
        fontsize=10,
    )

    # Despine unnecessary visual borders
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    # plt.show()
    plt.savefig(save_name, dpi=300, bbox_inches="tight")
    # plt.close()
    print(f"Publication-ready figure saved successfully as: {save_name}")

def extract_coords():
    optim_file_run9 = './data_store/Run9/optimized_airfoil.pkl'
    optim_file_run10 = './data_store/Run10/optimized_airfoil.pkl'
    optim_file_run11 = './data_store/Run11/optimized_airfoil.pkl'
    optim_file_run12 = './data_store/Run12/optimized_airfoil.pkl'
    optim_file_run13 = './data_store/Run13/optimized_airfoil.pkl'

    with open(f"{optim_file_run9}", "rb") as data:
        run9_optim_data = pickle.load(data)
    with open(f"{optim_file_run10}", "rb") as data:
        run10_optim_data = pickle.load(data)
    with open(f"{optim_file_run11}", "rb") as data:
        run11_optim_data = pickle.load(data)
    with open(f"{optim_file_run12}", "rb") as data:
        run12_optim_data = pickle.load(data)
    with open(f"{optim_file_run13}", "rb") as data:
        run13_optim_data = pickle.load(data)

    run9_optim_results = pd.DataFrame(run9_optim_data)
    run10_optim_results = pd.DataFrame(run10_optim_data)
    run11_optim_results = pd.DataFrame(run11_optim_data)
    run12_optim_results = pd.DataFrame(run12_optim_data)
    run13_optim_results = pd.DataFrame(run13_optim_data)

    llm_optim_results = pd.concat(
        [run9_optim_results, run10_optim_results, run11_optim_results, run12_optim_results, run13_optim_results])
    llm_optim_results['ID'] = range(1, 3 * len(llm_optim_results) + 1, 3)
    llm_optim_results['cl_cd'] = llm_optim_results['Cl_optim'] / llm_optim_results['Cd_optim']

    # plot_airfoil_comparison(llm_optim_results, row_index=0)
    # plot_airfoil_comparison(llm_optim_results, row_index=1)
    # plot_airfoil_comparison(llm_optim_results, row_index=2)
    # plot_airfoil_comparison(llm_optim_results, row_index=3)
    plot_airfoil_comparison(llm_optim_results, row_index=4)
    t = 1
#
optimization_summary = analyze_log()
# plot_optimization_comparison(optimization_summary)
extract_coords()
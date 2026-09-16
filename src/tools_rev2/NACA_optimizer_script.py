import aerosandbox as asb
import aerosandbox.numpy as anp
import neuralfoil as nf
import numpy as np
import pickle
from scipy.optimize import minimize
from scipy.special import comb
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from tqdm import tqdm



def get_naca_coords(m, p, t, n_points=200):
    """Standard NumPy NACA 4-digit coordinate generator."""
    x = np.linspace(0, 1, n_points)
    yt = 5 * t * (0.2969 * x ** 0.5 - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4)

    if p == 0:
        yc = np.zeros_like(x)
    else:
        # Avoid division by zero at the very start
        p_eff = max(p, 1e-6)
        yc = np.where(
            x < p_eff,
            m / (p_eff ** 2) * (2 * p_eff * x - x ** 2),
            m / ((1 - p_eff) ** 2) * ((1 - 2 * p_eff) + 2 * p_eff * x - x ** 2)
        )

    upper = np.stack([x[::-1], (yc + yt)[::-1]], axis=1)
    lower = np.stack([x[1:], (yc - yt)[1:]], axis=1)
    return np.vstack([upper, lower])

# def get_naca_coords(max_camber, camber_loc, thickness, n_points = 200):
#     x_t = anp.cosspace(
#         0, 1, n_points
#     )  # Generate some cosine-spaced points
#     y_t = (
#             5
#             * thickness
#             * (
#                     +0.2969 * x_t ** 0.5
#                     - 0.1260 * x_t
#                     - 0.3516 * x_t ** 2
#                     + 0.2843 * x_t ** 3
#                     - 0.1015 * x_t ** 4  # 0.1015 is original, #0.1036 for sharp TE
#             )
#     )
#
#     if camber_loc == 0:
#         camber_loc = (
#             0.5  # prevents divide by zero errors for things like naca0012's.
#         )
#
#     # Get camber
#     y_c = np.where(
#         x_t <= camber_loc,
#         max_camber / camber_loc ** 2 * (2 * camber_loc * x_t - x_t ** 2),
#         max_camber
#         / (1 - camber_loc) ** 2
#         * ((1 - 2 * camber_loc) + 2 * camber_loc * x_t - x_t ** 2),
#     )
#
#     # Get camber slope
#     dycdx = np.where(
#         x_t <= camber_loc,
#         2 * max_camber / camber_loc ** 2 * (camber_loc - x_t),
#         2 * max_camber / (1 - camber_loc) ** 2 * (camber_loc - x_t),
#     )
#     theta = np.arctan(dycdx)
#
#     # Combine everything
#     x_U = x_t - y_t * np.sin(theta)
#     x_L = x_t + y_t * np.sin(theta)
#     y_U = y_c + y_t * np.cos(theta)
#     y_L = y_c - y_t * np.cos(theta)
#
#     # Flip upper surface so it's back to front
#     x_U, y_U = x_U[::-1], y_U[::-1]
#
#     # Trim 1 point from lower surface so there's no overlap
#     x_L, y_L = x_L[1:], y_L[1:]
#
#     x = np.concatenate((x_U, x_L))
#     y = np.concatenate((y_U, y_L))
#
#     # Calculate the current chord length
#     # x_min = np.min(x)
#     # x_max = np.max(x)
#     # chord = x_max - x_min
#     #
#     # # Shift so LE is at 0 and scale so TE is at 1
#     # x_final = (x - x_min) / chord
#     #
#     # # Scale Y by the same chord factor to preserve the airfoil shape (thickness %)
#     # y_final = (y - y[np.argmin(x)]) / chord
#     #
#     # return np.stack((x_final, y_final), axis=1)
#
#     return np.stack((x, y), axis=1)


def get_aero(params):
    """Helper to run NeuralFoil and return the full dict."""
    m, p, t = params
    coords = get_naca_coords(m, p, t)
    af = asb.Airfoil(name="temp", coordinates=coords)
    # Using small model for speed in constraints, switch to xxlarge for final
    return af.get_aero_from_neuralfoil(alpha=0.0, Re=5e6, mach=0.6, model_size='xxxlarge')


# def objective(params):
#     try:
#         aero = get_aero(params)
#         if aero["analysis_confidence"] < 0.85: return 1e6
#         return -min(aero["CL"] / aero["CD"], 120)
#     except:
#         return 1e6

#-----------Soft constraint objective function----------------
def objective(params):
    try:
        aero = get_aero(params)
        # Use a soft penalty instead of a hard 1e6
        conf_penalty = 0
        if aero["analysis_confidence"] < 0.85:
            conf_penalty = 100 * (0.85 - aero["analysis_confidence"])

        ld = float(aero["CL"][0] / aero["CD"][0])
        return -min(ld, 120) + conf_penalty
    except:
        return 1e6

# --- Constraint Functions ---

def constraint_cl(params):
    return get_aero(params)["CL"] - 0.65  # CL >= 0.4


def constraint_cd(params):
    # CD < 0.1  =>  0.1 - CD >= 0
    return 0.01 - get_aero(params)["CD"]

def constraint_cm(params):
    # CM > -0.1  =>  0.1 +CM >= 0
    return get_aero(params)["CM"] + 0.2

def constraint_thickness(params):
    # t > 0.1  =>  t - 0.1 >= 0
    return params[2] - 0.15


# --- Trackers for the process ---


history = []
def callback_function(xk):
    """
    xk: The current parameter vector [m, p, t] at the end of the iteration.
    """
    m, p, t = xk
    aero = get_aero((m, p, t))
    cl = float(aero["CL"][0])
    cd = float(aero["CD"][0])
    cm = float(aero["CM"][0])
    ld = cl / cd

    # Save the step
    history.append({
        'm': m,
        'p': p,
        't': t,
        'ld': ld,
        'cl': cl,
        'cd': cd,
        'cm': cm
    })

    print(f"Iteration {len(history)}: m={m:.4f}, p={p:.4f}, t={t:.4f}, L/D={ld:.2f}")

def coords_to_kulfan(coords):
    coordinate_airfoil = asb.Airfoil(
        "dae11"
    )  # Initializing a default airfoil shape
    coordinate_airfoil.coordinates = coords
    coordinate_airfoil.name = 'naca'
    kulfan_airfoil = coordinate_airfoil.to_kulfan_airfoil()
    return kulfan_airfoil.coordinates

def optimize_airfoil_callback():
    airfoil_data_loc = "../data_storage/airfoil_designs.pkl"
    optimized_design_loc = "../data_storage/optimized_airfoil_random_samples.pkl"

    with open(f"{airfoil_data_loc}", "rb") as data:
        airfoil_data = pickle.load(data)
    coords = airfoil_data["coords"]
    design_ids = airfoil_data['design_ids']
    cambers = airfoil_data['max_camber']
    camber_locs = airfoil_data['camber_loc']
    thickness = airfoil_data['thickness']

    x0 = [cambers[6], camber_locs[6], thickness[6]]
    x0 = [0.05, 0.3, 0.12]        # Rev2 airfoil
    # x0 = [0.05, 0.4, 0.15]
    bounds = [(0.01, 0.095), (0.05, 0.9), (0.06, 0.40)]

    # Plot initial airfoil shape
    airfoil_orig = get_naca_coords(*x0)
    asb.Airfoil(coordinates=get_naca_coords(*x0), name='initial design').draw()

    print('\n', f'Initial CL: {get_aero(x0)['CL'][0]}, CD: {get_aero(x0)['CD'][0]}, CM: {get_aero(x0)['CM'][0]}')
    # List of all constraints
    cons = [
        {'type': 'ineq', 'fun': constraint_cl},
        {'type': 'ineq', 'fun': constraint_cd},
        # {'type': 'ineq', 'fun': constraint_cm},
        {'type': 'ineq', 'fun': constraint_thickness}
    ]

    print("Running optimization with Thickness and CD constraints...")
    res = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        callback=callback_function,
        options={'disp': True, 'maxiter': 50, 'eps': 0.001, 'ftol': 1e-6}
    )

    # --- Plotting the Results ---

    if history:
        iters = np.arange(1, len(history) + 1)
        lds = [h['ld'] for h in history]
        ms = [h['m'] for h in history]
        ps = [h['p'] for h in history]
        ts = [h['t'] for h in history]

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot L/D on primary Y-axis
        color = 'tab:red'
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('L/D Ratio', color=color)
        ax1.plot(iters, lds, color=color, marker='o', label='L/D Ratio')
        ax1.tick_params(axis='y', labelcolor=color)

        # Plot Parameters on secondary Y-axis
        ax2 = ax1.twinx()
        color_m = 'tab:blue'
        color_p = 'tab:green'
        color_t = 'tab:orange'

        ax2.set_ylabel('NACA Parameters', color='black')
        ax2.plot(iters, ms, color=color_m, linestyle='--', label='Camber (m)')
        ax2.plot(iters, ps, color=color_p, linestyle='--', label='Camber Pos (p)')
        ax2.plot(iters, ts, color=color_t, linestyle='--', label='Thickness (t)')

        fig.tight_layout()
        fig.legend(loc="upper left", bbox_to_anchor=(0.15, 0.85))
        plt.title("Optimization Evolution of NACA Airfoil")
        plt.show()

    if res.success:
        m_opt, p_opt, t_opt = res.x
        final_aero = get_aero(res.x)
        print(f"\nSuccess! NACA {int(m_opt * 100)}{int(p_opt * 10)}{int(t_opt * 100):02d}")
        print(f"Final thickness: {t_opt:.3f}")
        print(f"Final CD: {final_aero['CD'][0]}")
        print(f"Final CL: {final_aero['CL'][0]}")
        print(f"Final L/D: {final_aero['CL'][0] / final_aero['CD'][0]}")

        asb.Airfoil(coordinates=get_naca_coords(*res.x), name='optimized design').draw()
    else:
        print("Solver could not satisfy all constraints.")

    airfoil_optim = get_naca_coords(*res.x)


#-------------Optimization routine for one or multiple airfoils---------------
def optimize_airfoil_all(random_designs = False):
    results = []

    if random_designs:
        airfoil_data_loc = "../data_store/Run13/airfoil_designs.pkl"
        optimized_design_loc = "../data_store/optimized_airfoil_results_random.pkl"

        with open(f"{airfoil_data_loc}", "rb") as data:
            airfoil_data = pickle.load(data)
        coords = airfoil_data["coords"]
        design_ids = airfoil_data['design_ids']
        cambers = airfoil_data['max_camber']
        camber_locs = airfoil_data['camber_loc']
        thickness = airfoil_data['thickness']

        for i, (design_id, m, p, t) in enumerate(tqdm(zip(design_ids, cambers, camber_locs, thickness))):
            x0 = [m, p, t]
            bounds = [(0.01, 0.095), (0.05, 0.9), (0.06, 0.40)]

            # Plot initial airfoil shape
            airfoil_orig = get_naca_coords(*x0)
            asb.Airfoil(coordinates=get_naca_coords(*x0), name=f'initial design: {design_id}').draw()
            aero_orig = get_aero(x0)

            print('\n', f'Initial CL: {get_aero(x0)['CL'][0]}, CD: {get_aero(x0)['CD'][0]}, CM: {get_aero(x0)['CM'][0]}')
            # List of all constraints
            cons = [
                {'type': 'ineq', 'fun': constraint_cl},
                {'type': 'ineq', 'fun': constraint_cd},
                {'type': 'ineq', 'fun': constraint_cm},
                {'type': 'ineq', 'fun': constraint_thickness}
            ]

            print("Running optimization with Thickness and CD constraints...")
            res = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                callback=callback_function,
                options={'disp': True, 'maxiter': 50, 'eps': 0.001, 'ftol': 1e-6}
            )

            # --- Plotting the Results ---

            # if history:
            #     iters = np.arange(1, len(history) + 1)
            #     lds = [h['ld'] for h in history]
            #     ms = [h['m'] for h in history]
            #     ps = [h['p'] for h in history]
            #     ts = [h['t'] for h in history]
            #
            #     fig, ax1 = plt.subplots(figsize=(10, 6))
            #
            #     # Plot L/D on primary Y-axis
            #     color = 'tab:red'
            #     ax1.set_xlabel('Iteration')
            #     ax1.set_ylabel('L/D Ratio', color=color)
            #     ax1.plot(iters, lds, color=color, marker='o', label='L/D Ratio')
            #     ax1.tick_params(axis='y', labelcolor=color)
            #
            #     # Plot Parameters on secondary Y-axis
            #     ax2 = ax1.twinx()
            #     color_m = 'tab:blue'
            #     color_p = 'tab:green'
            #     color_t = 'tab:orange'
            #
            #     ax2.set_ylabel('NACA Parameters', color='black')
            #     ax2.plot(iters, ms, color=color_m, linestyle='--', label='Camber (m)')
            #     ax2.plot(iters, ps, color=color_p, linestyle='--', label='Camber Pos (p)')
            #     ax2.plot(iters, ts, color=color_t, linestyle='--', label='Thickness (t)')
            #
            #     fig.tight_layout()
            #     fig.legend(loc="upper left", bbox_to_anchor=(0.15, 0.85))
            #     plt.title("Optimization Evolution of NACA Airfoil")
            #     plt.show()
            #     plt.close()

            if res.success:
                m_opt, p_opt, t_opt = res.x
                final_aero = get_aero(res.x)
                print(f"\nSuccess! NACA {int(m_opt * 100)}{int(p_opt * 10)}{int(t_opt * 100):02d}")
                print(f"Final thickness: {t_opt:.3f}")
                print(f"Final CD: {final_aero['CD'][0]}")
                print(f"Final CL: {final_aero['CL'][0]}")
                print(f"Final L/D: {final_aero['CL'][0] / final_aero['CD'][0]}")

                asb.Airfoil(coordinates=get_naca_coords(*res.x), name=f'optimized design: {design_id}').draw()
            else:
                print("Solver could not satisfy all constraints.")

            airfoil_optim = get_naca_coords(*res.x)

            results.append({
                'design_id': design_id,
                'coord_init': airfoil_orig,
                'coord_optim': airfoil_optim,
                'Cd_optim': final_aero['CD'][0],
                'Cl_optim': final_aero['CL'][0] ,
                'Cm_optim': final_aero['CM'][0] ,
                'Cd_init': aero_orig ['CD'][0],
                'Cl_init': aero_orig ['CL'][0],
                'Cm_init': aero_orig ['CM'][0],
                'optim_history': history,
                'camber_opt': m_opt,
                'camber_loc_opt': p_opt,
                'thickness_opt': t_opt
            })

            del airfoil_optim
            del res


    else:
        run_num = 'Run11'
        airfoil_data_loc = f"../data_store/{run_num}/airfoil_designs_revision1.pkl"
        optimized_design_loc = f"../data_store/{run_num}/optimized_airfoil.pkl"

        with open(f"{airfoil_data_loc}", "rb") as data:
            airfoil_data = pickle.load(data)
        coords = airfoil_data["coords"]
        camber = airfoil_data['design_params'][0]
        camber_loc = airfoil_data['design_params'][1]
        thickness = airfoil_data['design_params'][2]

        x0 = [camber, camber_loc, thickness]
        bounds = [(0.01, 0.095), (0.05, 0.9), (0.06, 0.40)]

        # Plot initial airfoil shape
        airfoil_orig = get_naca_coords(*x0)

        np.savetxt(f'../data_store/{run_num}/coords_init.txt', airfoil_orig.astype(float), delimiter=',')
        asb.Airfoil(coordinates=get_naca_coords(*x0), name=f'initial design').draw()
        aero_orig = get_aero(x0)

        print('\n',
              f'Initial CL: {get_aero(x0)['CL'][0]}, CD: {get_aero(x0)['CD'][0]}, CM: {get_aero(x0)['CM'][0]}')
        # List of all constraints
        cons = [
            {'type': 'ineq', 'fun': constraint_cl},
            {'type': 'ineq', 'fun': constraint_cd},
            {'type': 'ineq', 'fun': constraint_cm},
            {'type': 'ineq', 'fun': constraint_thickness}
        ]

        print("Running optimization with Thickness and CD constraints...")
        res = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            callback=callback_function,
            options={'disp': True, 'maxiter': 50, 'eps': 0.001, 'ftol': 1e-6}
        )

        # --- Plotting the Results ---

        if history:
            iters = np.arange(1, len(history) + 1)
            lds = [h['ld'] for h in history]
            ms = [h['m'] for h in history]
            ps = [h['p'] for h in history]
            ts = [h['t'] for h in history]

            fig, ax1 = plt.subplots(figsize=(10, 6))

            # Plot L/D on primary Y-axis
            color = 'tab:red'
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('L/D Ratio', color=color)
            ax1.plot(iters, lds, color=color, marker='o', label='L/D Ratio')
            ax1.tick_params(axis='y', labelcolor=color)

            # Plot Parameters on secondary Y-axis
            ax2 = ax1.twinx()
            color_m = 'tab:blue'
            color_p = 'tab:green'
            color_t = 'tab:orange'

            ax2.set_ylabel('NACA Parameters', color='black')
            ax2.plot(iters, ms, color=color_m, linestyle='--', label='Camber (m)')
            ax2.plot(iters, ps, color=color_p, linestyle='--', label='Camber Pos (p)')
            ax2.plot(iters, ts, color=color_t, linestyle='--', label='Thickness (t)')

            fig.tight_layout()
            fig.legend(loc="upper left", bbox_to_anchor=(0.15, 0.85))
            plt.title("Optimization Evolution of NACA Airfoil")
            plt.show()

        if res.success:
            m_opt, p_opt, t_opt = res.x
            final_aero = get_aero(res.x)
            print(f"\nSuccess! NACA {int(m_opt * 100)}{int(p_opt * 10)}{int(t_opt * 100):02d}")
            print(f"Final thickness: {t_opt:.3f}")
            print(f"Final CD: {final_aero['CD'][0]}")
            print(f"Final CL: {final_aero['CL'][0]}")
            print(f"Final CM: {final_aero['CM'][0]}")
            print(f"Final L/D: {final_aero['CL'][0] / final_aero['CD'][0]}")

            asb.Airfoil(coordinates=get_naca_coords(*res.x), name=f'optimized design').draw()
        else:
            print("Solver could not satisfy all constraints.")

        airfoil_optim = get_naca_coords(*res.x)
        # kulfan_optim = coords_to_kulfan(airfoil_optim)
        np.savetxt(f'../data_store/{run_num}/coords_optim.txt', airfoil_optim.astype(float), delimiter=',')

        results.append({
            'coord_init': airfoil_orig,
            'coord_optim': airfoil_optim,
            'Cd_optim': final_aero['CD'][0],
            'Cl_optim': final_aero['CL'][0],
            'Cm_optim': final_aero['CM'][0],
            'Cd_init': aero_orig['CD'][0],
            'Cl_init': aero_orig['CL'][0],
            'Cm_init': aero_orig['CM'][0],
            'optim_history': history,
            'camber_opt': m_opt,
            'camber_loc_opt': p_opt,
            'thickness_opt': t_opt,
            'camber_init': camber,
            'camber_loc_init': camber_loc,
            'thickness_init': thickness
        })

        del airfoil_optim
        del res

    with open(f"{optimized_design_loc}", "wb") as outp:
        pickle.dump(results, outp)

#------------Function to convert NACA coordinates to Kulfan coordinates---------------
def naca_to_kulfan():
    # =====================================================================
    # 2. CORE KULFAN (CST) MATHEMATICAL FUNCTIONS
    # =====================================================================
    def bernstein_poly(i, n, x):
        """Calculates the i-th Bernstein polynomial of degree n."""
        return comb(n, i) * (x ** i) * ((1 - x) ** (n - i))

    def cst_shape_matrix(x, n_order):
        """Generates the system matrix mapping X coordinates to polynomial terms."""
        matrix = np.zeros((len(x), n_order + 1))
        class_func = np.sqrt(x) * (1 - x)
        for i in range(n_order + 1):
            matrix[:, i] = class_func * bernstein_poly(i, n_order, x)
        return matrix

    def fit_kulfan_weights(x, y, n_order=4):
        """Performs linear regression to find Kulfan weights for a set of points."""
        # Isolate trailing edge offset contribution
        # dy_te = y[-1] if x[-1] == 1.0 else 0.0
        dy_te = 0.0
        y_adjusted = y - x * dy_te

        matrix = cst_shape_matrix(x, n_order)

        # Solve linear matrix problem: Matrix * Weights = Y_adjusted
        weights, _, _, _ = np.linalg.lstsq(matrix, y_adjusted, rcond=None)
        return weights, dy_te

    def generate_cst_surface(x, weights, dy_te):
        """Reconstructs Y coordinates given arbitrary X bounds and Kulfan weights."""
        n_order = len(weights) - 1
        class_func = np.sqrt(x) * (1 - x)

        shape_func = np.zeros_like(x)
        for i, w in enumerate(weights):
            shape_func += w * bernstein_poly(i, n_order, x)

        return class_func * shape_func + x * dy_te

    # =====================================================================
    # 3. EXTRACT AND RECONSTRUCT LOOP FLOW
    # =====================================================================

    # Split continuous layout to isolate individual surfaces for the regression step
    run_num = 'Run13_human_user_test'
    raw_coords = np.loadtxt(f'../data_store/{run_num}/coords_optim.txt', delimiter=',')
    n_points = len(raw_coords)
    half_idx = n_points // 2

    # Raw points wrap from upper TE -> LE -> lower TE
    upper_raw = raw_coords[:half_idx + 1]
    lower_raw = raw_coords[half_idx:]

    # Sort from LE (0) to TE (1) linearly so the least-squares matrix maps properly
    upper_raw_sorted = upper_raw[np.argsort(upper_raw[:, 0])]
    lower_raw_sorted = lower_raw[np.argsort(lower_raw[:, 0])]

    # --- AUTOMATIC EXTRACTION ---
    # Fits a 4th-order polynomial (5 weights total per side)
    w_upper, dy_te_upper = fit_kulfan_weights(upper_raw_sorted[:, 0], upper_raw_sorted[:, 1], n_order=4)
    w_lower, dy_te_lower = fit_kulfan_weights(lower_raw_sorted[:, 0], lower_raw_sorted[:, 1], n_order=4)

    print("--- Automated Extraction Results ---")
    print("Extracted Upper Weights:", np.round(w_upper, 4))
    print("Extracted Lower Weights:", np.round(w_lower, 4))
    print(f"Extracted TE Offset:     {abs(dy_te_upper):.5f}\n")

    # --- CONSOLIDATE INTO ONE RECONSTRUCTED ARRAY LOOP ---
    # Generate high-resolution clean baseline spacing (e.g., 150 points per surface)
    n_output_points = 200
    x_clean_space = np.linspace(0, 1, n_output_points)

    # Reconstruct fresh Y surfaces from our calculated weights
    y_upper_rebuilt = generate_cst_surface(x_clean_space, w_upper, dy_te_upper)
    y_lower_rebuilt = generate_cst_surface(x_clean_space, w_lower, dy_te_lower)

    # Re-orient surfaces into single continuous loop boundary:
    # Part 1: Upper surface goes backward (Trailing Edge x=1 down to Leading Edge x=0)
    x_upper_loop = x_clean_space[::-1]
    y_upper_loop = y_upper_rebuilt[::-1]

    # Part 2: Lower surface goes forward (Leading Edge x=0 out to Trailing Edge x=1)
    # Drop index [1:] to remove leading edge point repetition at (0,0)
    x_lower_loop = x_clean_space[1:]
    y_lower_loop = y_lower_rebuilt[1:]

    # Combine everything into a unified array shape (N, 2)
    final_x = np.concatenate((x_upper_loop, x_lower_loop))
    final_y = np.concatenate((y_upper_loop, y_lower_rebuilt[1:]))
    kulfan_airfoil = np.stack((final_x, final_y), axis=1)
    np.savetxt(f'../data_store/{run_num}/coords_optim_kulfan.txt', kulfan_airfoil, delimiter=',')
    t = 1

    print("--- Unified Reconstructed Array Properties ---")
    print(f"Total Shape Matrix: {kulfan_airfoil.shape}")
    print(f"First Row (Upper TE): {kulfan_airfoil[0]}")
    print(f"Middle Row (Apex LE): {kulfan_airfoil[n_output_points - 1]}")
    print(f"Last Row (Lower TE):  {kulfan_airfoil[-1]}")


# optimize_airfoil_random()
optimize_airfoil_all(random_designs=False)
# naca_to_kulfan()
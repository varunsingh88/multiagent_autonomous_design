import aerosandbox.numpy as np
import aerosandbox as asb
import json
import neuralfoil as nf
import pickle
import aerosandbox.numpy as anp
from scipy.stats import qmc
from crewai.tools import tool, BaseTool
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field
from typing import Type, List, Any, Optional
import pandas as pd
from pydantic_core.core_schema import arguments_schema


class SamplerInput(BaseModel):
    """
    Input schema for GeometrySampler
    """

    strategy: str = Field(
        ...,
        description="strategy to be used for parameter sampling within the defined space. Strategy should be one of the following: [lhs, sobol, halton, random]",
    )
    num_samples: int = Field(
        ..., description="number of samples to be generated"
    )


class GeometrySampler(BaseTool):
    name: str = "Geometry_sampling_tool"
    description: str = (
        "Tool that samples parameter values from a given parameter space based on the sampling strategy"
    )
    args_schema: Type[BaseModel] = SamplerInput

    def _run(self, strategy: str, num_samples: int = 10) -> str:

        data_fileloc = "./data_storage/airfoil_params.pkl"
        design_space = {
            "max_camber": (0.01, 0.095),
            "camber_loc": (0.05, 0.9),
            "thickness": (0.01, 0.40),
        }

        keys = list(design_space.keys())
        bounds = np.array([design_space[k] for k in keys])
        dim = len(keys)

        # Select the sampling method
        method = strategy.lower()
        if method == "lhs":
            sampler = qmc.LatinHypercube(d=dim)
        elif method == "sobol":
            sampler = qmc.Sobol(d=dim, scramble=True)
        elif method == "halton":
            sampler = qmc.Halton(d=dim, scramble=True)
        elif method == "random":
            sampler = None
            sample = np.random.rand(num_samples, dim)
        else:
            raise ValueError(
                "Unsupported method. Choose from ['lhs', 'sobol', 'halton', 'random']."
            )

        # Generate samples
        if sampler is not None:
            sample = sampler.random(n=num_samples)

        # Scale to bounds
        scaled_samples = qmc.scale(sample, bounds[:, 0], bounds[:, 1])

        design_ID = [f"ID-{i + 1}" for i in range(num_samples)]

        # Optionally return as JSON
        # if as_json:
        #     data = {"variables": keys, "samples": scaled_samples.tolist()}
        #     return json.dumps(data, indent=4)
        pickle_obj = {
            "parameters": keys,
            "samples": scaled_samples.tolist(),
            "design_ID": design_ID,
        }

        with open(f"{data_fileloc}", "wb") as outp:
            pickle.dump(pickle_obj, outp)
        return f"Design samples generated and stored"


class NACAProfilerInput(BaseModel):
    """Input schema for NACAProfileGenerator."""

    design_file_loc: str = Field(
        ..., description="File location where newly sampled airfoil designs are located"
    )


class NACAProfileGenerator(BaseTool):
    name: str = "NACA Generator Tool"
    description: str = "Generates a given number of NACA profiles."

    # args_schema: Type[BaseModel] = NACAProfilerInput

    def _run(self) -> str:
        n_points_per_side = 200
        data_fileloc = "./data_storage/airfoil_designs.pkl"
        design_file_loc = "./data_storage/airfoil_params.pkl"

        with open(f"{design_file_loc}", "rb") as data:
            airfoil_data = pickle.load(data)

        geom_params = airfoil_data["samples"]
        design_id = airfoil_data["design_ID"]

        coords_ls = []
        design_id_ls = []
        max_camber_ls = []
        camber_loc_ls = []
        thickness_ls = []

        for i, (params, ID) in enumerate(zip(geom_params, design_id)):

            max_camber = params[0]
            camber_loc = params[1]
            thickness = params[2]

            x_t = np.cosspace(
                0, 1, n_points_per_side
            )  # Generate some cosine-spaced points
            y_t = (
                    5
                    * thickness
                    * (
                            +0.2969 * x_t ** 0.5
                            - 0.1260 * x_t
                            - 0.3516 * x_t ** 2
                            + 0.2843 * x_t ** 3
                            - 0.1015 * x_t ** 4  # 0.1015 is original, #0.1036 for sharp TE
                    )
            )

            if camber_loc == 0:
                camber_loc = (
                    0.5  # prevents divide by zero errors for things like naca0012's.
                )

            # Get camber
            y_c = np.where(
                x_t <= camber_loc,
                max_camber / camber_loc ** 2 * (2 * camber_loc * x_t - x_t ** 2),
                max_camber
                / (1 - camber_loc) ** 2
                * ((1 - 2 * camber_loc) + 2 * camber_loc * x_t - x_t ** 2),
            )

            # Get camber slope
            dycdx = np.where(
                x_t <= camber_loc,
                2 * max_camber / camber_loc ** 2 * (camber_loc - x_t),
                2 * max_camber / (1 - camber_loc) ** 2 * (camber_loc - x_t),
            )
            theta = np.arctan(dycdx)

            # Combine everything
            x_U = x_t - y_t * np.sin(theta)
            x_L = x_t + y_t * np.sin(theta)
            y_U = y_c + y_t * np.cos(theta)
            y_L = y_c - y_t * np.cos(theta)

            # Flip upper surface so it's back to front
            x_U, y_U = x_U[::-1], y_U[::-1]

            # Trim 1 point from lower surface so there's no overlap
            x_L, y_L = x_L[1:], y_L[1:]

            x = np.concatenate((x_U, x_L))
            y = np.concatenate((y_U, y_L))

            coords_ls.append(np.stack((x, y), axis=1))
            design_id_ls.append(ID)
            max_camber_ls.append(max_camber)
            camber_loc_ls.append(camber_loc)
            thickness_ls.append(thickness)

        pickle_obj = {"coords": coords_ls, "design_ids": design_id_ls, 'max_camber': max_camber_ls,
                      'camber_loc': camber_loc_ls, 'thickness': thickness_ls}
        with open(f"{data_fileloc}", "wb") as outp:
            pickle.dump(pickle_obj, outp)

        return f"All airfoil coordinate samples generated and saved "


# class PlotGeomInput(BaseModel):
#     """Input schema for PlotProfile."""
#
#     airfoil_data_loc: str = Field(
#         ...,
#         description="filename with location of airfoil design data is stored in .pkl format",
#     )


class PlotProfile(BaseTool):
    name: str = "Plot_NACA_airfoil"
    description: str = "Plots the shapes of new airfoil designs"

    # args_schema: Type[BaseModel] = PlotGeomInput

    def _run(self) -> str:
        airfoil_data_locn = "./data_storage/airfoil_designs.pkl"
        with open(f"{airfoil_data_locn}", "rb") as data:
            airfoil_data = pickle.load(data)
        if "coords" in airfoil_data:
            coords = airfoil_data["coords"]
            ids = airfoil_data["design_ids"]
            cambers = airfoil_data['max_camber']
            camber_loc = airfoil_data['camber_loc']
            thickness = airfoil_data['thickness']
            for i, (coord, design_id, m, p, t) in enumerate(zip(coords, ids, cambers, camber_loc, thickness)):
                fig = plt.figure(figsize=(8.5, 2.5), dpi=300)
                ax1 = fig.add_subplot(111)
                line1 = ax1.plot(coord[:, 0], coord[:, 1], color="red", label="Points")
                scatter1 = ax1.scatter(coord[:, 0], coord[:, 1], marker="o", s=10)
                # ax1.fill(coords[:, 0], coords[:, 1], color="red", alpha=0.7)

                ax1.set_xlabel("x")
                ax1.set_ylabel("y")
                ax1.grid(True, which="major", linestyle="-.", linewidth=1.0, alpha=0.5)
                ax1.minorticks_on()
                ax1.grid(True, which="minor", linestyle=":", linewidth=0.5)
                ax1.set_title(f"{design_id}, (max_camber: {m:.3f}, camber_loc: {p:.3f}, thickness:{t:.3f})", fontsize=12, loc="center", color="blue")
                # plt.legend(loc="upper right")
                # plt_axis.set_title(title_name)
                plt.tight_layout()
                plt.savefig(f"./data_storage/{design_id}.png", dpi=300)
                # plt.show()
                plt.close()
        else:
            print(
                "\n",
                "coords does not exist in dataset provided. Visualization tool is quitting.",
            )
        return "Visualization complete."


class DesignEvalInput(BaseModel):
    """
    Input schema for DesignEval
    """

    # airfoil_data_loc: str = Field(
    #     ...,
    #     description="filename with location of airfoil design data is stored in .pkl format",
    # )
    Re: float = Field(..., description="Reynolds number for design evaluation")
    aoa: float = Field(..., description="Angle of attack for design evaluation")
    Ma: float = Field(..., description="Mach number for design evaluation")


class DesignEvaluation(BaseTool):
    name: str = "Airfoil evaluation tool"
    description: str = (
        "To perform aerodynamic analysis of airfoil geometries generated by NACA Generator Tool. Export the analysis results as a .csv file"
    )
    args_schema: Type[BaseModel] = DesignEvalInput

    def get_naca_coords(self, max_camber, camber_loc, thickness, n_points=200):
        x_t = anp.cosspace(
            0, 1, n_points
        )  # Generate some cosine-spaced points
        y_t = (
                5
                * thickness
                * (
                        +0.2969 * x_t ** 0.5
                        - 0.1260 * x_t
                        - 0.3516 * x_t ** 2
                        + 0.2843 * x_t ** 3
                        - 0.1015 * x_t ** 4  # 0.1015 is original, #0.1036 for sharp TE
                )
        )

        if camber_loc == 0:
            camber_loc = (
                0.5  # prevents divide by zero errors for things like naca0012's.
            )

        # Get camber
        y_c = np.where(
            x_t <= camber_loc,
            max_camber / camber_loc ** 2 * (2 * camber_loc * x_t - x_t ** 2),
            max_camber
            / (1 - camber_loc) ** 2
            * ((1 - 2 * camber_loc) + 2 * camber_loc * x_t - x_t ** 2),
        )

        # Get camber slope
        dycdx = np.where(
            x_t <= camber_loc,
            2 * max_camber / camber_loc ** 2 * (camber_loc - x_t),
            2 * max_camber / (1 - camber_loc) ** 2 * (camber_loc - x_t),
        )
        theta = np.arctan(dycdx)

        # Combine everything
        x_U = x_t - y_t * np.sin(theta)
        x_L = x_t + y_t * np.sin(theta)
        y_U = y_c + y_t * np.cos(theta)
        y_L = y_c - y_t * np.cos(theta)

        # Flip upper surface so it's back to front
        x_U, y_U = x_U[::-1], y_U[::-1]

        # Trim 1 point from lower surface so there's no overlap
        x_L, y_L = x_L[1:], y_L[1:]

        x = np.concatenate((x_U, x_L))
        y = np.concatenate((y_U, y_L))

        return np.stack((x, y), axis=1)

    def get_aero(self, params, Re, alpha, Ma):
        """Helper to run NeuralFoil and return the full dict."""
        m, p, t = params
        coords = self.get_naca_coords(m, p, t)
        af = asb.Airfoil(name="temp", coordinates=coords)
        # Using small model for speed in constraints, switch to xxlarge for final
        return af.get_aero_from_neuralfoil(alpha=alpha, Re=Re, mach=Ma, model_size='xxxlarge')

    def _run(self, Re, aoa, Ma) -> str:
        data_fileloc = "./data_storage/design_evaluation.csv"
        airfoil_data_locn = "./data_storage/airfoil_designs.pkl"  # TODO: Remove this line

        design_id_ls = []
        CD_ls = []
        CL_ls = []
        CM_ls = []

        with open(f"{airfoil_data_locn}", "rb") as data:
            airfoil_data = pickle.load(data)
        coords = airfoil_data["coords"]
        ids = airfoil_data["design_ids"]
        max_camber = airfoil_data['max_camber']
        camber_loc = airfoil_data['camber_loc']
        thickness = airfoil_data['thickness']

        for i, (m, p, t, design_id) in enumerate(zip(max_camber, camber_loc, thickness, ids)):
            x0 = [m, p, t]
            aero_data = self.get_aero(x0, Re, aoa, Ma)

            design_id_ls.append(design_id)
            CD_ls.append(aero_data["CD"][0])
            CL_ls.append(aero_data["CL"][0])
            CM_ls.append(aero_data["CM"][0])

        arr = np.concatenate(
            [
                np.stack(design_id_ls)[:, None],
                np.stack(CD_ls)[:, None],
                np.stack(CL_ls)[:, None],
                np.stack(CM_ls)[:, None],
                np.stack(max_camber)[:, None],
                np.stack(camber_loc)[:, None],
                np.stack(thickness)[:, None]
            ],
            axis=1,
        )

        df = pd.DataFrame(arr, columns=["design_ID", "CD", "CL", "CM", "max_camber", "camber_loc", "thickness"])
        df.to_csv(data_fileloc, index=False)
        return f"All aerodynamic analysis data is stored"


class DesignSelectionInput(BaseModel):
    """
    Input schema for SelectDesignTool
    """
    CL_min: float = Field(
        ..., description="minimum value of acceptable CL for design screening"
    )
    num_samples: int = Field(
        ...,
        description="Number of design samples meeting constraint to be chosen for further assessment",
    )


class SelectDesignTool(BaseTool):
    name: str = "Airfoil_selection_tool"
    description: str = (
        "To select top 10 airfoil designs that meet the minimum CL requirements. Then export the list of selected design and their aerodynamic data as a .csv file"
    )
    args_schema: Type[BaseModel] = DesignSelectionInput

    def _run(self, CL_min, num_samples) -> str:
        data_fileloc = "./data_storage/selected_design.csv"
        # Evaluate each constraint for every row
        aerodynamic_data_loc = "./data_storage/design_evaluation.csv"  # TODO: Remove this line
        analysis_data = pd.read_csv(aerodynamic_data_loc)
        # Define constraints as lambdas or simple expressions
        constraints = {
            "CL": lambda x: x > CL_min,
        }

        # Evaluate each constraint for every row
        for metric, condition in constraints.items():
            analysis_data[f"{metric}_pass"] = analysis_data[metric].apply(condition)

        # Check overall pass/fail (all constraints satisfied)
        analysis_data["All_Pass"] = analysis_data[
            [f"{m}_pass" for m in constraints]
        ].all(axis=1)
        passing_df = analysis_data[analysis_data["All_Pass"]]
        sorted_df = passing_df.sort_values(by="CL", ascending=False)
        top_df = sorted_df.head(num_samples)
        top_designs = top_df["design_ID"].tolist()  # List of selected design IDs
        top_df.to_csv(data_fileloc, index=False)

        return f"selected airfoil design with evaluation stored. The selected design ID list is {top_designs}"


class DesignRevisionInput(BaseModel):
    """Input schema for ReviseNACAProfile"""

    max_camber: float = Field(...,
                              description="Camber value of revised airfoil design, expressed in value such as  0.05, 0.2 etc not percentage")
    camber_loc: float = Field(
        ...,
        description="Location of max camber for revised airfoil design, expressed in value such as  0.05, 0.2 etc not percentage"
    )
    thickness: float = Field(
        ...,
        description="Max thickness for revised airfoil design, expressed in value such as  0.05, 0.2 etc not percentage"
    )
    design_iter: int = Field(..., description="Design revision number")


class ReviseNACAProfile(BaseTool):
    name: str = "generate_revised_airfoil"
    description: str = "Generates a new NACA profile using parameter input"
    args_schema: Type[BaseModel] = DesignRevisionInput

    def _run(
            self, max_camber: float, camber_loc: float, thickness: float, design_iter: int
    ) -> str:
        n_points_per_side = 200
        data_fileloc = f"./data_storage/airfoil_designs_revision{design_iter}.pkl"

        x_t = np.cosspace(0, 1, n_points_per_side)  # Generate some cosine-spaced points
        y_t = (
                5
                * thickness
                * (
                        +0.2969 * x_t ** 0.5
                        - 0.1260 * x_t
                        - 0.3516 * x_t ** 2
                        + 0.2843 * x_t ** 3
                        - 0.1015 * x_t ** 4  # 0.1015 is original, #0.1036 for sharp TE
                )
        )

        if camber_loc == 0:
            camber_loc = (
                0.5  # prevents divide by zero errors for things like naca0012's.
            )

        # Get camber
        y_c = np.where(
            x_t <= camber_loc,
            max_camber / camber_loc ** 2 * (2 * camber_loc * x_t - x_t ** 2),
            max_camber
            / (1 - camber_loc) ** 2
            * ((1 - 2 * camber_loc) + 2 * camber_loc * x_t - x_t ** 2),
        )

        # Get camber slope
        dycdx = np.where(
            x_t <= camber_loc,
            2 * max_camber / camber_loc ** 2 * (camber_loc - x_t),
            2 * max_camber / (1 - camber_loc) ** 2 * (camber_loc - x_t),
        )
        theta = np.arctan(dycdx)

        # Combine everything
        x_U = x_t - y_t * np.sin(theta)
        x_L = x_t + y_t * np.sin(theta)
        y_U = y_c + y_t * np.cos(theta)
        y_L = y_c - y_t * np.cos(theta)

        # Flip upper surface so it's back to front
        x_U, y_U = x_U[::-1], y_U[::-1]

        # Trim 1 point from lower surface so there's no overlap
        x_L, y_L = x_L[1:], y_L[1:]

        x = np.concatenate((x_U, x_L))
        y = np.concatenate((y_U, y_L))

        pickle_obj = {
            "coords": np.stack((x, y), axis=1),
            "design_params": (max_camber, camber_loc, thickness),
        }
        with open(f"{data_fileloc}", "wb") as outp:
            pickle.dump(pickle_obj, outp)

        return f"Revision# {design_iter} of airfoil design generated and saved"


class RevisePlotInput(BaseModel):
    """Input schema for RevisePlotProfile."""

    # revised_design_loc: str = Field(
    #     ...,
    #     description="filename with location of revised airfoil design data is stored in .pkl format",
    # )
    design_iter: int = Field(..., description="Design revision number")


class RevisePlotProfile(BaseTool):
    name: str = "plot_revised_airfoil"
    description: str = "Plot the shapes of updated airfoil design"
    args_schema: Type[BaseModel] = RevisePlotInput

    def _run(self, design_iter) -> str:

        revised_design_locn = f"./data_storage/airfoil_designs_revision{design_iter}.pkl"
        with open(f"{revised_design_locn}", "rb") as data:
            airfoil_data = pickle.load(data)
        if "coords" in airfoil_data:
            coords = airfoil_data["coords"]
            design_params = airfoil_data["design_params"]
            try:
                fig = plt.figure(figsize=(8.5, 2.5), dpi=300)
                ax1 = fig.add_subplot(111)
                line1 = ax1.plot(
                    coords[:, 0], coords[:, 1], color="red", label="Points"
                )
                scatter1 = ax1.scatter(coords[:, 0], coords[:, 1], marker="o", s=10)
                # ax1.fill(coords[:, 0], coords[:, 1], color="red", alpha=0.7)

                ax1.set_xlabel("x")
                ax1.set_ylabel("y")
                ax1.grid(True, which="major", linestyle="-.", linewidth=1.0, alpha=0.5)
                ax1.minorticks_on()
                ax1.grid(True, which="minor", linestyle=":", linewidth=0.5)
                ax1.set_title(
                    f"(max_camber: {design_params[0]:.3f}, camber_loc: {design_params[1]:.3f}, thickness:{design_params[2]:.3f})",
                    fontsize=14,
                    loc="center",
                    color="blue",
                )
                plt.tight_layout()
                plt.savefig(f"./data_storage/revision_{design_iter}.png", dpi=300)
                # plt.show()
                plt.close()
            except Exception as e:
                print(f"Revision includes more than one design please check")
        else:
            print(
                "\n",
                "coords does not exist in dataset provided. Visualization tool is quitting.",
            )
        return "Visualization complete."


class ReviseEvaluationInput(BaseModel):
    """
    Input schema for ReviseEvaluation
    """

    # revised_design_loc: str = Field(
    #     ...,
    #     description="filename with location of revised airfoil design data is stored in .pkl format",
    # )
    Re: float = Field(..., description="Reynolds number for design evaluation")
    aoa: float = Field(..., description="Angle of attack for design evaluation")
    Ma: float = Field(..., description="Mach number for design evaluation")
    design_iter: int = Field(..., description="Design revision number")


class ReviseEvaluation(BaseTool):
    name: str = "evaluate_revision_tool"
    description: str = (
        "To perform aerodynamic analysis of airfoil geometries generated by NACA Generator Tool2. Export the analysis results as a .csv file"
    )
    args_schema: Type[BaseModel] = ReviseEvaluationInput

    def get_naca_coords(self, max_camber, camber_loc, thickness, n_points=200):
        x_t = anp.cosspace(
            0, 1, n_points
        )  # Generate some cosine-spaced points
        y_t = (
                5
                * thickness
                * (
                        +0.2969 * x_t ** 0.5
                        - 0.1260 * x_t
                        - 0.3516 * x_t ** 2
                        + 0.2843 * x_t ** 3
                        - 0.1015 * x_t ** 4  # 0.1015 is original, #0.1036 for sharp TE
                )
        )

        if camber_loc == 0:
            camber_loc = (
                0.5  # prevents divide by zero errors for things like naca0012's.
            )

        # Get camber
        y_c = np.where(
            x_t <= camber_loc,
            max_camber / camber_loc ** 2 * (2 * camber_loc * x_t - x_t ** 2),
            max_camber
            / (1 - camber_loc) ** 2
            * ((1 - 2 * camber_loc) + 2 * camber_loc * x_t - x_t ** 2),
        )

        # Get camber slope
        dycdx = np.where(
            x_t <= camber_loc,
            2 * max_camber / camber_loc ** 2 * (camber_loc - x_t),
            2 * max_camber / (1 - camber_loc) ** 2 * (camber_loc - x_t),
        )
        theta = np.arctan(dycdx)

        # Combine everything
        x_U = x_t - y_t * np.sin(theta)
        x_L = x_t + y_t * np.sin(theta)
        y_U = y_c + y_t * np.cos(theta)
        y_L = y_c - y_t * np.cos(theta)

        # Flip upper surface so it's back to front
        x_U, y_U = x_U[::-1], y_U[::-1]

        # Trim 1 point from lower surface so there's no overlap
        x_L, y_L = x_L[1:], y_L[1:]

        x = np.concatenate((x_U, x_L))
        y = np.concatenate((y_U, y_L))

        return np.stack((x, y), axis=1)

    def get_aero(self, params, Re, alpha, Ma):
        """Helper to run NeuralFoil and return the full dict."""
        m, p, t = params
        coords = self.get_naca_coords(m, p, t)
        af = asb.Airfoil(name="temp", coordinates=coords)
        # Using small model for speed in constraints, switch to xxlarge for final
        return af.get_aero_from_neuralfoil(alpha=alpha, Re=Re, mach=Ma, model_size='xxxlarge')

    def _run(self, Re, aoa, Ma, design_iter) -> str:
        data_fileloc = f"./data_storage/design_evaluation_revision{design_iter}.csv"
        revised_design_loc = f"./data_storage/airfoil_designs_revision{design_iter}.pkl"

        design_id_ls = []
        CD_ls = []
        CL_ls = []
        CM_ls = []
        max_camber_ls = []
        camber_loc_ls = []
        thickness_ls = []

        with open(f"{revised_design_loc}", "rb") as data:
            airfoil_data = pickle.load(data)
        coord = airfoil_data["coords"]
        design_param = airfoil_data["design_params"]
        design_id = f'revision{design_iter}'
        m = design_param[0]
        p = design_param[1]
        t = design_param[2]

        x0 = [m, p, t]
        aero_data = self.get_aero(x0, Re, aoa, Ma)

        design_id_ls.append(design_id)
        CD_ls.append(aero_data["CD"][0])
        CL_ls.append(aero_data["CL"][0])
        CM_ls.append(aero_data["CM"][0])
        max_camber_ls.append(m)
        camber_loc_ls.append(p)
        thickness_ls.append(t)

        arr = np.concatenate(
            [
                np.stack(design_id_ls)[:, None],
                np.stack(CD_ls)[:, None],
                np.stack(CL_ls)[:, None],
                np.stack(CM_ls)[:, None],
                np.stack(max_camber_ls)[:, None],
                np.stack(camber_loc_ls)[:, None],
                np.stack(thickness_ls)[:, None]
            ],
            axis=1,
        )

        df = pd.DataFrame(
            arr, columns=["design_ID", "CD", "CL", "CM", "max_camber", "camber_loc", "thickness"]
        )
        df.to_csv(data_fileloc, index=False)
        return f"aerodynamic analysis for revised airfoil design Revision#{design_iter} complete"


class OptimizationInputs(BaseModel):
    """
    Input schema for SelectDesignTool
    """

    design_id: str = Field(
        ...,
        description="ID of initial design selected for optimization",
    )
    cl_multipoint_targets: List = Field(
        ...,
        description="multipoint lift targets to be achieved at different angle of attacks",
    )
    re: float = Field(
        ..., description="Reynolds number to be used for aerodynamic analysis"
    )
    ma: float = Field(..., description="Mach number for aerodynamic analysis")


class OptimizationTool(BaseTool):
    name: str = "Optimization_tool"
    description: str = (
        "To optimize a given airfoil shape to meet the objectives and constraints stated in the optimization problem"
    )
    args_schema: Type[BaseModel] = OptimizationInputs

    def _run(self, design_id, cl_multipoint_targets, re, ma):
        airfoil_data_loc = "./data_storage/airfoil_geometry.pkl"
        optimized_design_loc = "./data_storage/optimized_airfoil.txt"
        with open(f"{airfoil_data_loc}", "rb") as data:
            airfoil_data = pickle.load(data)
        if "coords" in airfoil_data:
            coords = airfoil_data["coords"]
            ids = airfoil_data["design_ids"]

        selected_design_indx = ids.index(design_id)
        selected_design_coord = coords[selected_design_indx]
        CL_multipoint_targets = np.array(cl_multipoint_targets)
        CL_multipoint_weights = np.array([2, 3, 4, 4, 4, 5])

        Re = re * (CL_multipoint_targets / 1.25) ** -0.5
        mach = ma

        coordinate_airfoil = asb.Airfoil(
            "dae11"
        )  # Initializing a default airfoil shape
        coordinate_airfoil.coordinates = selected_design_coord
        coordinate_airfoil.name = design_id
        initial_guess_airfoil = coordinate_airfoil.to_kulfan_airfoil()

        opti = asb.Opti()

        optimized_airfoil = asb.KulfanAirfoil(
            name="Optimized",
            lower_weights=opti.variable(
                init_guess=initial_guess_airfoil.lower_weights,
                lower_bound=-5,
                upper_bound=5,
            ),
            upper_weights=opti.variable(
                init_guess=initial_guess_airfoil.upper_weights,
                lower_bound=-5,
                upper_bound=5,
            ),
            leading_edge_weight=opti.variable(
                init_guess=initial_guess_airfoil.leading_edge_weight,
                lower_bound=-1,
                upper_bound=1,
            ),
            TE_thickness=0,
        )

        alpha = opti.variable(
            init_guess=np.degrees(CL_multipoint_targets / (2 * np.pi)),
            lower_bound=-5,
            upper_bound=18,
        )

        aero = optimized_airfoil.get_aero_from_neuralfoil(
            alpha=alpha, Re=Re, mach=mach, model_size="xxxlarge"
        )

        opti.subject_to(
            [
                aero["analysis_confidence"] > 0.90,
                aero["CL"] == CL_multipoint_targets,
                np.diff(alpha) > 0,
                aero["CM"] >= -0.133,
                # optimized_airfoil.local_thickness(x_over_c=0.33) >= 0.128,
                # optimized_airfoil.local_thickness(x_over_c=0.90) >= 0.014,
                optimized_airfoil.TE_angle()
                >= 6.03,  # Modified from Drela's 6.25 to match DAE-11 case
                optimized_airfoil.lower_weights[0] < -0.05,
                optimized_airfoil.upper_weights[0] > 0.05,
                optimized_airfoil.local_thickness() > 0,
            ]
        )

        get_wiggliness = lambda af: sum(
            [
                np.sum(np.diff(np.diff(array)) ** 2)
                for array in [af.lower_weights, af.upper_weights]
            ]
        )

        opti.subject_to(
            get_wiggliness(optimized_airfoil)
            < 2 * get_wiggliness(initial_guess_airfoil)
        )

        opti.minimize(np.mean(aero["CD"] * CL_multipoint_weights))

        sol = opti.solve(
            max_iter=100,
            behavior_on_failure="return_last",
            options={"ipopt.mu_strategy": "monotone", "ipopt.start_with_resto": "yes"},
            # options={"ipopt.mu_strategy": "adaptive", "ipopt.start_with_resto": "yes"},
        )

        optimized_airfoil = sol(optimized_airfoil)
        np.savetxt(f"{optimized_design_loc}", optimized_airfoil.coordinates)
        return f"Optimization process complete with initial design {design_id}, optimized design stored at {optimized_design_loc}"


def get_naca_coords(max_camber, camber_loc, thickness, n_points=200):
    x_t = anp.cosspace(
        0, 1, n_points
    )  # Generate some cosine-spaced points
    y_t = (
            5
            * thickness
            * (
                    +0.2969 * x_t ** 0.5
                    - 0.1260 * x_t
                    - 0.3516 * x_t ** 2
                    + 0.2843 * x_t ** 3
                    - 0.1015 * x_t ** 4  # 0.1015 is original, #0.1036 for sharp TE
            )
    )

    if camber_loc == 0:
        camber_loc = (
            0.5  # prevents divide by zero errors for things like naca0012's.
        )

    # Get camber
    y_c = np.where(
        x_t <= camber_loc,
        max_camber / camber_loc ** 2 * (2 * camber_loc * x_t - x_t ** 2),
        max_camber
        / (1 - camber_loc) ** 2
        * ((1 - 2 * camber_loc) + 2 * camber_loc * x_t - x_t ** 2),
    )

    # Get camber slope
    dycdx = np.where(
        x_t <= camber_loc,
        2 * max_camber / camber_loc ** 2 * (camber_loc - x_t),
        2 * max_camber / (1 - camber_loc) ** 2 * (camber_loc - x_t),
    )
    theta = np.arctan(dycdx)

    # Combine everything
    x_U = x_t - y_t * np.sin(theta)
    x_L = x_t + y_t * np.sin(theta)
    y_U = y_c + y_t * np.cos(theta)
    y_L = y_c - y_t * np.cos(theta)

    # Flip upper surface so it's back to front
    x_U, y_U = x_U[::-1], y_U[::-1]

    # Trim 1 point from lower surface so there's no overlap
    x_L, y_L = x_L[1:], y_L[1:]

    x = np.concatenate((x_U, x_L))
    y = np.concatenate((y_U, y_L))

    return np.stack((x, y), axis=1)

def get_aero(params, Re, alpha, Ma):
    """Helper to run NeuralFoil and return the full dict."""
    m, p, t = params
    coords = get_naca_coords(m, p, t)
    af = asb.Airfoil(name="temp", coordinates=coords)
    # Using small model for speed in constraints, switch to xxlarge for final
    return af.get_aero_from_neuralfoil(alpha=alpha, Re=Re, mach=Ma, model_size='xxxlarge')

# -------------Test for Evaluation tool-----------
# evaluation_tool = DesignEvaluation()
# evaluation_tool._run(Re=5e6, aoa=0)

#-----------Profile plot----------------
# profile_plot = PlotProfile()
# profile_plot._run()

#----------Revised evaluation-----------
# evaluation = ReviseEvaluation()
# evaluation._run(Re=5e6, aoa=0, design_iter=1)

#-----------Spot checking design performances------------
aero_data = get_aero([0.0, 0.3, 0.12], Re=5e6, alpha=0, Ma=0.6)
t = 1
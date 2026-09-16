import pandas as pd
import os
from typing import Optional, List, Dict, Set
from crewai.flow.flow import Flow, router, start, listen, or_, and_
from pydantic import BaseModel
from crews.design_crew.design_crew import (
    DesignCrew,
)
from crews.systems_engineering_crew.systems_engg_crew import (
    SystemsEnggCrew
)
from crewai_files import ImageFile, TextFile
from utils import plot_airfoil


#--------Add your specific LLM API keys here-----------
os.environ['OPENAI_API_KEY'] = ''
os.environ['GEMINI_API_KEY'] = ''

class AirfoilDesignFlowState(BaseModel):
    design_ID: List = []
    sys_engg_feedback: Optional[List] = None
    human_feedback: Optional[str] = None
    retry_count: int = 0
    selection_state: bool = (
        False  # Indicates whether design selection process has been accomplished or not
    )
    design_analysis: List = []  # Design analysis from Systems Engineer
    valid: bool = None
    user_needs: str = ""
    engg_requirements: str = ""


class AirfoilDesignFlow(Flow[AirfoilDesignFlowState]):
    def __init__(self, reqs):
        super().__init__()
        self.user_goals = reqs

    @start()
    def initialize_design_flow(self):
        print("Starting Design flow")
        self.state.user_needs = self.user_goals
        self.state.retry_count = 0
        return 'method_initialized'

    @listen(initialize_design_flow)
    def generate_requirements(self):
        print("Generating requirements")
        # SystemsEnggCrew().crew_requirement().reset_memories(command_type="all")
        result = (
            SystemsEnggCrew()
            .crew_requirement()
            .kickoff(
                inputs={
                    "user_input": self.state.user_needs,
                }
            )
        )

        print("Product requirements generated")
        self.state.engg_requirements = result.raw

        return "tech_requirement_generated"

    @listen(generate_requirements)
    def generate_designs(self):
        print('\n''Creating designs')
        # DesignCrew().crew_design_tasks().reset_memories(command_type="all")
        design_output = (
            DesignCrew()
            .crew_design_tasks()
            .kickoff(
                inputs={
                    "tech_req": self.state.engg_requirements
                }
            )
        )
        return "new_design_generated"

    @listen("suggestions")
    def revise_designs(self):
        print('\n', 'Revising designs')
        response = (
            DesignCrew().crew_revise_design().kickoff(inputs={'feedback': self.state.sys_engg_feedback,
                                                              'tech_requirements': self.state.engg_requirements,
                                                              'design_iter': self.state.retry_count})
        )

    @router(or_(generate_designs, revise_designs))
    def review_designs(self):
        self.state.sys_engg_feedback = []

        if self.state.retry_count < 1:
            print('\n', 'Reviewing initial designs')
            img_folder = "./data_storage"
            data_evaluationloc = "./data_storage/selected_design.csv"  # TODO: Need to modify this part for iterative design segment
            df = pd.read_csv(data_evaluationloc)

            for _, row in df.iterrows():
                local_file_path = os.path.join(img_folder, row["design_ID"] + ".png")
                img_file_path = os.path.abspath(local_file_path)
                img_file = ImageFile(source=img_file_path)
                print("\n", f"Analyzing {row['design_ID']}")
                response = (
                    SystemsEnggCrew()
                    .crew_design_review()
                    .kickoff(
                        inputs={
                            "Design_ID": row["design_ID"],
                            "image_file": img_file,
                            "CD": row["CD"],
                            "CL": row["CL"],
                            "CM": row["CM"],
                            "camber": row["max_camber"],
                            "camber_loc": row["camber_loc"],
                            "thickness": row["thickness"],
                            "engg_requirements": self.state.engg_requirements,
                            "iter_num": self.state.retry_count,
                        }
                    )
                )
                self.state.sys_engg_feedback.append(response["Feedback"])

            self.state.retry_count += 1

        else:
            img_folder = "./data_storage"
            data_evaluationloc = f"./data_storage/design_evaluation_revision{self.state.retry_count}.csv"
            df = pd.read_csv(data_evaluationloc)
            print("\n", "Reviewing final design")

            for _, row in df.iterrows():
                local_file_path = os.path.join(img_folder, f'revision_{self.state.retry_count}' + ".png")
                img_file_path = os.path.abspath(local_file_path)
                img_file = ImageFile(source=img_file_path)
                print("\n", f"Analyzing revision_{self.state.retry_count}")
                response = (
                    SystemsEnggCrew()
                    .crew_design_review()
                    .kickoff(
                        inputs={
                            "Design_ID": row["design_ID"],
                            "image_file": img_file,
                            "CD": row["CD"],
                            "CL": row["CL"],
                            "CM": row["CM"],
                            "camber": row["max_camber"],
                            "camber_loc": row["camber_loc"],
                            "thickness": row["thickness"],
                            "engg_requirements": self.state.engg_requirements,
                            "iter_num": self.state.retry_count
                        }
                    )
                )
                self.state.sys_engg_feedback.append(response["Feedback"])
                self.state.valid = response['Valid']

            self.state.retry_count += 1
        if self.state.valid or self.state.retry_count > 3:
            return "review_complete"
        else:
            return "suggestions"

    @listen("review_complete")
    # @listen("tech_requirement_generated")
    def optimize_design(self):
        print('All processes completed')
        return self.state.sys_engg_feedback


def kickoff():
    with open("design_reqs.txt", "r") as reqs:
        design_requirements = reqs.read()
    design_flow = AirfoilDesignFlow(design_requirements)
    final_output = design_flow.kickoff()
    print(f"Final output: {final_output}")

if __name__ == "__main__":
    kickoff()

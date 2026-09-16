import shutil
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource
from tools_rev2.NACA_profile_generator import (
    NACAProfileGenerator,
    PlotProfile,
    GeometrySampler,
    DesignEvaluation,
    SelectDesignTool,
    ReviseNACAProfile,
    RevisePlotProfile,
    ReviseEvaluation,
    # OptimizationTool
)
from typing import List
from crewai_tools import VisionTool, CSVSearchTool
from natsort import natsorted
from pydantic import BaseModel, Field


class OptimizationOutput(BaseModel):
    design_ID: str = Field(
        ...,
        description="Final design ID selected for optimization process such as ID-1",
    )
    feedback: str = Field(
        ...,
        description="Reason for selecting the Design ID for optimization with details",
    )
    optimized_airfoil_loc: str = Field(
        ..., description="Location where coordinates of optimized design is saved"
    )


@CrewBase
class DesignCrew:
    """Design and andlysis crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self) -> None:

        # shutil.rmtree(".crewai_storage", ignore_errors=True)
        # shutil.rmtree(".crew_data", ignore_errors=True)
        self.knowledge_base_designer = CSVKnowledgeSource(
            file_paths="graph_subset_design_engg.csv",
            collection_name="design_engg_knowledge7",
        )
        # self.llm = LLM(model="ollama/llama3.2-vision:11b")
        # self.vision_tool = VisionTool(model="llama3.2-vision:11b", llm=self.llm)
        # self.llm_general = LLM(
        #     model="ollama/qwen3:14b", base_url="http://localhost:11434"
        #     # model="ollama/mistral-small3.2:24b", base_url="http://localhost:11434"
        # )
        # self.llm_general = LLM(model="openai/gpt-4o", temperature=0.2)
        # self.llm_general = LLM(model="ollama/gemma4:latest", base_url="http://localhost:11434")
        # self.llm_general = LLM(model="gemini/gemini-2.5-pro")
        # self.llm_general = LLM(model="gemini/gemini-3.1-pro-preview")
        # self.llm_general = LLM(model="openai/gpt-5")
        self.llm_general = LLM(model="openai/gpt-5")
        # self.llm_general = LLM(model="openai/gpt-5.4")

    @agent

    def airfoil_designer(self) -> Agent:
        return Agent(
            config=self.agents_config["airfoil_designer"],
            llm=self.llm_general,
            tools=[
                GeometrySampler(),
                NACAProfileGenerator(),
                PlotProfile(),
                DesignEvaluation(),
                SelectDesignTool(),
                # OptimizationTool(),
                ReviseNACAProfile(),
                RevisePlotProfile(),
                ReviseEvaluation(),
            ],
            verbose=False
        )

    @task
    def sample_designs(self) -> Task:
        return Task(
            config=self.tasks_config["sample_designs"],
        )

    @task
    def design_airfoil(self) -> Task:
        return Task(
            config=self.tasks_config["design_airfoil"], context=[self.sample_designs()]
        )

    @task
    def visualize_airfoil(self) -> Task:
        return Task(config=self.tasks_config["visualize_airfoil"])

    @task
    def analyze_airfoil(self) -> Task:
        return Task(
            config=self.tasks_config["analyze_airfoil"], context=[self.design_airfoil()]
        )

    @task
    def select_design(self) -> Task:
        return Task(
            config=self.tasks_config["select_design"],
            context=[self.design_airfoil(), self.analyze_airfoil()],
        )

    @task
    def revise_design(self) -> Task:
        return Task(config=self.tasks_config["revise_design"])

    @task
    def revise_plot(self) -> Task:
        return Task(config=self.tasks_config["revise_plot"])

    @task
    def revise_analysis(self) -> Task:
        return Task(config=self.tasks_config["revise_analysis"])

    @crew
    def crew_design_tasks(self) -> Crew:
        """Creates the designing crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=[
                self.sample_designs(),
                self.design_airfoil(),
                self.visualize_airfoil(),
                self.analyze_airfoil(),
                self.select_design(),
            ],
            # tasks=self.tasks,         # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=False,
            # memory=True,
            # knowledge_sources=[self.knowledge_base_designer],
            # embedder={
            #     "provider": "ollama",
            #     "model_name": "nomic-embed-text:v1.5",
            #     # "model_name": "embeddinggemma:300m",
            #     "url": "http://localhost:11434/api/embeddings",
            # }
        )

    @crew
    def crew_revise_design(self) -> Crew:
        """Creates the design revision crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=[self.revise_design(), self.revise_plot(), self.revise_analysis()],
            # tasks=self.tasks,         # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=False,
            # memory=True,
            knowledge_sources=[self.knowledge_base_designer],
            # embedder={
            #     "provider": "ollama",
            #     "model_name": "nomic-embed-text:v1.5",
            #     # "model_name": "embeddinggemma:300m",
            #     "url": "http://localhost:11434/api/embeddings",
            # }
        )
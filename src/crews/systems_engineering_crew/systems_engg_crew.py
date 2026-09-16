from typing import Optional
import os
import shutil
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
import pandas as pd

from pydantic import BaseModel, Field
from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource
from crewai_tools import VisionTool, CSVSearchTool
from natsort import natsorted


class DesignVerification(BaseModel):
    Design_ID: str = Field(
        ...,
        description="Filename extracted from the image that represents the ID# of each design such as ID_1, ID_2 etc",
    )
    Feedback: str = Field(
        ...,
        description="Analysis of airfoil profile, aerodynamic performance data and geometric parameters including suggestions for improving the design by modification of geometric features",
    )
    Valid: bool = Field(
        ...,
        description="True/False indicating whether the design is suitable for meeting the requirements or not",
    )


class ImageAnalysis(BaseModel):
    Design_ID: str = Field(
        ...,
        description="Filename extracted from the image that represents the ID# of each design such as ID_1, ID_2 etc",
    )
    Feedback: str = Field(
        ...,
        description="Analysis of airfoil profile based on provided image",
    )


class Requirements(BaseModel):
    Functional_req: str = Field(
        ...,
        description="Bullet-point list of functional requirements generated from user request",
    )
    Non_functional_req: str = Field(
        ...,
        description="Bullet-point list of non-functional requirements for airfoil design",
    )


@CrewBase
class SystemsEnggCrew:
    """Design review crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self) -> None:
        # Initialize the LLM with Ollama using Mistral
        # self.llm = LLM(model="ollama/gemma3:12b", base_url="http://localhost:11434")
        # self.llm = LLM(model="ollama/gemma3:12b", base_url="http://localhost:11434")
        # shutil.rmtree(".crewai_storage", ignore_errors=True)
        # shutil.rmtree(".crew_data", ignore_errors=True)
        # self.llm_ollama = LLM(model="ollama/llama3.2-vision:11b")
        # self.llm_general = LLM(model="ollama/gemma4:latest", base_url="http://localhost:11434")
        # self.llm_general = LLM(model="openai/gpt-5-mini")
        self.llm_general = LLM(model="gemini/gemini-2.5-pro")
        # self.llm_general = LLM(model="gemini/gemini-3.1-pro-preview")
        # self.llm_general = LLM(model="openai/gpt-5")
        # self.llm_general = LLM(model="openai/gpt-5.4")
        # self.vision_tool = VisionTool()
        # self.vision_tool = VisionTool(model='gpt-5')
        # self.llm_general = LLM(model="openai/gpt-5-mini")
        self.knowledge_base_sysengg = CSVKnowledgeSource(
            file_paths="graph_subset_systems_engg.csv",
            collection_name="sys_engg_knowledge7",
        )

    @agent
    def systems_engineer(self) -> Agent:
        return Agent(
            config=self.agents_config["systems_engineer"],
            verbose=True,
            llm=self.llm_general,
            respect_context_window=True,
            multimodal=True,
            # tools=[self.vision_tool]
        )

    @task
    def create_requirements(self) -> Task:
        return Task(
            config=self.tasks_config["create_requirements"],
            output_pydantic=Requirements,
        )

    # @task
    # def analyze_profile(self) -> Task:
    #     return Task(
    #         config=self.tasks_config["analyze_profile"],
    #         output_pydantic=ImageAnalysis,
    #     )

    @task
    def verify_designs(self) -> Task:
        return Task(
            config=self.tasks_config["verify_designs"],
            output_pydantic=DesignVerification,
            # context=[self.analyze_profile()]
        )

    # @task
    # def verify_designs_noimg(self) -> Task:
    #     return Task(
    #         config=self.tasks_config["verify_designs_noimg"],
    #         output_pydantic=DesignVerification,
    #         # context=[self.analyze_profile()]
    #     )

    @crew
    def crew_requirement(self) -> Crew:
        """Creates a crew to analyze user request and create requirements"""

        return Crew(
            agents=self.agents,
            tasks=[self.create_requirements()],
            process=Process.sequential,
            verbose=False,
            # memory=False,
            # cache=True,
            knowledge_sources=[self.knowledge_base_sysengg],
            # embedder={
            #     "provider": "ollama",
            #     "config": {"model": "nomic-embed-text:v1.5",
            #                "url":"http://localhost:11434/api/embeddings"},
            #     # "config": {"model": "embeddinggemma:300m"},
            # },
        )

    @crew
    def crew_design_review(self) -> Crew:
        """Creates a crew to analyze and critique airfoil design, shapes and aerodynamic performance"""

        return Crew(
            agents=self.agents,
            tasks=[self.verify_designs()],
            process=Process.sequential,
            verbose=False,
            # memory=False,
            # cache=True,
            knowledge_sources=[self.knowledge_base_sysengg],
            # embedder={
            #     "provider": "ollama",
            #     "model_name": "nomic-embed-text:v1.5",
            #     # "model_name": "embeddinggemma:300m",
            #    "url":"http://localhost:11434/api/embeddings",
            # }
        )

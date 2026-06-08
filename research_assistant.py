from pydantic import BaseModel, Field
from IPython.display import Image, display
from typing import Annotated, TypedDict, List, Dict, Any, Literal
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os
import requests
import uuid # for memory
from tools import playwright_tools, google_tool, reporter_tools
from datetime import datetime

# some structured outputs
class ResearchPlan(BaseModel):
  objective: str=Field(description="a refined research goal created from the user's query")
  key_topics: list[str]=Field(description="a list of topics to research relevant to the objective")

class EvaluatorOutput(BaseModel):
    rework_feedback: str = Field(description="Feedback on the assistant's response")
    criteria_met: bool = Field(description="Whether the success criteria have been met")

class RouteDecisions(BaseModel):
    route: Literal[
        "new_research",
        "generate_markdown",
        "generate_pdf"
    ]
    reasoning:str

class State(TypedDict):
  messages: Annotated[List, add_messages]
  research_plan: ResearchPlan
  research_result: str
  success_criteria: str
  criteria_met: bool
  rework_feedback: str
  route: str
  #status: str


class ResearchAssistant:
    def __init__(self):
        self.graph = None
        self.memory = MemorySaver()
        self.assistant_id = str(uuid.uuid4())
        self.routing_llm_with_output = None
        self.planner_llm_with_output = None
        self.research_llm_with_tools = None
        self.evaluator_llm_with_output = None
        self.search_tools = None
        self.file_tools = None
        self.playwright = None
        self.browser = None


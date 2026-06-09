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
import asyncio
import uuid # for memory
from tools import playwright_tools, google_tool, reporter_tools
from datetime import datetime

# you need this to call APIs
load_dotenv(override=True)
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_user = os.getenv("PUSHOVER_USER")
pushover_url = "https://api.pushover.net/1/messages.json"

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
        self.report_llm_with_tools = None
        self.search_tools = None
        self.reporter_tools = None
        self.playwright = None
        self.browser = None

    async def setup(self):
        self.search_tools, self.browser, self.playwright = await playwright_tools()
        self.search_tools += [google_tool]
        self.reporter_tools  = reporter_tools
        routing_llm = ChatAnthropic(model="claude-sonnet-4-5")
        self.routing_llm_with_output  = routing_llm.with_structured_output(RouteDecisions)
        planner_llm = ChatOpenAI(model="gpt-5")
        self.planner_llm_with_output = planner_llm.with_structured_output(ResearchPlan)
        research_llm = ChatOpenAI(model="gpt-5.4")
        self.research_llm_with_tools= research_llm.bind_tools(self.search_tools)
        evaluator_llm = ChatAnthropic(model="claude-sonnet-4-5")
        self.evaluator_llm_with_output = evaluator_llm.with_structured_output(EvaluatorOutput)
        report_llm=ChatOpenAI(model="gpt-4.1")
        self.report_llm_with_tools = report_llm.bind_tools(self.reporter_tools)
        await self.build_graph()

    
    def intent_checker(self, state:State) -> Dict[str, Any]:
        latest_user_msg = state["messages"][-1].content
        has_research_result  = bool(state.get("research_result")) # bool turns None into boolean
        #final_criteria_met = state.get("criteria_met", False) #already a boolean value
        
        system_message = f"""
        You are a helpful assistant that verifies the user's request and decides the purpose of it.
        For example:
        - if the user requests researching on topics, such as 'give me the history of huskies', then they need the research
        to be conducted, and you should define the intent as a "new_research".
        - if the user requests research a topic as well as generating a report for the results, then the intent should also be "new_research".
        - if a user's request says something along the lines of 'generate a report of that' without any request for a research topic, then they probably want
        to get a report file for the research they had conducted already. In that case you should define the intent as either
        "generate_markdown" or "generate_pdf" based on the format of file requested.
        Rules:
        - If there is no existing research result, choose new_research.
        - If there is existing research results but the user's query requested a new topic of research, choose new_research.
        - If the user asks to export, save, download, convert, or produce a file from the existing report, choose the matching export route.
        - Do not perform the task. Only classify the route.
        """
        user_message = f"""
        You are deciding best route to handle the incoming request base on the following information:
        - latest user message: {latest_user_msg}
        - existing research available:  {has_research_result}
        """

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message)
        ]

        decision = self.routing_llm_with_output.invoke(messages)
        print(f"routing llm response: {decision}")
        return{
            "messages" : [AIMessage(content=f"The intention for this run is {decision.route} and the reasoning is {decision.reasoning}")],
            "route" : decision.route
        }


    def intent_router(self, state:State) ->str:
     route = state.get("route", "new_research")
     if route == "new_research":
          return "new_research"
     else:
          return "generate_file"

    def planner(self, state:State) -> Dict[str, Any]: #means keys are strings and values can be anything
        system_message = f"""You are a helpful assistant specialized in research planning. You take the user's research query
            and decide on a list of key topics to research. 
            You make sure the research topics are relevant to the current times and provide useful insghts.
            The current date and time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            """
        user_message = f"""
        You are to plan the research on the user's newest request {state["messages"][-2].content}
        """

        messages = [SystemMessage(content=system_message), HumanMessage(content=user_message)]
        response = self.planner_llm_with_output.invoke(messages)
        print(f"Planner node content:  {response}")
        new_message = [AIMessage(content=f"Generated research plan: {response.objective}")]
        return  {
            "messages":new_message,
            "research_plan": response,
            "status" : "Planning research strategy..."
        }


    def researcher(self, state:State) -> str:
        system_message = f"""You are a seasoned researcher who specialized in deep research using search tools. 
        The research plan is {state['research_plan']} which includes the research objective and a list of topics to research on.
        You MUST use either or both of these tools to help you: "browser_tools" and "google_tool".
        This is the success criteria: {state['success_criteria']}
        When you have the research results already you need to go ahead and write the research report without further tool calls
        You should produce a research report that is concise, accurate, up-to-date and neatly written. 
        Only output the research report itself in a markdown format and do not wrap it in code blocks or add extra explanation.
        Keep in mind that the result of your research will be evaluated by an evaluator based on accuracy and depth using the criteria,
        so you will do the best you can to pass the evaluation.
        """

        # human_message = f"""You are performing a deep research based on the research plan {state['research_plan']} which includes the research objective and a list of topics to research on.
        # This is the success criteria: {state['success_criteria']}
        # """
        if state.get("rework_feedback"):
            system_message += f"""
            Previously you thought you completed the research assignment, but your reply was rejected because the success criteria was not met.
            Here is the feedback on why this was rejected:
            {state['rework_feedback']}
            With this feedback, please continue the task and rework your research report to ensure that you meet the success criteria.
            """
    
        found_system_message = False
        messages = state["messages"]
        for message in messages: # try to find if a system message already exist cuz this execution might be after the tool call, meaning system_message already exist
            if isinstance(message, SystemMessage):
                message.content = system_message #just replace that first message
                found_system_message = True
                break
        if not found_system_message: # if first time here, create our own, and append the messages
            messages = [SystemMessage(content=system_message)] + messages
            # why put system message before the messages thread?
            # because instructions at the tails are weaker; some models treat early messages as the main thread
            # and barely weight a late system block

        response = self.research_llm_with_tools.invoke(messages)
        #print(f"research node content:{response.content}")
        if response.tool_calls:
            return {
                "messages" : [response],
                "status" : "Researcher requested tool use..."
            }
        
        return{
            "messages": [response], #necessary to return the whole response to preseve the too-call flow
            "research_result": response.content,
            "status": "Research report drafted..."
        }


    def researcher_router(self, state: State) -> str:
        last_message = state["messages"][-1] #output from researcher
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "evaluator"

    
    def evaluator(self, state:State) -> Dict[str, Any]:
        system_message = f"""You are an evalutor that determines if a deep research report on a given topic has been done
        adequately and accurately by the researcher. You assess the report based on the given success criteria. 
        Respond with your decision on whether the success criteria has been met.
        If the report has failed to pass the evaluation, you should provide a 2-3 line feedback on rework as well so the researcher can improve the report.
        if the report has passed evaluation, do not provide feedback.
        """

        human_message = f"""You are evaluating the research report {state['research_result']} using the success criteria {state['success_criteria']}.
        You are judging whether the report is adequately and accurately done based on the research plan {state['research_plan']}.
        """

        if state.get("rework_feedback"):
            human_message += f"Also, note that in a prior attempt from the researcher, you provided this feedback: {state["rework_feedback"]}"
            human_message += f"See if the research result has been improved based on the previous feedback."
        
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=human_message)
        ]
        eval_result=self.evaluator_llm_with_output.invoke(messages) #structured output EvaluatorOutput
        return{
            "messages": [{"role": "assistant", "content": f"Report passed evaluation: {eval_result.criteria_met}"}],
            "criteria_met": eval_result.criteria_met,
            "rework_feedback": eval_result.rework_feedback or None,
            "status": "Evaluating research..."
        }


    def evaluator_router(self, state:State) -> str:
        if state["criteria_met"]:
            return "convert"
        else:
            return "researcher"



    def reporter(self, state:State) -> Dict[str, list]:

     # if tools have already been run, then return directly. 
     # if node doesn't detect a tool was just called, it may keep running tool_calls
     # Stop condition: if we just came back from a tool execution, end the loop.
     
     # handles when the tool call comes back to this node
     #error handling: if error then report as is.
     # this handles if after tool call
        last_msg = state["messages"][-1]
        if isinstance(last_msg, ToolMessage):
            tool_output = last_msg.content or ""
            if tool_output.strip().lower().startswith("error"):
                return {
                        "messages": [
                            AIMessage(
                                content=(
                                    f"Report file generation failed: {tool_output}. "
                                    "Please retry or use write_file for markdown instead."
                                )
                            )
                        ],
                        "status": "Report generation failed", 
                }
            
            return {
                "messages": [
                        AIMessage(content=f"Report generated successfully. Tool output: {tool_output}")
                ],
                "status": "Report generation complete and push notification sent",
            }
        
        # the part below handles when we just arrived at this node before tools
        system_message = f"""
        NEVER do your own research.
        You are an assistant responsible for handling the final research report using available tools.
        You have a tool "write_file" for writing the final research result to a markdown report.
        You have another tool "convert_to_pdf" for writing the final research result to a pdf file.
        You will decide whether to use these tools based on the user request, and which tool to use depending on the format they requested.
        """

        # Always use the latest user request from the current thread, not the first one.
        latest_user_query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                latest_user_query = msg.content
                break

        human_message=f"""
        You are responsible for reporting the research result.
        DO NOT PERFORM THE RESEARCH. Only use the available research result {state['research_result']}.
        The latest user request is: {latest_user_query}
        If the user DID NOT request to generate any report file, DO NOT use markdown or pdf tools.
        If the request asks for markdown, use write_file.
        If the request asks for pdf, use convert_to_pdf.
        If both are requested, do both.
        """
        
        # if node doesn't detect a tool was just called, it may keep running tool_calls
        # Stop condition: if we just came back from a tool execution, end the loop.
    
        messages=[
            SystemMessage(content=system_message),
            HumanMessage(content=human_message)
        ]
        response = self.report_llm_with_tools.invoke(messages) 
        print(f"Reporter node: response is {response}")

        #new_message = AIMessage(content="File generated")
        return {
            "messages": [response], #need to check if there's "tool_call"
            "status": "Research completed"
        }

    def reporter_router(self, state: State) -> str:
        last_message=state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "push"

    def push(self, text: str):
     """Useful when you want to send a push notification to the user"""
     requests.post(pushover_url, data = {"token": pushover_token, "user": pushover_user, "message": text})

    def notifier(self, state:State) -> Dict[str, Any]:
        notification = "Report is ready"
        self.push(notification)
        return{
            "messages":[AIMessage(content="Push notification has been sent")]
        }


    async def build_grraph(self):
        graph_builder = StateGraph(State)
        # nodes
        graph_builder.add_node("intent_checker", self.intent_checker)
        graph_builder.add_node("planner", self.planner)
        graph_builder.add_node("researcher", self.researcher)
        graph_builder.add_node("search_tools", ToolNode(tools=self.search_tools))
        graph_builder.add_node("evaluator", self.evaluator)
        graph_builder.add_node("reporter", self.reporter)
        graph_builder.add_node("reporter_tools", ToolNode(tools=self.reporter_tools))
        graph_builder.add_node("notifier", self.notifier)

        #edges
        graph_builder.add_edge(START, "intent_checker")
        graph_builder.add_conditional_edges("intent_checker", self.intent_router, {"new_research": "planner", "generate_file":"reporter"})
        graph_builder.add_edge("planner", "researcher")
        graph_builder.add_conditional_edges("researcher", self.researcher_router, {"tools":"search_tools", "evaluator": "evaluator"})
        graph_builder.add_edge("search_tools", "researcher")
        graph_builder.add_conditional_edges("evaluator", self.evaluator_router, {"convert":"reporter", "researcher":"researcher"})
        graph_builder.add_conditional_edges("reporter", self.reporter_router, {"tools":"reporter_tools", "push": "notifier"})
        graph_builder.add_edge("reporter_tools", "reporter")
        graph_builder.add_edge("notifier", END)

        #compile the graph
        
        self.graph=graph_builder.compile(checkpointer=self.memory)


    #this is the running of the graph in gradio UI (aka the callback used by gradio)
    async def run_superstep(self, message, success_criteria, history):
        config={"configurable":{"thread_id": self.assistant_id}}

        state={
            "messages": [HumanMessage(content=message)],  #from user input
            "success_criteria": success_criteria, #user input
        }

        result = await self.graph.ainvoke(state, config=config)

        user={"role":"user", "content":message}
        reply={"role":"assistant", "content": "Research report generated"} #hardcore it here
        history += [user, reply]
        research_objective=result['research_plan'].objective
        research_topics=result['research_plan'].key_topics
        #return history in the chat
        return(
            history,
            research_objective, #lets just return the objective for now
            result["rework_feedback"],
            result["research_result"]
        )
    
    
    
    def cleanup(self):
        if self.browser:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.browser.close())
                if self.playwright:
                    loop.create_task(self.playwright.stop())
            except RuntimeError:
                # If no loop is running, do a direct run
                asyncio.run(self.browser.close())
                if self.playwright:
                    asyncio.run(self.playwright.stop())  

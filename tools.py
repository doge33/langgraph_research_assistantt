from langchain.agents import Tool
from langchain_community.agent_toolkits import FileManagementToolkit, PlayWrightBrowserToolkit
from langchain_community.utilities import GoogleSerperAPIWrapper
from dotenv import load_dotenv
from playwright.async_api import async_playwright
from markdown_pdf import md_to_pdf



# This file contains different tools: Playwright and others
# PlayWright is special because it returns objects(browser, playwright) that needs to be cleaned up later
# PW is also heavier than other tools
# 1. playwright
async def playwright_tools():
    # steps: start playwright, launch chromium
    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless=False)
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
    return toolkit.get_tools(), browser, playwright # return these for cleaning

# 2. serper/google, to be used for researcher toghether with playwright
serper = GoogleSerperAPIWrapper()
google_tool = Tool(
    name="google",
    func=serper.run,
    description="Use this tool when you want to get the results of an online web search"
)

# 3. 2 tools for reporter
def get_file_tools():
    toolkit=FileManagementToolkit(root_dir="markdown_reports")
    return toolkit.get_tools()
markdown_tool=get_file_tools()


pdf_tool=Tool(
    name="convert_to_pdf",
    func=md_to_pdf,
    description="Useful when you want to generate a pdf file from markdown content"
)

reporter_tools = markdown_tool + [pdf_tool]

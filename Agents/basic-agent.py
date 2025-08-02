# LLM Agent for Markdown Report Analysis
import re
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq
import pandas as pd
from gen_ai_hub.proxy.langchain.init_models import init_llm
from creds import env_vars
from collections import ChainMap
import os
import re  
os.environ.update(env_vars)


class MarkdownAgent:
    def __init__(self, md_path):
        with open(md_path, 'r', encoding='utf-8') as f:
            self.content = f.read()

    def summarize(self, max_lines=10):
        """
        Simple summarization: returns the first N non-empty lines as a summary.
        Replace with LLM call for advanced summarization.
        """
        lines = [line.strip() for line in self.content.splitlines() if line.strip()]
        return '\n'.join(lines[:max_lines])

    def find_text(self, query):
        """
        Finds all occurrences of the query string in the markdown report.
        Returns a list of matching lines with context.
        """
        matches = []
        for i, line in enumerate(self.content.splitlines()):
            if query.lower() in line.lower():
                # Add previous and next line for context
                prev = self.content.splitlines()[i-1] if i > 0 else ''
                next_ = self.content.splitlines()[i+1] if i+1 < len(self.content.splitlines()) else ''
                matches.append({'line': line, 'prev': prev, 'next': next_})
        return matches

    def extract_code_output_pairs(self):
        """
        Extracts code blocks and their respective outputs from the markdown report.
        Returns a list of dicts: { 'code': ..., 'output': ... }
        """
        pattern = re.compile(r'### Code Block (\d+)\n```python\n(.*?)```\n\n#### Output:\n((?:```\n.*?```\n\n|\|.*?\n\n)?)', re.DOTALL)
        pairs = []
        for match in pattern.finditer(self.content):
            code = match.group(2).strip()
            output = match.group(3).strip()
            # Clean output: remove code block markers if present
            if output.startswith('```'):
                output = output.strip('`\n')
            pairs.append({'code': code, 'output': output})
        return pairs

# Initialize your MarkdownAgent
md_agent = MarkdownAgent("submission_report-beta.md")

# Define LangChain tools
summarize_tool = Tool(
    name="SummarizeMarkdown",
    func=lambda _: md_agent.summarize(),
    description="Summarizes the markdown report."
)

find_text_tool = Tool(
    name="FindTextInMarkdown",
    func=lambda query: str(md_agent.find_text(query)),
    description="Finds all occurrences of a query string in the markdown report."
)

extract_code_output_tool = Tool(
    name="ExtractCodeOutputPairs",
    func=lambda _: str(md_agent.extract_code_output_pairs()),
    description="Extracts code blocks and their respective outputs from the markdown report."
)

tools = [summarize_tool, find_text_tool, extract_code_output_tool]

# Set up Groq Llama 3 LLM
## TODO: replace this with more powerfull LLM with more token limit and better performance (gpt4, 40, 4.1, etc.)
# llm = ChatGroq(
#     groq_api_key="",  # Replace with your Groq API key
#     model_name="llama3-70b-8192"       # Or "llama3-8b-8192" for smaller model
# )


llm = init_llm('gpt-4o', temperature=0.1, max_tokens=10000)

# Initialize the agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)


# Interactive loop: always use tools for markdown-based answers
def run_interactive_agent():
    print("Markdown LLM Agent (Groq Llama 3)\nType 'exit' to quit.")
    while True:
        prompt = input("\nAsk a question about the markdown report: ")
        if prompt.strip().lower() in ["exit", "quit"]:
            print("Exiting agent.")
            break
        # Encourage agent to use tools for file-based answers
        tool_prompt = f"Use the available tools to answer: {prompt}"
        try:
            result = agent.run(tool_prompt)
            print("\nResult:\n", result)
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    run_interactive_agent()

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

# ✅ STEP 1: Force-load the .env file from the project root
# This ensures the key loads even if Streamlit or PyCharm runs from another folder
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_PATH = os.path.join(BASE_DIR, ".env")

print(f"🔍 Looking for .env at: {ENV_PATH}")

if os.path.exists(ENV_PATH):
    load_dotenv(ENV_PATH)
    print("✅ .env file loaded successfully.")
else:
    print("❌ .env file NOT FOUND at:", ENV_PATH)

# ✅ STEP 2: Confirm that the key is being loaded
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("⚠️ Warning: GROQ_API_KEY not found in environment variables.")
else:
    print(f"🔑 Loaded GROQ_API_KEY: {api_key[:8]}*********")

# ✅ STEP 3: Define the Chain class
class Chain:
    def __init__(self):
        if not api_key:
            raise ValueError("❌ GROQ_API_KEY is missing. Please check your .env file.")
        # Initialize LLM connection
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=api_key,
            model_name="llama-3.1-8b-instant"
        )

    def extract_jobs(self, cleaned_text):
        """Extract job postings as JSON from a scraped webpage (robust version)."""
        import re, json

        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}

            ### INSTRUCTION:
            The text above is scraped from a company's career page.
            Extract *all* job postings and return them strictly in JSON format.

            Each posting must contain these keys:
            - role
            - experience
            - skills
            - description

            Respond with ONLY JSON — no explanations, no markdown, no notes.
            Example:

            [
                {{
                    "role": "Software Engineer",
                    "experience": "2+ years",
                    "skills": ["Python", "AI", "NLP"],
                    "description": "Responsible for developing AI-driven applications."
                }}
            ]
            """
        )

        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke({"page_data": cleaned_text})
        raw_output = res.content.strip()

        # ✅ Step 1: Extract only the JSON-like part
        json_pattern = r'(\[.*\]|\{.*\})'
        matches = re.findall(json_pattern, raw_output, re.DOTALL)

        if not matches:
            raise OutputParserException("No valid JSON found in model output.")

        # ✅ Step 2: Combine multiple JSON blocks safely
        json_text = matches[0]  # start with first match
        if raw_output.count("{") > 1 and not raw_output.strip().startswith("["):
            # Model returned multiple dicts -> wrap them into a list
            json_text = "[" + ",".join(matches) + "]"

        # ✅ Step 3: Cleanup markdown or garbage
        json_text = json_text.replace("```json", "").replace("```", "").strip()

        # ✅ Step 4: Try parsing safely
        try:
            parsed = json.loads(json_text)
            return parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError as e:
            print("⚠️ Raw Output:", raw_output[:1500])
            raise OutputParserException(f"Invalid JSON returned: {e}")

    def write_mail(self, job, links):
        """Generate a professional cold email using job data and portfolio links."""
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Mohan, a business development executive at AtliQ — an AI & Software Consulting firm.
            Write a cold email to the client regarding the job mentioned above,
            showcasing AtliQ's capability to deliver similar solutions.
            Add the most relevant portfolio links from: {link_list}.
            Do NOT include any preamble or explanation — only the email content.
            """
        )
        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content


# ✅ Test block — run only when executing directly
if __name__ == "__main__":
    print("\n🔁 Testing environment loading...")
    print("GROQ_API_KEY (from environment):", os.getenv("GROQ_API_KEY"))

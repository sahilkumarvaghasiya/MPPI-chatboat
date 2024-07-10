# Required imports
from fastapi import FastAPI, Request
from langserve import add_routes
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
# from accelerate import init_empty_weights, load_checkpoint_in_model, dispatch_model, BigModel

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import T5Tokenizer, T5ForConditionalGeneration
import uvicorn
import os
from dotenv import load_dotenv
# import uvicorn
# Load environment variables
load_dotenv()
# huggingface_api_token = os.getenv("HUGGING_FACE_KEY")
# Initialize FastAPI app
app = FastAPI(
    title="MPPI app",
    description="This is a great app for mathematicians, Q&A from images and PDFs, programming",
    version="0.1"
)

# Set random seed for reproducibility
torch.random.manual_seed(0)

# Load model and tokenizer with accelerate configurations
model_name = "microsoft/Phi-3-mini-128k-instruct"

# Ensure accelerate is installed
# !pip install accelerate


# Load the model with low CPU memory usage and device mapping
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create the pipeline
pipe = pipeline(
    task='text-generation',
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map='auto',
    max_new_tokens=512,
    min_new_tokens=-1,
    # top_k=30
    do_sample=True
)

# Create the LLM
llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={'temperature': 0})

# Define a chat prompt template
prompt1 = ChatPromptTemplate.from_template("You are an expert. Tell them Hi! How are you? How can I assist you? provide me an essay about {topic}")

# Add routes to the FastAPI app
add_routes(
    app,
    prompt1|llm,
    path="/home"
)

# # # Run the FastAPI app
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

@app.post("/home/invoke")
async def invoke(request: Request):
    data = await request.json()
    print(f"Received input: {data}")
    output = prompt1(data)
    print(f"Generated output: {output}")
    return {"output": output}

if __name__ == "__main__":
    uvicorn.run("mainapp", host="0.0.0.0", port=8000, reload=True)
## JPery's LLM Airline Policy App

### Features

Chatbot application that can answer questions about airline policies. The application has the following features:

1. **User Interface**: A simple web interface where users can input questions and get answers has been created using plan HTML, CSS and JS. Connection with the agent has been developed via WebSockets using FastAPI.
2. **LLM Integration**: A pre-trained LLM model (gpt-4.1-mini was selected as it provides a nice price to performance ratio) is used to understand and answer questions. In order to improve its performance it has been given a custom System Prompt which can be located at `agent/constants.py`.
3. **Document Processing**: In order to extract text from the policy documents, a script `policy_parser.py` was created. It uses `pypdf` in order to parse PDF files.
4. **Vector Database**: In order to do an efficient similarity search, an Hybrid Retriever (70% dense, 30% sparse) has been developed using the model `sentence-transformers/all-MiniLM-L6-v2` as Dense Retriever and a `BM25` as Sparse Retriever. Pinecone has been used as vector database in order to store all the generated embeddings.

## Installation

Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Parser

In order to run the scrapper to get the data from policies, you have to run the `policy_parser.py` script. This script will parse the files and convert them into a custom format inside a `parsed_policies` folder.

```bash
python policy_parser.py
```

This script can take a while to run, depending on the amount of data to be scraped. It will create a directory named `parsed_policies` containing the parsed data.

### Agent

You have to provide your OpenAI API and Pinecone API keys by setting the environment variables `OPENAI_API_KEY`, `OPENAI_API_URL` and `PINECONE_API_KEY`. You can do this in your terminal or command prompt:

```bash
export OPENAI_API_KEY='your_openai_api_key'
export OPENAI_API_URL='your_openai_api_proxy_url'
export PINECONE_API_KEY='your_pinecone_api_key'
```

To run the agent, you can run the main script with fastapi:

```bash
fastapi dev main.py
```

### Challenge Queries

The following set of test queries has been tested with successful results:

1. `Can my pet travel with me on the plane on Delta?`

2. `I have three kids 2, 8 and 10 years old and I am traveling with them on a United flight, what are the rules for children traveling?`

3. `What is the baggage policy for American Airlines?`

4. `My wife is 8 months pregnant, can she travel on a Delta flight?`
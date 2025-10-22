FROM python:3.10
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --no-cache-dir && rm requirements.txt
WORKDIR /opt/airline-agent
COPY . /opt/airline-agent
# Parse policies
RUN python policy_parser.py
# Run server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0"]
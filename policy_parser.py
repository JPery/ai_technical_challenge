import glob
import os
from pypdf import PdfReader

from agent.constants import DATA_FOLDER


def parse_pdf(path):
    """
    Parse a pdf file
    :param path: path to pdf file
    :return: text parsed from pdf file
    """
    reader = PdfReader(path)
    text = "\n".join([page.extract_text() for page in reader.pages])
    return text

def parse_markdown(path):
    """
    Parse a markdown file
    :param path: path to markdown file
    :return: text parsed from markdown file
    """
    with open(path, "r", encoding="utf8") as f:
        content = f.read()
    return content

if __name__ == "__main__":
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)

    # Iterate over files in policies directory
    for filename in glob.glob('policies/**', recursive=True):
        # Skip folders
        if os.path.isdir(filename):
            continue
        # Get company and policy names
        company, policy = os.path.split(filename)
        _, company = os.path.split(company)
        policy = policy.rsplit(".", 1)[0]
        # Check file extension and parse it accordingly
        if filename.endswith('.pdf'):
            text = parse_pdf(filename)
        elif filename.endswith('.md'):
            text = parse_markdown(filename)
        else:
            print('Cannot parse {}'.format(filename))
            continue
        # Write the parsed file in custom format
        with open(f"{DATA_FOLDER}/{company}_{policy}.txt", "w", encoding="utf8") as f:
            f.write(f"# {company} - {policy}\n{text}")
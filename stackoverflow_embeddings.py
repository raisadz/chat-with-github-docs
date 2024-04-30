from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
import requests
from bs4 import BeautifulSoup
from typing import Optional

TAG = "python-polars"
QA_FILE = f"qa_most_voted_{TAG}.txt"
INDEX_NAME = "polars-docs"


def scrape_stackoverflow(tag, file_name: Optional[str] = None):
    url = "https://stackoverflow.com"
    most_voted_url = f"{url}/questions/tagged/{tag}?sort=MostVotes&edited=true"
    response = requests.get(most_voted_url)
    soup = BeautifulSoup(response.text, "html.parser")
    questions = soup.find_all("div", class_="s-post-summary")

    all_questions_answers = ""
    for question in questions:
        question_link = question.find("a", class_="s-link")
        question_title = question_link.text.strip()
        question_url = url + question_link["href"]
        question_response = requests.get(question_url)
        question_soup = BeautifulSoup(question_response.text, "html.parser")
        question_body = question_soup.find(
            "div", class_="s-prose js-post-body"
        ).text.strip()
        accepted_answer = question_soup.find(
            "div", class_="answer js-answer accepted-answer js-accepted-answer"
        )
        question_votes = int(
            question.find(
                "span", class_="s-post-summary--stats-item-number"
            ).text.strip()
        )
        question_tags = question.find_all("a", class_="post-tag")
        tags = [tag.text for tag in question_tags]

        if accepted_answer:
            question_excerpt = question.find(
                "div", class_="s-post-summary--content-excerpt"
            ).text.strip()
            accepted_answer_text = accepted_answer.find(
                "div", class_="s-prose js-post-body"
            ).text.strip()
            accepted_answer_votes = int(
                accepted_answer.find("div", class_="js-vote-count").get_text(strip=True)
            )
            all_questions_answers += f"""Question:\n{question_body}\nAnswer:\n{accepted_answer_text}\nEND\n\n\n"""

        if file_name is not None:
            with open(file_name, "w") as f:
                f.write(all_questions_answers)
            f.close()
    return all_questions_answers


if __name__ == "__main__":
    all_questions_answers = scrape_stackoverflow(tag=TAG, file_name=QA_FILE)
    loader = TextLoader(f"./{QA_FILE}")
    docs = []
    docs.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(
        separators="END\n\n\n", chunk_size=1000, chunk_overlap=200
    )
    docs = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()

    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    vectorstore.add_documents(docs)

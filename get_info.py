from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, OpenAI
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

import os

from bs4 import BeautifulSoup
from html2text import html2text



OPENAI_API_KEY = "YOUR KEY"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
DOCUMENT_SIZE=1000
CONTEXT_QUERY_TEMPLATE="""Inquiry:{query}
    Answer the inquiry solely based on the following context.
    
    {context}
    """
llm = OpenAI()


#should have base class contextHandler
class WebsiteHandler():
    DOCUMENT_ID=0
    embedding=OpenAIEmbeddings()

    def __init__(self, page_content: str, removeUnreadable=True):
        self.page_content = page_content
        text=page_content
        if removeUnreadable:
            text=WebsiteHandler.getReadableText(self.page_content)

        self.docs: list[Document] = self.split(text)


    def split(self, long_text: str):
        markdown_document = long_text

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False)
        md_header_splits = markdown_splitter.split_text(markdown_document)
        md_header_splits

        chunk_size = DOCUMENT_SIZE
        chunk_overlap = 30
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        # Split
        splits = text_splitter.split_documents(md_header_splits)
        return splits

    def get_info_f(self, query: str, outputType: str, exampleOutput):
        
        docs=self.get_relevant_docs(query) ##issue must be somewhere around here -> docs are first document

        output = self.get_ans_from_context(docs, query)
        print("unformatted output")
        print(output)
        print("-"*50)
        formatted_output = self.format(output, outputType, exampleOutput)
        print("formatted output")
        print(formatted_output)
        print("-"*50)

        return formatted_output
    
    def get_info(self, query:str):
        docs=self.get_relevant_docs(query)
        output = self.get_ans_from_context(docs, query)
        return output
    def get_relevant_docs(self, query:str):
        db = Chroma.from_documents(self.docs, WebsiteHandler.embedding)
        print(self.docs[3])
        retriever = db.as_retriever(
            search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
        )
        docs = retriever.get_relevant_documents(query)
        print(docs[0])
        return docs
    
    def get_ans_from_context(self, docs, query):
        document_prompt = PromptTemplate(
            input_variables=["page_content"], template="Part of Page: {page_content}"
        )
        document_variable_name = "context"
        stuff_prompt_override = CONTEXT_QUERY_TEMPLATE
        prompt = PromptTemplate(
            template=stuff_prompt_override, input_variables=["context", "query"]
        )

        llm_chain = LLMChain(llm=llm, prompt=prompt)
        chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_prompt=document_prompt,
            document_variable_name=document_variable_name,
        )
        output=chain.run(input_documents=docs, query=query)

        return output

    def format(self, text, outputType, example: str):
        openai_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)
        output = openai_model.invoke(f'''
            Just return the {outputType} from this context:
            {text}
            ------------
            
            Here's an example for you to understand the output correctly:
            The text \"{example["text"]}\" should return \"{example["return-value"]}
            ''')
        return output.content

    def deduce(self, short_text: str, query: str, outputType: str, exampleOutput):
        raise NotImplementedError()

    def fulfill_goal(self, page_content: str, goal: str):
        raise NotImplementedError()

    def best_actions(self, page_content: str, query: str):
        raise NotImplementedError()

    #effect:writes 
    #return: returns a readable text
    def getReadableText(page_content:str):

        # Parse the HTML
        soup = BeautifulSoup(page_content, "html.parser")

        # Find and remove the tag you want to delete
        def removeTag(tag:str):
            tags_to_remove = soup.find_all(tag)  # Replace "tag_name" with the name of the tag you want to remove

            for tag in tags_to_remove:
                tag.decompose()

        removeTag("head")
        removeTag("script")
        removeTag("code")
        removeTag("meta")

        # Remove all empty lines

        soup = BeautifulSoup(str(soup).strip(), "html.parser")
        prettified_text = soup.prettify()

        cleaned_text = html2text(prettified_text)

        # Save the cleaned text to file
        file_name = f"src/temp/websites/{WebsiteHandler. DOCUMENT_ID}.md"


        with open(file_name, "w") as file:
            file.write(cleaned_text)

        # Increase the document id
        WebsiteHandler. DOCUMENT_ID += 1

        return cleaned_text

from langchain_core.output_parsers import StrOutputParser
def get_ans(short_context:str, query:str):
    llm = OpenAI()
    stuff_prompt_override = """Given this context:
    -----
    {context}
    -----
    Please give the result to the following query:
    {query}"""
    prompt = PromptTemplate(
        template=stuff_prompt_override, input_variables=["context", "query"]
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    #need to parse the output
    return chain.invoke({"context":short_context,"query":query})




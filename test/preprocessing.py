from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from constant import md_header

def split(doc:str,
          type:str):
    if type == 'md':
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=md_header, 
                                                       strip_headers=False)
        md_header_splits = markdown_splitter.split_text(doc)
        return md_header_splits

def text_split(text,
               chunk_size=1000,
               chunk_overlap=30,
               method=RecursiveCharacterTextSplitter):
    text_splitter = method(chunk_size=chunk_size,
                           chunk_overlap=chunk_overlap
                           )
    splits = text_splitter.split_documents(text)
    return splits
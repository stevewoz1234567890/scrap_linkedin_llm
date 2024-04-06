from langchain_community.document_loaders import TextLoader
from constant import PATH_TO_EXAMPLE

def text_loader(url=PATH_TO_EXAMPLE,
                encoding='utf-8'):
    print('loading file')
    loader = TextLoader(url, encoding=encoding)
    result=loader.load() 
    page_content=result[0].page_content
    return page_content

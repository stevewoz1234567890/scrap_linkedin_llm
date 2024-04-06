from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAI
#turn all of the functions to a class (with the public functions: get_info, deduce, next_action)
PATH_TO_EXAMPLE="https--www.linkedin.com-jobs-view-3841921475-.md"
OPENAI_API_KEY="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"

def split(long_text:str):
    markdown_document = long_text

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on,strip_headers = False)
    md_header_splits = markdown_splitter.split_text(markdown_document)
    md_header_splits
    
        # Char-level splits
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    chunk_size = 1000
    chunk_overlap = 30
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    # Split
    splits = text_splitter.split_documents(md_header_splits)
    return splits

def embed(texts:list[str]):
    embeddings_model = OpenAIEmbeddings()
    embeddings = embeddings_model.embed_documents(texts)
    return embeddings

def get_info(long_text:str,query:str,outputType:str, exampleOutput):
    splits=split(long_text)
    #embeddings=embed(splits)
    db = Chroma.from_documents(splits, OpenAIEmbeddings(api_key=OPENAI_API_KEY))

    retriever=db.as_retriever(
            search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.2}
    )
    docs = retriever.get_relevant_documents(query)
    #get an agent to get the best result from those docs and then turn it to the correct type (+maybe the reference long_text / reasoning)

    #get the best result from those results
    output= get_ans_from_context(docs,query)
    #turn to desired outputType
    formatted_output = format(output,outputType,exampleOutput)
    return formatted_output

def get_ans_from_context_deprecated(retriever,searchQuery:str):
    from langchain_openai import ChatOpenAI
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)
    
    '''template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI bot. Your name is {name}."),
            ("human", "Hello, how are you doing?"),
            ("ai", "I'm doing well, thanks!"),
            ("human", "{user_input}"),
        ])

    messages = template.format_messages(
            name="Bob",
            user_input="What is your name?"
        )'''
    
    prompt = ChatPromptTemplate.from_template("""Get the best results of the search query only based on the following context:

    <context>
    {context}
    </context>

    Search Query: {input}""")
    document_chain=create_stuff_documents_chain(llm,prompt)
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    response = retrieval_chain.invoke({"input": searchQuery})

    print(response["answer"])

def get_ans_from_context(docs, query):
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import PromptTemplate
    from langchain.chains import LLMChain, StuffDocumentsChain

        # We prepare and run a custom Stuff chain with reordered docs as context.

    # Override prompts
    document_prompt = PromptTemplate(
        input_variables=["page_content"], template="{page_content}"
    )
    document_variable_name = "context"
    llm = OpenAI(api_key=OPENAI_API_KEY)
    stuff_prompt_override = """Given this context:
    -----
    {context}
    -----
    Please give the result to the following query:
    {query}"""
    prompt = PromptTemplate(
        template=stuff_prompt_override, input_variables=["context", "query"]
    )

    # Instantiate the chain
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_prompt=document_prompt,
        document_variable_name=document_variable_name,
    )
    #how to turn run to invoke
    return chain.run(input_documents=docs, query=query)
    ''' from langchain.chains.openai_functions.extraction import create_extraction_chain
    create_extraction_chain(docs,llm,prompt).run()
    '''
    
def format(text,outputType,example:str):
    from langchain_openai import ChatOpenAI

    # Create an instance of the ChatOpenAI model
    openai_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=OPENAI_API_KEY)

    # Invoke the model with the prompt
    output = openai_model.invoke(f'''
        Just return the {outputType} from this long_text:
        {text}
        ------------
        
        Here's an example for you to understand the output corectly:
        The long_text \"{example["text"]}\" should return \"{example["return-value"]}
        ''')
    return output.content

def deduce(short_text:str,query:str, outputType:str, exampleOutput):
    #make deduction
    formatted_output = format(output,outputType,exampleOutput)
    return formatted_output

def fulfill_goal(page_content:str, goal:str):
    #do next action while nextaction is not null
    raise NotImplementedError()
# next action needs to be a tool (maybe already some tools exist for it)
#effect: executes next action to achieve the goal
def next_action(page_content:str, goal:str):
    raise NotImplementedError()

loader = TextLoader(PATH_TO_EXAMPLE, encoding='utf-8')
result=loader.load() 
page_content=result[0].page_content
example={
    "text":"This company is called Batman GmbH.",
    "return-value":"Batman GmbH"
}

company_name=get_info(page_content,"The company responsible for the job post", "Company name", example)
print(company_name)


job_tasks=get_info(page_content,
    query=f"Get the job tasks of {company_name} in detail",
    outputType="Detailed List",
    exampleOutput={
        "text":"The tasks of this job are - Cleaning the kitchen, - going to class",
        "return-value":"-Cleaning the kitchen\n-Going to class"
    }
    )
print(job_tasks)
# need more to test it
neukunden=get_info(job_tasks,
    query='''Decide whether the person in this job will talk with a lot of potentially new customers
Can be determined by whether a lot of points talk about it. 
Only say "yes"/"no" when you are very sure.
''',
    outputType="Yes/No/Unsure",
    exampleOutput={
        "text":"The tasks of this job are - Talking with customers - going to class",
        "return-value":"No"
    }
    )
print(neukunden)

max_employees=get_info(page_content,
    query=f"The amount of employees of {company_name}",
    outputType="Maximum amount"    ,
    exampleOutput={
    "text":"Has 2-10 employees",
    "return-value":"10"
}
    )

print(max_employees)


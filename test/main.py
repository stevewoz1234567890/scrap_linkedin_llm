from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain, StuffDocumentsChain
from loader import text_loader
from preprocessing import split, text_split
from constant import OPENAI_API_KEY, EXAMPLES, QUERIES, GOALS

llm = OpenAI(api_key=OPENAI_API_KEY)
openai_model = ChatOpenAI(model_name="gpt-3.5-turbo", 
                        temperature=0,
                        api_key=OPENAI_API_KEY)
stuff_prompt_template = """
Given this context:
-----
{context}
-----
Please give the result to the following query:
{query}
"""
stuff_prompt = PromptTemplate(
    template=stuff_prompt_template,
    input_variables=["context", "query"]
)
llm_chain = LLMChain(llm=llm,
                    prompt=stuff_prompt)

def retrieve(long_text,
             query,
             embedding='openai',
             search_type="similarity_score_threshold"
             ):
    doc = split(long_text, 'md')
    doc = text_split(doc)

    if embedding == 'openai':
        db = Chroma.from_documents(doc, 
                                   OpenAIEmbeddings(api_key=OPENAI_API_KEY))

    if search_type == "similarity_score_threshold":
        retriever=db.as_retriever(search_type=search_type, 
                                search_kwargs={"score_threshold": 0.2}
                                )
    
    docs = retriever.get_relevant_documents(query)

    return docs

def get_info(text, goal):
    example = EXAMPLES[goal]
    output = openai_model.invoke(
        f'''
        Just return the {goal} from this long_text:
        {text}
        ------------
        
        Here's an example for you to understand the output corectly:
        The long_text \"{example["text"]}\" should return \"{example["return-value"]}
        ''')
    return output.content


if __name__ == "__main__":
    page_content = text_loader()

    company_name = ''

    for goal in GOALS:
        query = QUERIES[goal]

        retrieved = retrieve(page_content,
                         query=query)
        
        document_prompt = PromptTemplate(
            input_variables=["page_content"], 
            template="{page_content}"
        )

        chain = StuffDocumentsChain(llm_chain=llm_chain,
                                    document_prompt=document_prompt,
                                    document_variable_name="context",
        )
            
        text = chain.run(input_documents=retrieved, 
                        query=query) 
    
        output = get_info(text, goal)

        if goal == "Company name":
            company_name = output
            QUERIES["Detailed List"] = QUERIES["Detailed List"].replace("this company", company_name)
            QUERIES["Company Location"] = QUERIES["Company Location"].replace("this company", company_name)
            QUERIES["Maximum amount"] = QUERIES["Maximum amount"].replace("this company", company_name)
            QUERIES["Minimum amount"] = QUERIES["Minimum amount"].replace("this company", company_name)
        print(goal + "  :  " + output)





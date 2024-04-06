PATH_TO_EXAMPLE="https--www.linkedin.com-jobs-view-3841921475-.md"

# OPENAI_API_KEY = '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'
# OPENAI_API_KEY = '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'
OPENAI_API_KEY = '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'

md_header = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

GOALS = ["Company name", 
         "Company Location",
         "Detailed List", 
         "Maximum amount", 
         "Minimum amount", 
         "neukunden Yes/No/Unsure"]

EXAMPLES = {
    "Company name": {
        "text":"This company is called Batman GmbH.",
        "return-value":"Batman GmbH"
    },
    "Company Location": {
        "text":"This company is based on New York.",
        "return-value":"New York"
    },
    "Detailed List": {
        "text":"The tasks of this job are - Cleaning the kitchen, - going to class",
        "return-value":"-Cleaning the kitchen\n-Going to class"
    },
    "Maximum amount": {
        "text":"has 2,134-10,234 employees",
        "return-value":"10,234"
    },
    "Minimum amount": {
        "text":"has 2,134-10,234 employees",
        "return-value":"2,134"
    },
    "neukunden Yes/No/Unsure": {
        "text":"The tasks of this job are - Talking with customers - going to class",
        "return-value":"No"
    }
}

QUERIES = {
    "Company name": "The company responsible for the job post",
    "Company Location": "The location of this company",
    "Detailed List": "Get the job tasks of this company in detail",
    "Maximum amount": "The maximum number of employees of this company",
    "Minimum amount": "The minimum number of employees of this company",
    "neukunden Yes/No/Unsure": '''Decide whether the person in this job will talk with a lot of potentially new customers
        Can be determined by whether a lot of points talk about it. 
        Only say "yes"/"no" when you are very sure.
        '''
}
import os
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
from langchain_community.chat_models import ChatOllama
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Initialize Neo4j connection
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD")
)

# Initialize Ollama LLM
llm = ChatOllama(model=os.getenv("OLLAMA_MODEL"))

# Create Cypher prompt template
cypher_prompt = PromptTemplate(
    input_variables=["schema", "question"],
    template="""
You are a Neo4j Cypher expert.
Generate Cypher to retrieve:
- Generic alternatives of a given BrandDrug
- For each generic: name, manufacturer name, ingredient name
- And all price info: amount, currency, quantity, and date

Use these relationships:
- (BrandDrug)-[:HAS_GENERIC_ALTERNATIVE]->(GenericDrug)
- (GenericDrug)-[:CONTAINS]->(Ingredient)
- (GenericDrug)-[:MANUFACTURED_BY]->(Manufacturer)
- (GenericDrug)-[:HAS_PRICE]->(Price)

Always use exact label and property names.
Use a single RETURN statement at the end of the query.

Schema: {schema}
Question: {question}
"""
)


# Initialize GraphRAG chain
chain = GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    cypher_prompt=cypher_prompt,
    verbose=True,
    return_intermediate_steps=True,
    allow_dangerous_requests=True
)

# --- User Query ---
while True:
    query = input("\n💊 Enter a brand drug name (or 'exit'): ")
    if query.lower() == 'exit':
        break

    formatted_query = f"What are the generic alternatives for {query}, their prices, ingredients, and manufacturers?"
    response = chain.run(formatted_query)
    print("\n🧠 Result:", response)

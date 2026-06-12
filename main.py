from agents import (
    research_agent,
    analysis_agent,
    reviewer_agent
)

query = input("Ask: ")

docs = research_agent(query)

draft = analysis_agent(
    query,
    docs
)

final_answer = reviewer_agent(
    query,
    draft
)

print('Final Answer is \n',final_answer)
## Architecting Sourcing Subagent

1. Understand the requirements, Inputs, outputs, Edge cases etc. We will mock the output of the planning agent to feed to our sourcing subagent. DO NOT make decisions on arch
2. Break down tasks for the sourcing subagent, which Searches Roboflow Universe, Kaggle, and the web for matching datasets/papers; fork or download candidates.
3. Do a detailed research on the following tools that our subagent will require: Roboflow MCP, Kaggle MCP, web_search. Research their docs if and when required.
4. chalk out the implementation steps in chunks.
5. For any external services, api keys, stop and ask for it. I will drop those keys in .env file. Make sure you update the .env.example file as well. 
6. After the subagent is implemented, test it out with the mocked output of the planning agent. 
7. Document the subagent in detail. Make sure you document the following: 
    a. Input parameters
    b. Output
    c. Edge cases
    d. Any other relevant information

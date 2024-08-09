The goal of this project was to create a more efficient gmail search than the existing one. 
I aimed to reach this goal by performing Retrieval Augmented Generation (RAG) to increase the accuracy of my LLM's responses. 

What is RAG?
- Providing the best context possible for a model to answer questions

How Does it Work?
- Use a text embedding model to convert the important email info into a high dimensional vector format
- Store those vectors in a vector database
  - I used QDRANT for this project
- Get a query from the user, and use the text embedding model to vector embed it
- Perform a similarity search between the query vector and the QDRANT database to find the most similar email to the query
- Retrieve the important info of the most similar email, and combine it with the text query to make a better prompt for the LLM
- The LLM then uses the final prompt to generate an accurate response

How I Approached the Project
- Used the Gmail API to retrieve the last few hundred messages in my inbox
   - Get all the necessary info from each message
   - Wrote a get_message_body() function to parse through any html content
      - Saved a plain text version of the email body as well as an html version
- Saved .html versions of each email
   - Wrote the save_open_html for this purpose
- Adding my emails to QDRANT db
   - Split up the text body into chunks so it won't exceed the token limit for the text embedding model
   - Created an email summary consisting of the sender, subject, time stamp, and text body of the email
       - Vector Embedded the summary to use as the vector to compare against during the similarity search
   - Include the path of the email html file in the metadata 
- Performing Similarity Search
   - Vector embed the user query
   - Compare it to every email summary vector using COSINE similarity
   - Retrieve the metadata of the most similar email
- Creating context for the LLM
   - Wrote the responsibilities of the LLM so it knows what to do
- Creating and passing the prompt to the LLM
   - Put together the text versions of the user query and the relevant info of the most similar email
   - Pass it to the gpt-3.5-turbo LLM to get a response
- Returning a response
   - Print the first response the model generates, while also opening the html file of the email
      - Makes it easier to understand for the user
- Updating the context
   - Append a log of the user query and response to the context so it will be able to maintain a coherent conversation

So far this project is able to support text-only emails. I plan to implement the necessary functionalities needed to process attachments to make this even more powerful. Stay tuned!

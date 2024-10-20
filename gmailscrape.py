import os.path
import base64
import re 

from openai import OpenAI
from dotenv import load_dotenv
import httpx

from qdrant_client import QdrantClient
from qdrant_client.http import models 
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from qdrant_client.http.models import CollectionStatus, UpdateStatus

from bs4 import BeautifulSoup
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
import tiktoken

import comet_llm 
from IPython.display import display
import ipywidgets as widgets
import time

'''
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from PIL import Image
from io import BytesIO
import torch
from torchvision import models, transforms
'''

#Context I feed to comet llm to train it 
context = [{'role': 'system',
            'content': f"""
You are Email Assistant, an AI assistant for users who would like to find specific emails in their inbox.
Your role is to assist users in finding any email they need. Be friendly and help in your interactions. 
You might get a query from the user asking you to find an email given some keywords or context.
You might even get a one word query. Your job is to find the most similar email to the user query. 
Make sure to add the from email, subject, date, and summary of the email body in the response. 
The summary should show how you have answered the user's question, or found the keyword they were looking for.
Always make sure to greet the user before giving a response.
"""}]

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

# Load environment variables
load_dotenv()

MY_COMET_KEY = os.getenv('MY_COMET_KEY')

#create a comet project
comet_llm.init(project="my_email_search",
               api_key=MY_COMET_KEY)

# Set OpenAI API key
MY_OPENAI_KEY = os.getenv('MY_OPENAI_KEY')
if not MY_OPENAI_KEY:
    raise ValueError("MY_OPENAI_KEY is not set in the environment variables")

# Initialize Qdrant client
QDRANT_URL = os.getenv('QDRANT_URL')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')

if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("QDRANT_URL or QDRANT_API_KEY is not set in the environment variables")

q_client = QdrantClient(
    url=QDRANT_URL, 
    timeout=60.0,
    api_key=QDRANT_API_KEY
)
'''
# Create collection
collection_name = "TestDatabase"  # Ensure this name is appropriate for your project
try:
    q_client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "summary": VectorParams(size=1536, distance=Distance.COSINE),
        }
    )
    print(f"Collection '{collection_name}' created successfully.")
except Exception as e:
    print(f"An error occurred while creating the collection: {e}")
'''

tokenizer = tiktoken.get_encoding('cl100k_base')

#create vector embeddings
def get_embedding(text, model="text-embedding-3-small"):
    client = OpenAI(api_key=MY_OPENAI_KEY,)
    response = client.embeddings.create(input = [text], model=model)
    
    return response.data[0].embedding

def main():
  """Shows basic usage of the Gmail API.
  Lists the user's Gmail labels.
  """
  creds = None
  # The file token.json stores the user's access and refresh tokens, and is
  # created automatically when the authorization flow completes for the first
  # time.
  if os.path.exists("token.json"):
    creds = Credentials.from_authorized_user_file("token.json", SCOPES)
  # If there are no (valid) credentials available, let the user log in.
  if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
      creds.refresh(Request())
    else:
      flow = InstalledAppFlow.from_client_secrets_file(
          "credentials.json", SCOPES
      )
      creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open("token.json", "w") as token:
      token.write(creds.to_json())

  try:
    # Call the Gmail API
    service = build("gmail", "v1", credentials=creds)


    #fetch the last 10 emails
    result = service.users().messages().list(userId='me', maxResults=100).execute()
    messages = result.get('messages', [])

    if not messages:
      print("No messages found.")
      return

    for idx, msg in enumerate(messages, start=1):
      msg_id = msg['id']
      message = service.users().messages().get(userId='me', id=msg_id).execute()
      email_dir = os.path.join('/Users/varunganesan/Downloads/EmailScrapeProject/allEmails/', f'email_{idx}')
      os.makedirs(email_dir, exist_ok=True)
      print_email_details(message, service, email_dir, idx)

  except HttpError as error:
    print(f"An error occurred: {error}")

def get_message_body(message, service):

    body = ""  # Initialize an empty string to store the email body
    attachments = []  # Initialize an empty list to store attachment details
    html_body = None

    def process_part(part):
        """Helper function to process each part of the email message."""
        nonlocal body  # Use the nonlocal keyword to modify the outer body variable
        nonlocal html_body

        if part['mimeType'] == 'text/plain':  # Check if the part is plain text
            # Decode the base64 encoded plain text and append it to the body
            body += base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
        elif part['mimeType'] == 'text/html':  # Check if the part is HTML
            # Decode the base64 encoded HTML content
            html_content = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')

            
            # Use BeautifulSoup to parse the HTML and extract text
            soup = BeautifulSoup(html_content, 'html.parser')
            body += soup.get_text()  # Append the extracted text to the body

            html_body = html_content 
        elif 'attachmentId' in part['body']:  # Check if the part is an attachment
            # Create a dictionary with attachment details
            attachment = {
                'filename': part['filename'],  # The filename of the attachment
                'mimeType': part['mimeType'],  # The MIME type of the attachment
                'attachmentId': part['body']['attachmentId']  # The attachment ID to fetch the actual data
            }
            attachments.append(attachment)  # Add the attachment details to the list
        
    if 'parts' in message['payload']:  # Check if the email is multipart
        for part in message['payload']['parts']:  # Iterate over each part in the email
            if 'parts' in part:  # Check if the part itself has sub-parts (nested structure)
                for sub_part in part['parts']:  # Iterate over each sub-part
                    process_part(sub_part)  # Process each sub-part
            else:
                process_part(part)  # Process the part if it has no sub-parts
    else:  # If the email is not multipart
        process_part(message['payload'])  # Process the single part

    return body, html_body, attachments  # Return the email body and the list of attachments

def save_open_html(html_body, attachments, service, message_id, message,  save_path):
  
   # saves the html body and the embedded resources and then opens the file
   if not os.path.exists(save_path):
      os.makedirs(save_path)

   headers = message['payload']['headers']

   subject = None
   from_email = None
   time_stamp = None

   for header in headers:
    if header['name'] == 'Subject':
      subject = header['value']
    if header['name'] == 'From':
      from_email = header['value']
    if header['name'] == 'Date':
      time_stamp = header['value']

    email_details = f"From: {from_email}<br>Subject: {subject}<br>Date: {time_stamp}<br>"

    # Create the HTML template
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Email</title>
    </head>
    <body>
        <h2>Email Details</h2>
        <p>{email_details}</p>
        <hr>
        <div>
            {html_body}
        </div>
    </body>
    </html>
    """
    
   # Save the HTML content
   html_file_path = os.path.join(save_path, 'email.html')
   with open(html_file_path, 'w', encoding='utf-8') as f:
     
     f.write(html_template)
   #print(f"HTML email saved to {html_file_path}")
   
   #ss_dir = os.path.join('/Users/varunganesan/Downloads/EmailScrapeProject/htmlss/', os.path.basename(save_path) + ".png")
   #take_screenshot_of_file(html_file_path, ss_dir)
   '''
   # Open the HTML file with the default web browser
   if os.name == 'posix':  # For macOS and Linux
     os.system(f'open "{html_file_path}"')
   elif os.name == 'nt':  # For Windows
     os.startfile(html_file_path)
   '''
   return html_file_path
   
def add_to_db(sender, 
              subject, 
              time_stamp,
              chunks, 
              html_path, 
              id, 
              qdrant_client: QdrantClient = q_client, 
              collection_name: str = "TestDatabase"):
   #create an empty list for the points 
   points = []

   for index, chunk in enumerate(chunks):
      summary = (
      f"Sender: {sender}\n"
      f"Subject: {subject}\n"
      f"Time Stamp: {time_stamp}\n"
      f"Chunk{index}: chunk\n"
      )
      #get vector embeddings for what we need
      summary_vector = get_embedding(summary)

      #create dictionary with the embeddings
      vector_dict = {"summary": summary_vector}

      #create a dictionary containing payload data
      payload = {
      "id":f"{id} chunk{index}",
      "sender": sender,     
      "subject": subject, 
      "timestamp": time_stamp,
      "chunk": chunk,
      "html_path": html_path
      }
                
      #you dont just insert a vector by itself, you create a point with the id, vector, and payload and add the POINT to the collection
      point = PointStruct(id=id, vector=vector_dict, payload=payload)
      points.append(point)

      operation_info = qdrant_client.upsert(
        collection_name=collection_name,
        wait=True,
        points=points
      )
      
      # Check the operation_info for success
      if operation_info.status == 'completed':
        print(f"Successfully inserted chunk{index} of point {id} into the database.")
      else:
        print(f"Failed to insert point {id} into the database. Status: {operation_info.status}")

def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

def chunk_body(body):
    # Initialize text_splitter
    tokenizer = tiktoken.get_encoding('cl100k_base')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=20,
        length_function=tiktoken_len,
        separators=['\n\n', '\n', ' ', '']
    )

    documents = []

    # Split the content into chunks
    chunks = text_splitter.split_text(body)

    for chunk in chunks:
       documents.append(chunk)
    return documents
      
def print_email_details(message, service, email_dir, id):
  """Prints the details of an email message.
     - retrieve the headers from the message 
     - initialize the subject and sender email"""
  headers = message['payload']['headers']
  subject = None
  from_email = None


  for header in headers:
    if header['name'] == 'Subject':
      subject = header['value']
    if header['name'] == 'From':
      from_email = header['value']
    if header['name'] == 'Date':
       date_stamp = header['value']
  
  # get the message body 
  body, html_body, attachments = get_message_body(message, service)
  '''
  print(f"From: {from_email}")
  print(f"Subject: {subject}")
  print(f"Date:) {date_stamp}")
  print("=" * 50) 
  print(f"Body: {body}")
  '''
  html_path = save_open_html(html_body, attachments, service, message['id'], message,  email_dir)
  chunks = chunk_body(body)
  print(f"email {id} has {len(chunks)} chunks")
  add_to_db(from_email, subject, date_stamp, chunks, html_path, id)

  '''
  if attachments:
     attachments_dir = os.path.join(email_dir, 'attachments')
     os.makedirs(attachments_dir, exist_ok=True)

     print("Attachments:")
     for attachment in attachments:
       print(f"- {attachment['filename']} ({attachment['mimeType']})")
       #save_open_attachment(service, attachment, message['id'], attachments_dir)
    '''
  print("=" * 50)

def similarity_search(query:str, 
                      search_vector:str, 
                      limit: int=3, 
                      client: QdrantClient = q_client, 
                      collection_name: str = "TestDatabase", 
                      **kwargs):
  """
  Perform a similarity search and return the HTML path of the top result.

  Parameters:
  - query: The query text for which we are performing the similarity search.
  - search vector: the vector we will compare the query against 
  - limit: how many results we want 
  - client: the qdrant client we initialized before
  - collection_name: The name of the collection in the vector database.
  - kwargs

  Returns:
  - The HTML path of the top result.
"""
  # Get the embedding vector for the query
  query_vector = get_embedding(query)

  # Perform the similarity search
  search_results = q_client.search(
    collection_name=collection_name,
    query_vector=(search_vector, query_vector),
    limit=limit,  # Get the top result
    with_payload=True,
    **kwargs

  )

  # Check if any results were found
  if not search_results:
    print("No results found.")
    

  # Get the payload of the top result
  top_result = search_results[0]
  payload = top_result.payload
  
  return payload

def create_prompt(user_query, payload):
   prompt = {
        "User Query": user_query,
        "Email Document": {
          "Sender": payload['sender'],
          "Subject": payload['subject'],
          "Timestamp": payload['timestamp'],
          "Body": payload['chunk0']
        }
   }
   html_path = payload['html_path']

   return prompt, html_path

#chatbot 
def chatBot(messages, model="gpt-3.5-turbo"):
    
    client = OpenAI(api_key=MY_OPENAI_KEY,) #creates an instance of the openAI client
    chat_completion = client.chat.completions.create( #sends a request to the OpenAI API with the content of the message/the model needed as args
        messages=messages,
        model=model,
    )
    return chat_completion.choices[0].message.content #there are a bunch of responses that can be returned, we select the first one. 

def collect_messages(input, html_path):
    context.append({'role':'user', 'content':f"{input}"})

    start_time = time.time()
    response = chatBot(context)

    
    #calculate the duration
    end_time = time.time()
    duration = end_time - start_time

    #log to comet
    
    comet_llm.log_prompt(
        prompt=input,
        output=response,
        duration=duration,
        metadata={
            "role": context[-1]['role'],
            "context": context[-1]['content'],
            "context": context,
        },
    )
    
    context.append({'role': 'assistant', 'content': f"{response}"})
    if response == 'None':
       return "it didnt work"
    print(response)
    if html_path:
      print(f"Top result HTML path: {html_path}")
      # Optionally open the HTML file
      with open(html_path, 'r', encoding='utf-8') as file:
        # Open the HTML file with the default web browser
        if os.name == 'posix':  # For macOS and Linux
          os.system(f'open "{html_path}"')
        elif os.name == 'nt':  # For Windows
          os.startfile(html_file_path)
    else:
        print("HTML path not found in payload.")
    return response

#if __name__ == "__main__":
  #main()

query = input("What would you like to find in your emails?")

while(input != "Quit"):
  payload = similarity_search(search_vector= "summary", query=query)
    
  prompt, html_path = create_prompt(query, payload)

  collect_messages(prompt, html_path)

  query = input("What would you like to find in your emails?")


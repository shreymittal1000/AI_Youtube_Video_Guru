import docarray as da
import glob, os, tiktoken, yt_dlp
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from IPython.core.display import Markdown, display_markdown
from openai import OpenAI
import warnings

# Load the .env file (it contains the OpenAI API key)
load_dotenv()

# Ignore warnings
warnings.filterwarnings("ignore")

# Fetch the YouTube video URLs
with open("video_urls.txt", "r") as f:
    video_urls = f.readlines()
    
# Make sure the audiostore directory exists. If not, create it
try:
    os.path.exists("audiostore")
except:
    os.mkdir("audiostore")

# Make sure the transcripts directory exists. If not, create it
try:
    os.path.exists("transcripts")
except:
    os.mkdir("transcripts")

# Configure the YouTube downloader
ydl_config = {
    "format": "bestaudio/best",
    "outtmpl": "audiostore/%(title)s.%(ext)s",
    "postprocessors": [{
        "key": "FFmpegExtractAudio",
        "preferredcodec": "wav",
        "preferredquality": "192",
    }],
    "verbose": True
}

# Download the YouTube videos
print("Downloading the YouTube videos...")
try:
    with yt_dlp.YoutubeDL(ydl_config) as ydl:
        ydl.download(video_urls)
except Exception as e:
    print("Failed to download all the YouTube videos. Error: ", e)
    
# Get the list of downloaded files
downloaded_files = glob.glob("audiostore/*.wav")
    
# Initialize the OpenAI client. Assign to client
client = OpenAI()

# Open the downloaded files as read-binary and transcribe them
print("Transcribing the downloaded files...")
for file in downloaded_files:
    with open(file, "rb") as f:
        # Transcribe the audio
        data = f.read()
        transcription = client.audio.transcriptions.create(file=data, model="whisper-1")
        # Save the transcription to a text file
        with open(f"transcripts/{file}.txt", "w") as t:
            t.write(transcription)
            
# Create a Retriever of the knowledge from the transcriptions
print("Creating a Knowledge Retriever...")
retriever = DocArrayInMemorySearch.from_documents(TextLoader().load("transcripts/*.txt"), OpenAIEmbeddings()).as_retriever()

# Create a RetrievalQA instance. This will be used to... retrieve data
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0.0),
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)

print("System is ready to answer questions. Press CTRL+C to quit.")
while True:
    # Get the user's question
    question = input("Ask me a question: ")
    # Get the answer to the question
    answer = qa.invoke(question)
    # Print the answer
    print(answer)
    print()

import torch
import clip
import numpy as np
import cv2
import whisper
import psycopg2
import os
import re
from PIL import Image as PILImage
from moviepy.editor import VideoFileClip
from tqdm import tqdm
# For Arabic text display
import arabic_reshaper
from bidi.algorithm import get_display

# LangChain and Vector Store imports
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.pgvector import DistanceStrategy
# You may also need the following import if you use the hybrid retriever as implied in the original code
# from langchain.retrievers import EnsembleRetriever

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print("DONE LOADING CLIP MODEL")

def connect_to_db ():
    host = ''
    port = ''
    username = ''
    password = ''
    database_schema = ''

    connection_string = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database_schema}"
    # print("done")

    # Connect to PostgreSQL database
    conn = psycopg2.connect(database=database_schema, user=username, host=host, port=port, password=password)
    cur = conn.cursor()
    print("Connected to Vector DB")
    return connection_string, cur, conn


# Step 1: Load the LLM
def load_llm():
    ollama = Ollama(
        base_url='http://ollama:11434',
        model='gemma:latest',
        temperature=0.2
    )

    print("LOADED OLLAMA")
    return ollama


# Extract the audio from the video
def audio_extraction(video_path, mp3_file):
    # Define the input video file and output audio file
    mp4_file = video_path
    mp3_file = mp3_file
    # Load the video clip
    video_clip = VideoFileClip(mp4_file)
    # Extract the audio from the video clip
    audio_clip = video_clip.audio
    # Write the audio to a separate file
    audio_clip.write_audiofile(mp3_file)
    # Close the video and audio clips
    video_clip.close()
    audio_clip.close()
    print("Audio extraction successful!")

# Transcript the audio
def speach_to_text(mp3_file, text_path):
    # Choose a model size (e.g., 'tiny', 'base', 'small', 'medium', 'large')
    model = whisper.load_model("base")
    audio_file_path = mp3_file
    result = model.transcribe(audio_file_path)
    transcribed_text = result["text"]
    # print("Transcription:", transcribed_text)

    # Define the output file path
    output_file_path = text_path

    # Save the transcription result to a text file
    with open(output_file_path, 'w') as file:
        file.write(transcribed_text)
    print(f"Transcription saved to {output_file_path}")

    print("DONE Transcription")
    return result["segments"]

# Step 3: Extract one frame per second
def extract_video_frames(video_path, target_fps=1):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / original_fps
    frames = []

    for sec in range(int(duration_sec)):
        frame_idx = int(sec * original_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frames.append({
            "second": sec,
            "frame": frame
        })

    cap.release()
    print(f"Extracted {len(frames)} frames.")
    return frames

# Step 4: Embed transcript text using CLIP
def embed_text(text: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokens = clip.tokenize([text]).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features.cpu().numpy().flatten()
        print("DONE TEXT EMBEDDING")
        return features / np.linalg.norm(features)

# Step 5: Embed image/frame using CLIP
def embed_frame(frame):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = PILImage.fromarray(frame)
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encode_image(image_input)
        features = features.cpu().numpy().flatten()
        print("DONE FRAME EMBEDDING")
        return features / np.linalg.norm(features)

# Step 4: Align Whisper segments directly and embed text + closest frame
def align_and_embed_by_segments(video_path, transcript_segments):
    frames = extract_video_frames(video_path)
    data = []

    for seg in tqdm(transcript_segments, desc="Processing Whisper segments"):
        start_time = int(seg["start"])
        text = seg["text"]

        # Get the nearest frame to the segment start time
        matched_frame = next((f["frame"] for f in frames if f["second"] == start_time), None)

        if matched_frame is None:
            print(f"⚠️ No frame found at {start_time}s for segment: {text}")
            continue

        frame_embedding = embed_frame(matched_frame)
        text_embedding = embed_text(text)
        combined_embedding = (frame_embedding + text_embedding) / 2

        data.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": text,
            "frame": matched_frame,
            "frame_embedding": frame_embedding,
            "text_embedding": text_embedding,
            "combined_embedding": combined_embedding
        })

    print(f"DONE ALIGNING")
    print(f"✅ Embedded {len(data)} segments.")
    return data

# Save to PGVector DB
def save_video_embeddings_to_pgvector(embedding_data: list, connection_string):
    documents = []
    embeddings = []

    for entry in embedding_data:
        metadata = {
            "start": entry["start"],
            "end": entry["end"],
            "text": entry["text"]
        }

        doc = Document(
            page_content=entry["text"] or "[No Transcript]",
            metadata=metadata
        )

        documents.append(doc)
        embeddings.append(entry["combined_embedding"].tolist())

    vectorstore = PGVector(
        connection_string=connection_string,
        collection_name="",
        distance_strategy=DistanceStrategy.COSINE,
        embedding_function=None
    )

    vectorstore.add_embeddings(
        texts=[doc.page_content for doc in documents],
        embeddings=embeddings,
        metadatas=[doc.metadata for doc in documents]
    )

    print(f"✅ Saved {len(documents)} video segments into PGVector 'video_segments'")

# Retrieve from PGVector using same CLIP model
class CLIPTextEmbedder:
    def __init__(self, model):
        self.model = model

    def embed_query(self, text: str):
        return embed_text(text)


def retrieve_segments(connection_string, top_k: int = 10):
    embedder = CLIPTextEmbedder(model)
    store = PGVector(
        connection_string=connection_string,
        embedding_function=embedder,
        collection_name="",
        distance_strategy=DistanceStrategy.COSINE
    )
    print("✅ Connected to PGVector")

    retriever = store.as_retriever(search_kwargs={"k": top_k})
    query = "What seasonings are used to flavor the chicken breasts?"
    docs = retriever.invoke(query)

    print(f"\n Top {len(docs)} results for query: {query}\n")

    for idx, doc in enumerate(docs):
        start = doc.metadata.get("start")
        end = doc.metadata.get("end")

        if start is not None and end is not None:
            start_min = int(start) // 60
            start_sec = int(start) % 60
            end_min = int(end) // 60
            end_sec = int(end) % 60
            time_str = f"[{start_min:02d}m{start_sec:02d}s] -> [{end_min:02d}m{end_sec:02d}s]"
        else:
            time_str = "[Time not available]"
        
        print(f"--- Segment {idx + 1} ({time_str}) ---")
        print(doc.page_content)
        print("-" * (25 + len(time_str)))
        
# Hybrid Retriever (MMR + keyword)
def hybrid_retriever(embeddings, connection_string):
    store = PGVector(
        connection_string=connection_string,
        embedding_function=embeddings,
        collection_name="documents",
        distance_strategy=DistanceStrategy.COSINE
    )

    print("Connected to PGVector")

    dense = store.as_retriever(search_type="mmr", search_kwargs={"k": 20})
    sparse = store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 5, "score_threshold": 0.6})

    retriever = EnsembleRetriever(retrievers=[dense, sparse], weights=[0.7, 0.3])

    #query = "What seasonings are used to flavor the chicken breasts?" # Example query
    #docs = retriever.invoke(query)
    
    return retriever



# Initialize the RAG Pipeline
def rag_pipline(ollama, retriever, query):

    # To show ARABIC in the terminal
    def contains_arabic(text: str) -> bool:
        return bool(re.search(r'[\u0600-\u06FF]', text))

    def for_display(text: str) -> str:
        if contains_arabic(text):
            return get_display(arabic_reshaper.reshape(text))
        return text # No reshaping for English

    prompt_RAG = """
        You are a helpful and accurate assistant helping the user analyze and summarize video content that has been embedded and retrieved
        Follow these instructions carefully:
        
        1. **Use only the retrieved context**: Do not hallucinate, guess, or rely on external knowledge. Your answer must strictly reflect
        2. **Respect video timestamps**: Each content snippet includes a 'start' and 'end' time (in seconds). When relevant, cite these to
        3. **Multimodal awareness**: Assume the context combines both audio (spoken text) and visual frames. Even if not explicitly stated,
        4. **Handle informal or dense speech**: The transcript may be casual, messy, or unstructured. Parse carefully to extract clear fact
        5. **Preserve factual details**: Do not change or paraphrase important values like quantities, names, ingredients, timestamps, or
        6. **Language awareness**: Automatically detect whether the context is in English or Arabic. Respond in the **same language** as the
        7. **If multiple relevant segments are found**: Combine their insights into a single coherent answer. You may refer to multiple tim
        8. **If no answer is found**: Clearly state that the retrieved content does not contain the information needed.
        
        Question:
        {question}
        Retrieved Segments:
        {context}
        
        Your Answer:
        """

    prompt_RAG_tempate = PromptTemplate(
        template=prompt_RAG, input_variables=["question", "context"]
    )

    qa_chain_kwargs = {"prompt": prompt_RAG_tempate}
    qa_chain = RetrievalQA.from_chain_type(
        llm=ollama, chain_type="stuff", chain_type_kwargs=qa_chain_kwargs, retriever=retriever, return_source_documents=True
    )

    query = query
    responses = qa_chain.invoke({"query": query}, callbacks=None)

    source_documents = responses["source_documents"]
    # print(source_documents)
    source_content = [doc.page_content for doc in source_documents]
    source_metadata = [doc.metadata for doc in source_documents]

    # Construct a single string with the LLM output and the source titles and urls
    def construct_result_with_sources(responses, source_content, source_metadata):
        result = responses["result"]
        result += "\n\nSources used:"

        for i in range(len(source_content)):
            metadata = source_metadata[i]
            start = metadata.get("start")
            end = metadata.get("end")

            if start is not None and end is not None:
                start_min = int(start) // 60
                start_sec = int(start) % 60
                end_min = int(end) // 60
                end_sec = int(end) % 60
                time_str = f"[{start_min:02d}m{start_sec:02d}s] -> [{end_min:02d}m{end_sec:02d}s]"
            else:
                time_str = "Unknown timestamp"

            result += f"\n\n\u25b6 Source {i + 1}"
            result += f"\n\t\u23f1 Time: {time_str}"
            # result += f"\n\t\u1f4c4 Type: {metadata.get('document_type', 'Unknown')}"
            result += f"\n\tText: {source_content[i].strip()}"

        return result

    response = construct_result_with_sources(responses, source_content, source_metadata)
    print(for_display(response))


if __name__ == "__main__":
    # Ask user for mode
    mode = input("Do you want to [1] Chat with existing video or [2] Add new video? Enter 1 or 2: ").strip()

    # Connect to database
    connection_string, cur, conn = connect_to_db()

    if mode == "1":
        # Chat with existing video
        # video_path = "./Documentation/Video_to_text/Creamy Cajun Chicken Pasta _ How Make Cajun Chicken Pasta.mp4"
        retriever = hybrid_retriever(connection_string)
        ollama = load_llm()

        while True:
            query = input("\n Enter your question about the video: ")
            rag_pipline(ollama, retriever, query)

            again = input("Do you want to ask another question? (y/n): ").strip().lower()
            if again != 'y':
                print("Goodbye!")
                break

    elif mode == "2":
        # New video processing
        video_path = input(" Enter the full path to the video file (e.g., ./myvideo.mp4): ").strip()
        filename = os.path.split(os.path.basename(video_path))[0]
        mp3_file = f"./Documentation/Video_to_text/{filename}.mp3"
        text_path = f"./Documentation/Video_to_text/{filename}_transcript.txt"

        print(" Extracting audio...")
        audio_extraction(video_path, mp3_file)

        print(" Transcribing...")
        segments = speach_to_text(mp3_file, text_path)

        print(" Embedding...")
        embedding_data = align_and_embed_by_segments(video_path, segments)

        print(" Saving to PGVector...")
        save_video_embeddings_to_pgvector(embedding_data, connection_string)

        retriever = hybrid_retriever(connection_string)
        ollama = load_llm()

        while True:
            query = input("\n Enter your question about the video: ")
            rag_pipline(ollama, retriever, query)

            again = input("Do you want to ask another question? (y/n): ").strip().lower()
            if again != 'y':
                print(" Goodbye!")
                break
    else:
        print(" Invalid input. Please enter 1 or 2.")

# Take the query as a Voice
# def record_voice_to_wav(output_path="./Documentation/Video_to_text/output.wav", timeout=10, phrase_time_limit=15):

#     recognizer = sr.Recognizer()


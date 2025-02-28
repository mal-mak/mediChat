import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
from typing import Dict, List, Tuple
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.table import Table
from dotenv import load_dotenv
import json

load_dotenv()

# Constants
HOST = "http://0.0.0.0:8181"
DOWNLOADED_LOCAL_DIRECTORY = "./downloaded_files"
CSV_FILE_PATH = os.path.join(DOWNLOADED_LOCAL_DIRECTORY, "medquad.csv")
EVAL_DIR = "/Users/malmak/Dauphine/M2/Generative AI/mediChat/eval"
RESULTS_DIR = os.path.join(EVAL_DIR, "results")
NUM_TEST_SAMPLES = 10
SAMPLE_SEED = 42

# Initialize sentence transformer for semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')
console = Console()

def load_test_data() -> pd.DataFrame:
    """Load and prepare test data."""
    if not os.path.exists(CSV_FILE_PATH):
        raise FileNotFoundError(f"Dataset not found at {CSV_FILE_PATH}")
    
    df = pd.read_csv(CSV_FILE_PATH)
    return df.sample(n=NUM_TEST_SAMPLES, random_state=SAMPLE_SEED)

def get_chatbot_response(question: str) -> Tuple[Dict, float]:
    """Get response from the chatbot API and measure response time."""
    start_time = time.time()
    
    try:
        # Get sources
        sources_response = requests.post(
            f"{HOST}/get_sources",
            json={
                "question": question,
                "temperature": 0.2,
                "similarity_threshold": 0.75,
                "max_sources": 4,
                "language": "English",
                "documents": [],
                "previous_context": []
            },
            timeout=30
        )
        
        if sources_response.status_code != 200:
            raise Exception(f"Sources API Error: {sources_response.status_code}")
            
        sources = sources_response.json()
        
        # Get answer
        answer_response = requests.post(
            f"{HOST}/answer",
            json={
                "question": question,
                "temperature": 0.2,
                "similarity_threshold": 0.75,
                "max_sources": 4,
                "language": "English",
                "documents": sources,
                "previous_context": []
            },
            timeout=30
        )
        
        if answer_response.status_code != 200:
            raise Exception(f"Answer API Error: {answer_response.status_code}")
        
        response_time = time.time() - start_time
        return {"sources": sources, "message": answer_response.json()["message"]}, response_time
        
    except Exception as e:
        print(f"Error in API call: {str(e)}")
        return {"sources": [], "message": ""}, 0.0

def calculate_answer_similarity(answer: str, source_answers: List[str]) -> float:
    """Calculate similarity between chatbot answer and source answers."""
    if not answer or not source_answers:
        return 0.0
    
    # Embed chatbot answer and source answers
    answer_embedding = model.encode([answer])[0]
    source_embeddings = model.encode(source_answers)
    
    # Calculate similarities with each source answer
    similarities = cosine_similarity([answer_embedding], source_embeddings)[0]
    
    # Return the highest similarity score
    return float(np.max(similarities))

def save_detailed_comparison(
    output_file: str,
    question: str,
    chatbot_answer: str,
    source_answers: List[str],
    similarity_score: float
) -> None:
    """Save detailed comparison to a file."""
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"Question: {question}\n")
        f.write(f"\nChatbot Answer: {chatbot_answer}\n")
        f.write("\nSource Answers:\n")
        for i, src_answer in enumerate(source_answers, 1):
            f.write(f"\nSource {i}: {src_answer[:500]}...\n")
        f.write(f"\nSimilarity Score: {similarity_score:.3f}\n")

def run_evaluation() -> pd.DataFrame:
    """Run the evaluation and return results."""
    # Create directories if they don't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    test_data = load_test_data()
    results = []
    
    # Update file path to use results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_output = os.path.join(RESULTS_DIR, f"detailed_evaluation_{timestamp}.txt")
    
    for idx, row in test_data.iterrows():
        console.print(f"\n[bold cyan]Testing sample {idx + 1}/{len(test_data)}[/bold cyan]")
        try:
            response, response_time = get_chatbot_response(row["question"])
            sources = response["sources"]
            chatbot_answer = response["message"]
            
            source_answers = [source["metadata"]["answer"] for source in sources]
            answer_similarity = calculate_answer_similarity(chatbot_answer, source_answers)
            
            # Save detailed comparison to file
            save_detailed_comparison(
                detailed_output,
                row["question"],
                chatbot_answer,
                source_answers,
                answer_similarity
            )
            
            results.append({
                "question": row["question"],
                "chatbot_answer": chatbot_answer,
                "source_answers": source_answers,
                "answer_similarity": answer_similarity,
                "response_time": response_time
            })
            
        except Exception as e:
            console.print(f"[bold red]Error processing sample {idx + 1}: {str(e)}[/bold red]")
            results.append({
                "question": row["question"],
                "chatbot_answer": "",
                "source_answers": [],
                "answer_similarity": 0.0,
                "response_time": 0.0
            })
    
    console.print(f"\n[bold blue]Detailed comparisons saved to: {detailed_output}[/bold blue]")
    return pd.DataFrame(results)

def display_results(results: pd.DataFrame) -> None:
    """Display evaluation results in a formatted table."""
    table = Table(title="Chatbot Evaluation Results")
    
    metrics = ["answer_similarity", "response_time"]
    for metric in metrics:
        table.add_column(metric.replace("_", " ").title(), justify="right")
    
    # Add mean scores row
    means = results[metrics].mean()
    table.add_row(
        *[f"{means[metric]:.3f}" for metric in metrics],
        style="bright_green"
    )
    
    console.print(table)

def main():
    console.print("[bold green]Starting chatbot evaluation...[/bold green]")
    
    # Run evaluation
    results = run_evaluation()
    
    # Display results
    display_results(results)
    
    # Save detailed results in JSON format with updated path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(RESULTS_DIR, f"evaluation_results_{timestamp}.json")
    
    # Convert DataFrame to dictionary format
    results_dict = {
        "metadata": {
            "timestamp": timestamp,
            "num_samples": NUM_TEST_SAMPLES,
            "mean_scores": {
                "answer_similarity": float(results["answer_similarity"].mean()),
                "response_time": float(results["response_time"].mean())
            }
        },
        "evaluations": []
    }
    
    # Add individual evaluations
    for _, row in results.iterrows():
        evaluation = {
            "question": row["question"],
            "chatbot_answer": row["chatbot_answer"],
            "source_answers": row["source_answers"],
            "metrics": {
                "answer_similarity": float(row["answer_similarity"]),
                "response_time": float(row["response_time"])
            }
        }
        results_dict["evaluations"].append(evaluation)
    
    # Save to JSON file with pretty printing
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    console.print(f"\n[bold green]Detailed results saved to {output_file}[/bold green]")

if __name__ == "__main__":
    main()

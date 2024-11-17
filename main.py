import json
from pathlib import Path
from typing import Dict

import torch
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredHTMLLoader
from langchain_huggingface import HuggingFaceEmbeddings
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)


def _is_text_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            # Try reading a small chunk from the file.
            file.read(1024)
            return True
    except (UnicodeDecodeError, IsADirectoryError, FileNotFoundError):
        # File is not a file text file, doesn't exist, or it is a directory.
        return False


class RAGManager:
    def __init__(self, base_directory):
        # Create base directory if it doesn't exist
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(parents=True, exist_ok=True)

        # Create cache directory
        self.cache_dir = self.base_directory / ".cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.vectordb_dir = self.base_directory / ".vectordb"

        # Check for GPU availability. If not available, go with CPU.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.debug(f"Using device: {self.device}")
        logging.debug(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logging.debug(f"GPU Name: {torch.cuda.get_device_name(0)}")
            logging.debug(f"GPU count: {torch.cuda.device_count()}")

        # Initialize Embedding model.
        # NOTE: Use "all-mpnet-base-v2" if you have good processing environment, else use "all-MiniLM-L6-v2".
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": self.device}
        )

        # Configure File Loaders
        self.loader_mapping = {
            ".pdf": PyPDFLoader,
            ".docx": UnstructuredWordDocumentLoader,
            ".html": UnstructuredHTMLLoader,
        }

    def process_all_projects(self):
        projects = [directory for directory in self.base_directory.iterdir()
                    if directory.is_dir()
                    and directory != self.cache_dir
                    and directory != self.vectordb_dir]
        logging.debug(f"Found {len(projects)} projects to process.")

        for project in projects:
            self.process_project(project)

    def process_project(self, project):
        logging.debug(f"Processing project: {project.name}")
        project_dir = self.base_directory / project
        repos_dir = project_dir / "repos"
        docs_dir  =  project_dir / "docs"

        # Create repos and docs directories if they don't exist
        repos_dir.mkdir(parents=True, exist_ok=True)
        docs_dir.mkdir(parents=True, exist_ok=True)

        # Load existing cache
        project_cache = self._load_project_cache(project)

        # Get all project repositories.
        repos = [repo for repo in repos_dir.iterdir() if repo.is_dir()]
        logging.debug(f"Found {len(repos)} repositories to process for project {project.name}.")

        all_files = []
        for repo in repos:
            repo_files = self._get_supported_files(repo)
            logging.debug(f"Found {len(repo_files)} supported files in repository {repo.name}.")
            all_files.extend(repo_files)

        if docs_dir.exists():
            docs_files = self._get_supported_files(docs_dir)
            logging.debug(f"Found {len(docs_files)} supported files in docs directory.")
            all_files.extend(docs_files)

        logging.debug(f"Total files to process: {len(all_files)}")

    def _load_project_cache(self, project) -> Dict[str, str]:
        cache_file = self.cache_dir / f"{project}_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logging.ERROR(f"Could not load cache for {project}: {e}")
        return {}

    def _get_supported_files(self, repo):
        supported_files = []
        try:
            for file in repo.rglob("*"):
                # Skip hidden files and directories
                if file.name.startswith("."):
                    continue

                if file.is_file():
                    file_extension = file.suffix.lower()
                    if file_extension in self.loader_mapping:
                        supported_files.append(file)
                    elif _is_text_file(file):
                        supported_files.append(file)
                    else:
                        logging.ERROR(f"Unsupported file type: {file_extension}")
        except Exception as e:
            logging.ERROR(f"Could not get supported files for {repo}: {e}")

        return supported_files

if __name__ == "__main__":
    rag_manager = RAGManager(base_directory="c:\\ragmanager\\userdata")
    rag_manager.process_all_projects()


    logging.debug("Bye Bye")

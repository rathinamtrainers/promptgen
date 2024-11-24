import inspect
import os
import hashlib
import stat
from pathlib import Path
import json
from typing import List, Dict
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from tqdm import tqdm

import torch


class ProjectRAGManager:
    def __init__(self, base_directory: str = "./userdata"):
        self.base_directory = Path(base_directory)
        # Create base directory if it does not exist
        self.base_directory.mkdir(parents=True, exist_ok=True)

        self.cache_dir = self.base_directory / ".cache"
        self.cache_dir.mkdir(exist_ok=True)

        # Initialize embedding model with GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU count: {torch.cuda.device_count()}")
            print(f"GPU name: {torch.cuda.get_device_name(0)}")

        # TODO: Test with "all-mpnet-base-v2"
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': device}
        )

        # Configure file loaders with updated loader classes
        self.loader_mapping = {
            ".txt": TextLoader,
            ".py": TextLoader,
            ".js": TextLoader,
            ".java": TextLoader,
            ".cpp": TextLoader,
            ".h": TextLoader,
            ".md": UnstructuredMarkdownLoader,
            ".html": UnstructuredHTMLLoader,
            ".pdf": PyPDFLoader,
            ".docx": UnstructuredWordDocumentLoader,
            ".c": TextLoader,
            ".hpp": TextLoader,
            ".css": TextLoader,
            ".json": TextLoader,
            ".yaml": TextLoader,
            ".yml": TextLoader,
            ".xml": TextLoader,
            ".rst": TextLoader,
            ".ini": TextLoader,
            ".conf": TextLoader,
            ".properties": TextLoader,
            ".sql": TextLoader,
        }

        # Text splitter configuration
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    #@retry(tries=3, delay=2, backoff=2, exceptions=(PermissionError, FileNotFoundError))
    def _load_document_with_backoff(self, file_path: Path, loader_class):
        """Load document using the specified loader class with backoff on permission errors."""
        try:
            # if file_path.name == 'docusaurus.config.js':
            #     print("Debug from here")

            # Check if 'autodetect_encoding' is a parameter of the loader_class
            init_signature = inspect.signature(loader_class)
            if 'autodetect_encoding' in init_signature.parameters:
                loader = loader_class(str(file_path), autodetect_encoding=True)
            else:
                loader = loader_class(str(file_path))

            return loader.load()
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return ""

    #@retry(tries=3, delay=2, backoff=2, exceptions=(PermissionError, FileNotFoundError))
    def _get_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file content and modification time, with retries on permission issues."""
        try:
            # Ensure file is readable
            os.chmod(file_path, stat.S_IREAD)
            with open(file_path, "rb") as f:
                content = f.read()
            mod_time = str(os.path.getmtime(file_path)).encode()
            return hashlib.md5(content + mod_time).hexdigest()
        except Exception as e:
            print(f"Warning: Could not hash file {file_path}: {str(e)}")
            raise  # Reraise to trigger retry mechanism

    def _load_file_cache(self, project_name: str) -> Dict[str, str]:
        """Load the cache of processed files for a project."""
        cache_file = self.cache_dir / f"{project_name}_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load cache for {project_name}: {str(e)}")
        return {}

    def _save_file_cache(self, project_name: str, cache: Dict[str, str]):
        """Save the cache of processed files for a project."""
        try:
            cache_file = self.cache_dir / f"{project_name}_cache.json"
            with open(cache_file, "w") as f:
                json.dump(cache, f)
        except Exception as e:
            print(f"Warning: Could not save cache for {project_name}: {str(e)}")

    def _get_supported_files(self, directory: Path) -> List[Path]:
        """Get all supported files in the directory."""
        supported_files = []
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in self.loader_mapping:

                    # TODO: Fix the algorithm to filter out hidden directories
                    if not any(part.startswith('.') for part in file_path.parts[:-1]):  # Skip hidden directories
                        supported_files.append(file_path)
        except Exception as e:
            print(f"Error scanning directory {directory}: {str(e)}")
        return supported_files

    def _get_repositories(self, project_dir: Path) -> List[Path]:
        """Get all repository directories under the project's repos directory."""
        repos_dir = project_dir / "repos"
        if not repos_dir.exists():
            repos_dir.mkdir(parents=True, exist_ok=True)
            return []

        return [d for d in repos_dir.iterdir() if d.is_dir()]

    def process_project(self, project_name: str):
        """Process or update a single project's documents from multiple repositories."""
        print(f"\nProcessing project: {project_name}")
        project_dir = self.base_directory / project_name
        repos_dir = project_dir / "repos"
        docs_dir = project_dir / "docs"

        # Ensure project directories exist
        repos_dir.mkdir(parents=True, exist_ok=True)
        docs_dir.mkdir(parents=True, exist_ok=True)

        # Load existing cache
        file_cache = self._load_file_cache(project_name)

        # Get all repositories
        repositories = self._get_repositories(project_dir)
        print(f"Found {len(repositories)} repositories in project {project_name}")

        # Get all supported files from repositories and docs
        all_files = []
        for repo_dir in repositories:
            repo_files = self._get_supported_files(repo_dir)
            print(f"Found {len(repo_files)} files in repository {repo_dir.name}")
            all_files.extend(repo_files)

        if docs_dir.exists():
            docs_files = self._get_supported_files(docs_dir)
            print(f"Found {len(docs_files)} files in docs directory")
            all_files.extend(docs_files)

        # Process new and modified files
        new_documents = []
        updated_cache = {}

        print(f"Processing {len(all_files)} total files")
        for file_path in tqdm(all_files, desc="Processing files"):
            current_hash = self._get_file_hash(file_path)
            rel_path = str(file_path.relative_to(self.base_directory))

            # Determine if file is from a repository and which one
            repo_name = None
            for repo_dir in repositories:
                if str(file_path).startswith(str(repo_dir)):
                    repo_name = repo_dir.name
                    break

            if rel_path not in file_cache or file_cache[rel_path] != current_hash:
                try:
                    loader_class = self.loader_mapping[file_path.suffix.lower()]
                    documents = self._load_document_with_backoff(file_path, loader_class)

                    # Add metadata including repository information
                    for doc in documents:
                        metadata = {
                            'project': project_name,
                            'file_path': rel_path,
                            'last_updated': datetime.now().isoformat()
                        }
                        if repo_name:
                            metadata['repository'] = repo_name
                        doc.metadata.update(metadata)

                    chunks = self.text_splitter.split_documents(documents)
                    new_documents.extend(chunks)
                    updated_cache[rel_path] = current_hash
                except Exception as e:
                    print(f"\nError processing {file_path}: {str(e)}")
            else:
                # File unchanged, keep the old hash
                updated_cache[rel_path] = current_hash

        # Save updated cache
        self._save_file_cache(project_name, updated_cache)

        # Update vector store if there are new documents
        if new_documents:
            print(f"\nAdding {len(new_documents)} new document chunks to vector store")
            vector_store = Chroma(
                collection_name=f"project_{project_name}",
                embedding_function=self.embeddings,
                persist_directory=str(self.base_directory / "vectorstore")
            )

            # Add new documents to the vector store
            for document in tqdm(new_documents, desc="Adding docs to vector db..."):
                try:
                    vector_store.add_documents([document])  # Pass a single document as a list
                except Exception as e:
                    print(f"Error adding document: {str(e)}")
            print("Vector store updated successfully")
        else:
            print("\nNo new documents to process")

    def process_all_projects(self):
        """Process all projects in the userdata directory."""
        projects = [d for d in self.base_directory.iterdir()
                    if d.is_dir() and d != self.cache_dir and d.name != "vectorstore"]
        print(f"Found {len(projects)} projects to process")

        for project_dir in projects:
            self.process_project(project_dir.name)

    def search_project(self, project_name: str, query: str, k: int = 5) -> List[Document]:
        """Search within a specific project."""
        try:
            vector_store = Chroma(
                collection_name=f"project_{project_name}",
                embedding_function=self.embeddings,
                persist_directory=str(self.base_directory / "vectorstore")
            )
            return vector_store.similarity_search(query, k=k)
        except Exception as e:
            print(f"Error searching project {project_name}: {str(e)}")
            return []

    def search_all_projects(self, query: str, k: int = 5) -> Dict[str, List[Document]]:
        """Search across all projects."""
        results = {}
        projects = [d for d in self.base_directory.iterdir()
                    if d.is_dir() and d != self.cache_dir and d.name != "vectorstore"]

        for project_dir in projects:
            project_name = project_dir.name
            project_results = self.search_project(project_name, query, k)
            if project_results:
                results[project_name] = project_results

        return results

    def list_repositories(self, project_name: str) -> List[str]:
        """List all repositories in a project."""
        project_dir = self.base_directory / project_name
        repos = self._get_repositories(project_dir)
        return [repo.name for repo in repos]

    def get_repository_stats(self, project_name: str) -> Dict[str, Dict]:
        """Get statistics for each repository in a project."""
        project_dir = self.base_directory / project_name
        repositories = self._get_repositories(project_dir)
        stats = {}

        for repo_dir in repositories:
            repo_files = self._get_supported_files(repo_dir)
            file_types = {}
            for file_path in repo_files:
                file_type = file_path.suffix.lower()
                file_types[file_type] = file_types.get(file_type, 0) + 1

            stats[repo_dir.name] = {
                'total_files': len(repo_files),
                'file_types': file_types
            }

        return stats


# Example usage
if __name__ == "__main__":
    # Initialize the RAG manager

    rag_manager = ProjectRAGManager(base_directory="c:\\ragmanager\\userdata")

    # Process all projects
    rag_manager.process_all_projects()


    # Example search across all projects
    search_query = "Write a program to demonstrate the usage of HuggingFaceEmbeddings in LangChain"
    results = rag_manager.search_all_projects(search_query)

    # Display results
    for project_name, documents in results.items():
        print(f"\nResults from project {project_name}:")
        for doc in documents:
            repo_info = f" (Repository: {doc.metadata['repository']})" if 'repository' in doc.metadata else ""
            print(f"- {doc.metadata['file_path']}{repo_info}: {doc.page_content}...")


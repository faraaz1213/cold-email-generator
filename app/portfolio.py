import os
import pandas as pd
import chromadb
import uuid


class Portfolio:
    def __init__(self, file_path=None):
        # ✅ Step 1: Auto-detect the CSV file path
        if file_path is None:
            base_dir = os.path.dirname(__file__)
            file_path = os.path.join(base_dir, "resource", "my_portfolio.csv")

        # ✅ Step 2: Print to verify actual file path (for debugging)
        print(f"📂 Loading portfolio file from: {file_path}")

        # ✅ Step 3: Check if file actually exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Portfolio file not found at: {file_path}")

        # ✅ Step 4: Load CSV
        self.file_path = file_path
        self.data = pd.read_csv(file_path)

        # ✅ Step 5: Setup ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path='vectorstore')
        self.collection = self.chroma_client.get_or_create_collection(name="portfolio")

    def load_portfolio(self):
        """Add data from CSV to the Chroma vector store if not already present."""
        if self.collection.count() == 0:
            for _, row in self.data.iterrows():
                self.collection.add(
                    documents=[row["Techstack"]],
                    metadatas={"links": row["Links"]},
                    ids=[str(uuid.uuid4())]
                )
            print("✅ Portfolio loaded into vector database.")
        else:
            print("ℹ️ Portfolio already loaded.")

    def query_links(self, skills):
        """Query the portfolio vector DB for related links based on skills."""
        if not skills:
            return []
        result = self.collection.query(query_texts=skills, n_results=2)
        return result.get("metadatas", [])

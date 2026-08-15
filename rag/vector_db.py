import chromadb
from chromadb.utils import embedding_functions
import os
import uuid

# Base directory for DB
DB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

class AdaptiveMemory:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=DB_DIR)
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        
        self.collection = self.client.get_or_create_collection(
            name="crm_corrections",
            embedding_function=self.embedding_fn
        )

    def add_correction(self, text: str, correct_label: str, customer_context: str = ""):
        """
        Stores a human-corrected ticket mapping with optional customer context.
        Uses UUID for IDs to prevent collisions on concurrent adds.
        """
        from datetime import datetime
        # BUG FIX: was using collection.count()+1 which causes ID collisions
        # under concurrent requests. Use UUID instead.
        unique_id = f"id_{uuid.uuid4().hex}"
        self.collection.add(
            documents=[text],
            metadatas=[{
                "label": correct_label,
                "customer_context": customer_context,
                "timestamp": datetime.now().isoformat()
            }],
            ids=[unique_id]
        )
        print(f"✅ Correction added: '{text[:60]}' → {correct_label}")

    def query_memory(self, text: str, threshold: float = 0.4):
        """
        Searches for a similar corrected ticket in memory.
        Returns (label, distance) if a sufficiently similar match is found, else (None, None).
        Threshold lowered to 0.4 (from 0.5) for better recall — main_enhanced checks dist<0.3
        for high-confidence RAG hits, so this still filters out weak matches.
        """
        # Guard: empty collection causes ChromaDB error
        if self.collection.count() == 0:
            return None, None

        results = self.collection.query(
            query_texts=[text],
            n_results=1
        )
        
        if (results['documents'] and 
                results['distances'] and 
                results['distances'][0]):
            distance = results['distances'][0][0]
            if distance < threshold:
                label = results['metadatas'][0][0]['label']
                return label, distance
                
        return None, None

    def get_stats(self) -> dict:
        """Return memory statistics."""
        return {
            "total_memories": self.collection.count(),
            "collection_name": self.collection.name,
        }

if __name__ == "__main__":
    memory = AdaptiveMemory()
    memory.add_correction("I want to cancel my account but keep the data", "Technical Support")
    label, dist = memory.query_memory("How do I cancel account but save my data?")
    print(f"Memory Retrieval: {label} (Distance: {dist})")
    print(f"Stats: {memory.get_stats()}")

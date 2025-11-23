"""Incremental update and refresh pipeline."""

import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime

from flip.document_processing.loader import DocumentLoader, Document
from flip.core.exceptions import FlipException


class DocumentTracker:
    """Track document changes for incremental updates."""
    
    def __init__(self, tracking_file: Optional[Path] = None):
        """
        Initialize document tracker.
        
        Args:
            tracking_file: Path to tracking file (default: ./flip_data/document_tracking.json)
        """
        self.tracking_file = tracking_file or Path("./flip_data/document_tracking.json")
        self.tracking_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing tracking data
        self.tracked_documents: Dict[str, Dict[str, Any]] = {}
        self._load_tracking_data()
    
    def _load_tracking_data(self):
        """Load tracking data from file."""
        if self.tracking_file.exists():
            try:
                with open(self.tracking_file, 'r') as f:
                    self.tracked_documents = json.load(f)
            except:
                self.tracked_documents = {}
    
    def _save_tracking_data(self):
        """Save tracking data to file."""
        try:
            with open(self.tracking_file, 'w') as f:
                json.dump(self.tracked_documents, f, indent=2)
        except Exception as e:
            raise FlipException(f"Failed to save tracking data: {str(e)}")
    
    def _compute_file_hash(self, filepath: str) -> str:
        """Compute hash of file content."""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return ""
    
    def _get_file_mtime(self, filepath: str) -> float:
        """Get file modification time."""
        try:
            return Path(filepath).stat().st_mtime
        except:
            return 0.0
    
    def track_document(self, filepath: str, chunk_ids: List[str]):
        """
        Track a document.
        
        Args:
            filepath: Path to document
            chunk_ids: List of chunk IDs for this document
        """
        file_hash = self._compute_file_hash(filepath)
        mtime = self._get_file_mtime(filepath)
        
        self.tracked_documents[filepath] = {
            "hash": file_hash,
            "mtime": mtime,
            "chunk_ids": chunk_ids,
            "indexed_at": datetime.now().isoformat()
        }
        
        self._save_tracking_data()
    
    def is_document_modified(self, filepath: str) -> bool:
        """
        Check if document has been modified since last indexing.
        
        Args:
            filepath: Path to document
            
        Returns:
            True if modified or new, False otherwise
        """
        if filepath not in self.tracked_documents:
            return True  # New document
        
        current_hash = self._compute_file_hash(filepath)
        tracked_hash = self.tracked_documents[filepath]["hash"]
        
        return current_hash != tracked_hash
    
    def get_chunk_ids(self, filepath: str) -> List[str]:
        """
        Get chunk IDs for a document.
        
        Args:
            filepath: Path to document
            
        Returns:
            List of chunk IDs
        """
        if filepath in self.tracked_documents:
            return self.tracked_documents[filepath]["chunk_ids"]
        return []
    
    def remove_document(self, filepath: str):
        """
        Remove document from tracking.
        
        Args:
            filepath: Path to document
        """
        if filepath in self.tracked_documents:
            del self.tracked_documents[filepath]
            self._save_tracking_data()
    
    def get_all_tracked_files(self) -> Set[str]:
        """Get set of all tracked file paths."""
        return set(self.tracked_documents.keys())
    
    def clear(self):
        """Clear all tracking data."""
        self.tracked_documents = {}
        if self.tracking_file.exists():
            self.tracking_file.unlink()


class IncrementalUpdater:
    """Handle incremental updates to the document index."""
    
    def __init__(
        self,
        loader: DocumentLoader,
        tracker: DocumentTracker,
        pipeline_orchestrator
    ):
        """
        Initialize incremental updater.
        
        Args:
            loader: Document loader
            tracker: Document tracker
            pipeline_orchestrator: Pipeline orchestrator for processing
        """
        self.loader = loader
        self.tracker = tracker
        self.pipeline = pipeline_orchestrator
    
    def scan_for_changes(
        self,
        directory: str,
        recursive: bool = True,
        file_extensions: Optional[List[str]] = None
    ) -> Dict[str, List[str]]:
        """
        Scan directory for changes.
        
        Args:
            directory: Directory to scan
            recursive: Whether to search subdirectories
            file_extensions: Optional file extensions filter
            
        Returns:
            Dictionary with 'new', 'modified', 'deleted' file lists
        """
        # Get current files
        current_files = set()
        for doc in self.loader.discover_files(directory, recursive, file_extensions):
            current_files.add(str(Path(doc).absolute()))
        
        # Get tracked files
        tracked_files = self.tracker.get_all_tracked_files()
        
        # Categorize changes
        new_files = list(current_files - tracked_files)
        deleted_files = list(tracked_files - current_files)
        
        # Check for modifications
        modified_files = []
        for filepath in current_files & tracked_files:
            if self.tracker.is_document_modified(filepath):
                modified_files.append(filepath)
        
        return {
            "new": new_files,
            "modified": modified_files,
            "deleted": deleted_files
        }
    
    def update_index(
        self,
        directory: str,
        vector_store,
        recursive: bool = True,
        file_extensions: Optional[List[str]] = None,
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Incrementally update the index.
        
        Args:
            directory: Directory to update from
            vector_store: Vector store to update
            recursive: Whether to search subdirectories
            file_extensions: Optional file extensions filter
            show_progress: Whether to show progress
            
        Returns:
            Dictionary with counts of added, updated, deleted documents
        """
        # Scan for changes
        changes = self.scan_for_changes(directory, recursive, file_extensions)
        
        stats = {
            "added": 0,
            "updated": 0,
            "deleted": 0,
            "unchanged": 0
        }
        
        # Handle deleted files
        for filepath in changes["deleted"]:
            chunk_ids = self.tracker.get_chunk_ids(filepath)
            if chunk_ids:
                # Delete chunks from vector store
                for chunk_id in chunk_ids:
                    try:
                        vector_store.delete([chunk_id])
                    except:
                        pass
                
                self.tracker.remove_document(filepath)
                stats["deleted"] += 1
                
                if show_progress:
                    print(f"🗑️  Deleted: {Path(filepath).name}")
        
        # Handle new and modified files
        files_to_process = changes["new"] + changes["modified"]
        
        if not files_to_process:
            if show_progress:
                print("✅ No changes detected. Index is up to date.")
            return stats
        
        if show_progress:
            print(f"📝 Processing {len(files_to_process)} changed documents...")
        
        for filepath in files_to_process:
            try:
                # Load document
                doc = self.loader.load_file(filepath)
                
                # If modified, delete old chunks first
                if filepath in changes["modified"]:
                    old_chunk_ids = self.tracker.get_chunk_ids(filepath)
                    if old_chunk_ids:
                        for chunk_id in old_chunk_ids:
                            try:
                                vector_store.delete([chunk_id])
                            except:
                                pass
                
                # Process document
                ids, embeddings, texts, metadatas = self.pipeline.process_documents([doc])
                
                # Add to vector store
                vector_store.add(
                    ids=ids,
                    embeddings=embeddings,
                    texts=texts,
                    metadatas=metadatas
                )
                
                # Track document
                self.tracker.track_document(filepath, ids)
                
                # Update stats
                if filepath in changes["new"]:
                    stats["added"] += 1
                    if show_progress:
                        print(f"➕ Added: {Path(filepath).name}")
                else:
                    stats["updated"] += 1
                    if show_progress:
                        print(f"🔄 Updated: {Path(filepath).name}")
                
            except Exception as e:
                if show_progress:
                    print(f"❌ Error processing {Path(filepath).name}: {str(e)}")
        
        if show_progress:
            print(f"\n✅ Update complete!")
            print(f"   Added: {stats['added']}")
            print(f"   Updated: {stats['updated']}")
            print(f"   Deleted: {stats['deleted']}")
        
        return stats

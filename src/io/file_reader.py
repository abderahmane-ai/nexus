from pathlib import Path
from typing import Optional


class FileReader:
    """Handles reading and basic preprocessing of text files."""
    
    def __init__(self, encoding: str = "utf-8", cleanup_whitespace: bool = True):
        """
        Initialize the file reader.
        
        Args:
            encoding: Text encoding to use when reading files
            cleanup_whitespace: Whether to normalize whitespace in the text
        """
        self.encoding = encoding
        self.cleanup_whitespace = cleanup_whitespace
    
    def read_file(self, path: str) -> str:
        """
        Read a text file and return its content as a string.
        
        Args:
            path: Path to the text file
            
        Returns:
            File content as a string
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            UnicodeDecodeError: If the file can't be decoded with the specified encoding
        """
        file_path = Path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        with open(file_path, "r", encoding=self.encoding) as file:
            text = file.read()
        
        if self.cleanup_whitespace:
            text = " ".join(text.split())  # Basic cleanup to remove extra whitespace
        
        return text
    
    def get_file_info(self, path: str) -> dict:
        """
        Get basic information about a file.
        
        Args:
            path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        file_path = Path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        return {
            "path": str(file_path.absolute()),
            "size_bytes": file_path.stat().st_size,
            "name": file_path.name,
            "suffix": file_path.suffix
        }
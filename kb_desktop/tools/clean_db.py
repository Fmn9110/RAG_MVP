import sys
import os

sys.path.append(os.path.join(os.getcwd(), 'kb_desktop'))
from core.storage import DBManager

def clean_database():
    db = DBManager(db_path="kb_desktop/data/kb.sqlite")
    conn = db.get_connection()
    cursor = conn.cursor()
    
    # Delete all chunks
    cursor.execute('DELETE FROM chunks')
    print("Deleted all chunks")
    
    # Delete all documents
    cursor.execute('DELETE FROM documents')
    print("Deleted all documents")
    
    conn.commit()
    conn.close()
    
    print("\nDatabase cleaned successfully!")
    print("You can now import fresh documents.")

if __name__ == "__main__":
    response = input("This will delete ALL documents and chunks. Continue? (yes/no): ")
    if response.lower() == 'yes':
        clean_database()
    else:
        print("Cancelled.")

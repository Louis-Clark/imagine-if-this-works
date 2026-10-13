import os
import random
import string
from google.oauth2 import service_account
from googleapiclient.discovery import build

SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY")
DOCUMENT_ID = os.environ.get("GOOGLE_DOC_ID")
import os
import random
import string
from google.oauth2 import service_account
from googleapiclient.discovery import build

SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY")
DOCUMENT_ID = os.environ.get("GOOGLE_DOC_ID")

def generate_garbled_text():
    # Mix of letters, numbers, and wild symbols
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()_+-=[]{}|;':\",./<>?~`~# "
    length = random.randint(20, 50)
    return "".join(random.choice(alphabet) for _ in range(length)) + "\n"

def main():
    if not SERVICE_ACCOUNT_JSON or not DOCUMENT_ID:
        raise ValueError("Missing GOOGLE_SERVICE_ACCOUNT_KEY or GOOGLE_DOC_ID environment variables.")

    creds_info = service_account.Credentials.from_service_account_info(
        eval(SERVICE_ACCOUNT_JSON),
        scopes=["https://www.googleapis.com/auth/documents"]
    )
    service = build('docs', 'v1', credentials=creds_info)

    garbled_text = f"[GLITCH_BOT]: {generate_garbled_text()}"
    insert_index = 1

    # 1. Insert the garbled text at the top
    # 2. Update the text style to explicitly force white color
    requests = [
        {
            'insertText': {
                'location': {
                    'index': insert_index,
                },
                'text': garbled_text
            }
        },
        {
            'updateTextStyle': {
                'range': {
                    'startIndex': insert_index,
                    'endIndex': insert_index + len(garbled_text)
                },
                'textStyle': {
                    'foregroundColor': {
                        'color': {
                            'rgbColor': {
                                'red': 1.0,
                                'green': 1.0,
                                'blue': 1.0
                            }
                        }
                    }
                },
                'fields': 'foregroundColor'
            }
        }
    ]

    result = service.documents().batchUpdate(
        documentId=DOCUMENT_ID, body={'requests': requests}
    ).execute()
    print(f"Successfully injected white garbled text! Reply: {result}")

if __name__ == '__main__':
    main()
# Reliable public SVG / image URLs that accept external rendering
SAMPLE_IMAGES = [
    "https://upload.wikimedia.org/wikipedia/commons/3/3f/Placeholder_view_vector.svg",
    "https://upload.wikimedia.org/wikipedia/commons/8/84/Example.svg",
    "https://upload.wikimedia.org/wikipedia/commons/2/25/Icon-round-Question_mark.svg"
]

def generate_garbled_text():
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()_+-=[]{}|;':\",./<>?~`~# "
    length = random.randint(20, 50)
    return "".join(random.choice(alphabet) for _ in range(length)) + "\n"

def get_safe_insertion_points(doc):
    """Scan document structure to find valid text element indices to avoid API crashes."""
    indices = [1] # Always safe to insert near the top
    body_content = doc.get('body', {}).get('content', [])
    for element in body_content:
        if 'paragraph' in element:
            for elem in element['paragraph'].get('elements', []):
                if 'startIndex' in elem:
                    indices.append(elem['startIndex'])
    return indices

def main():
    if not SERVICE_ACCOUNT_JSON or not DOCUMENT_ID:
        raise ValueError("Missing GOOGLE_SERVICE_ACCOUNT_KEY or GOOGLE_DOC_ID environment variables.")

    creds_info = service_account.Credentials.from_service_account_info(
        eval(SERVICE_ACCOUNT_JSON),
        scopes=["https://www.googleapis.com/auth/documents"]
    )
    service = build('docs', 'v1', credentials=creds_info)

    # Fetch document content to find safe indices
    doc = service.documents().get(documentId=DOCUMENT_ID).execute()
    safe_indices = get_safe_insertion_points(doc)
    
    insert_index = random.choice(safe_indices)
    garbled_text = f"\n[GLITCH_BOT @ {insert_index}]: {generate_garbled_text()}\n"
    random_image_url = random.choice(SAMPLE_IMAGES)

    requests = [
        {
            'insertText': {
                'location': {
                    'index': insert_index,
                },
                'text': garbled_text
            }
        },
        {
            'updateTextStyle': {
                'range': {
                    'startIndex': insert_index,
                    'endIndex': insert_index + len(garbled_text)
                },
                'textStyle': {
                    'foregroundColor': {
                        'color': {
                            'rgbColor': {
                                'red': 1.0,
                                'green': 1.0,
                                'blue': 1.0
                            }
                        }
                    }
                },
                'fields': 'foregroundColor'
            }
        },
        {
            'insertInlineImage': {
                'uri': random_image_url,
                'location': {
                    'index': insert_index
                },
                'objectSize': {
                    'height': {'magnitude': 80, 'unit': 'PT'},
                    'width': {'magnitude': 80, 'unit': 'PT'}
                }
            }
        }
    ]

    result = service.documents().batchUpdate(
        documentId=DOCUMENT_ID, body={'requests': requests}
    ).execute()
    print(f"Successfully injected white text and image at safe index {insert_index}! Reply: {result}")

if __name__ == '__main__':
    main()

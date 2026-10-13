import os
import random
import string
from google.oauth2 import service_account
from googleapiclient.discovery import build

SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY")
DOCUMENT_ID = os.environ.get("GOOGLE_DOC_ID")

# A pool of sample public images for random insertion
SAMPLE_IMAGES = [
    "https://upload.wikimedia.org/wikipedia/commons/3/3f/Placeholder_view_vector.svg",
    "https://upload.wikimedia.org/wikipedia/commons/8/84/Example.svg",
    "https://upload.wikimedia.org/wikipedia/commons/2/25/Icon-round-Question_mark.svg"
]

def generate_garbled_text():
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()_+-=[]{}|;':\",./<>?~`~# "
    length = random.randint(20, 50)
    return "".join(random.choice(alphabet) for _ in range(length)) + "\n"

def get_document_length(service, document_id):
    doc = service.documents().get(documentId=document_id).execute()
    # Find the end index of the document body contents
    body_content = doc.get('body', {}).get('content', [])
    if body_content:
        return body_content[-1].get('endIndex', 1) - 1
    return 1

def main():
    if not SERVICE_ACCOUNT_JSON or not DOCUMENT_ID:
        raise ValueError("Missing GOOGLE_SERVICE_ACCOUNT_KEY or GOOGLE_DOC_ID environment variables.")

    creds_info = service_account.Credentials.from_service_account_info(
        eval(SERVICE_ACCOUNT_JSON),
        scopes=["https://www.googleapis.com/auth/documents"]
    )
    service = build('docs', 'v1', credentials=creds_info)

    # Fetch max length to pick a random insert index inside the document
    max_len = get_document_length(service, DOCUMENT_ID)
    insert_index = random.randint(1, max(1, max_len - 1))

    garbled_text = f"\n[GLITCH_BOT @ {insert_index}]: {generate_garbled_text()}"
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
                    'index': insert_index + len(garbled_text)
                },
                'objectSize': {
                    'height': {'magnitude': 100, 'unit': 'PT'},
                    'width': {'magnitude': 100, 'unit': 'PT'}
                }
            }
        }
    ]

    result = service.documents().batchUpdate(
        documentId=DOCUMENT_ID, body={'requests': requests}
    ).execute()
    print(f"Successfully injected text and image at index {insert_index}! Reply: {result}")

if __name__ == '__main__':
    main()

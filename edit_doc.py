import os
import random
import string
from google.oauth2 import service_account
from googleapiclient.discovery import build

SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY")
DOCUMENT_ID = os.environ.get("GOOGLE_DOC_ID")

def generate_garbled_text():
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

this is what we have so fare can we make it so when it runs it randomly picks a place anywhere in the document to do the string thing , and that it also adds a random image to a random slide

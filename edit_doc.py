import os
import random
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Load credentials from the environment variable (JSON string stored as a secret)
SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY")
DOCUMENT_ID = os.environ.get("GOOGLE_DOC_ID")

def main():
    if not SERVICE_ACCOUNT_JSON or not DOCUMENT_ID:
        raise ValueError("Missing GOOGLE_SERVICE_ACCOUNT_KEY or GOOGLE_DOC_ID environment variables.")

    # Authenticate with Google
    creds_info = service_account.Credentials.from_service_account_info(
        eval(SERVICE_ACCOUNT_JSON),
        scopes=["https://www.googleapis.com/auth/documents"]
    )
    service = build('docs', 'v1', credentials=creds_info)

    # Some fun random text options to "randomly do stuff"
    random_phrases = [
        "\n[Bot]: Automated check-in successful. All systems nominal.",
        "\n[Bot]: Random thought of the day: Code compiles, nobody knows why.",
        "\n[Bot]: Ping received from the GitHub cloud execution runner.",
        "\n[Bot]: Status update: Document successfully modified by automated script."
    ]
    chosen_text = random.choice(random_phrases)

    # Prepare the API request to insert text at the end of the document
    # (Note: Google Docs requires insertion indexes; checking document length or appending safely 
    # usually involves reading the document structure first, but here we'll target the very end)
    doc = service.documents().get(documentId=DOCUMENT_ID).execute()
    document_content = doc.get('body').get('content')
    # Get the index just before the final newline of the document body
    end_index = document_content[-1]['endIndex'] - 1

    requests = [
        {
            'insertText': {
                'location': {
                    'index': end_index,
                },
                'text': chosen_text
            }
        }
    ]

    # Execute the request
    result = service.documents().batchUpdate(
        documentId=DOCUMENT_ID, body={'requests': requests}
    ).execute()
    print(f"Successfully updated document! Reply: {result}")

if __name__ == '__main__':
    main()

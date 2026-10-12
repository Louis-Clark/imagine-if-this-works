import os
import random
from google.oauth2 import service_account
from googleapiclient.discovery import build

SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY")
DOCUMENT_ID = os.environ.get("GOOGLE_DOC_ID")

def main():
    if not SERVICE_ACCOUNT_JSON or not DOCUMENT_ID:
        raise ValueError("Missing GOOGLE_SERVICE_ACCOUNT_KEY or GOOGLE_DOC_ID environment variables.")

    creds_info = service_account.Credentials.from_service_account_info(
        eval(SERVICE_ACCOUNT_JSON),
        scopes=["https://www.googleapis.com/auth/documents"]
    )
    service = build('docs', 'v1', credentials=creds_info)

    random_phrases = [
        "[Bot]: Automated check-in successful. All systems nominal.\n",
        "[Bot]: Random thought of the day: Code compiles, nobody knows why.\n",
        "[Bot]: Ping received from the GitHub cloud execution runner.\n",
        "[Bot]: Status update: Document successfully modified by automated script.\n"
    ]
    chosen_text = random.choice(random_phrases)

    # Insert right at the beginning of the document (index 1)
    requests = [
        {
            'insertText': {
                'location': {
                    'index': 1,
                },
                'text': chosen_text
            }
        }
    ]

    result = service.documents().batchUpdate(
        documentId=DOCUMENT_ID, body={'requests': requests}
    ).execute()
    print(f"Successfully updated document at the top! Reply: {result}")

if __name__ == '__main__':
    main()

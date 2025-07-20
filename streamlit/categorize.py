import requests
import os
import dotenv

dotenv.load_dotenv("../config.env")


def extract_response_text(api_response: dict) -> str:
    """
    Extract text from LangFlow API response
    """
    try:
        return api_response["outputs"][0]["outputs"][0]["results"]["message"][
            "text"
        ].strip()
    except (KeyError, IndexError, TypeError):
        try:
            return api_response["outputs"][0]["outputs"][0]["outputs"]["message"][
                "message"
            ].strip()
        except (KeyError, IndexError, TypeError):

            def find_text(data):
                if isinstance(data, dict):
                    for k, v in data.items():
                        if k == "text" and isinstance(v, str):
                            return v
                        result = find_text(v)
                        if result:
                            return result
                elif isinstance(data, list):
                    for item in data:
                        result = find_text(item)
                        if result:
                            return result
                return None

            text = find_text(api_response)
            return text.strip() if text else "Answer not found"


def categorize(question: str) -> str | None:
    """
    Determines the category of the question and returns it as str
    """
    try:
        api_key = os.environ["LANGFLOW_API_KEY"]
        flow = os.environ["CATEGORY_FLOW"]
    except KeyError:
        raise ValueError(
            "LANGFLOW_API_KEY environment variable not found. Please set your API key in the environment variables."
        )

    url = f"http://localhost:7860/api/v1/run/{flow}"  # The complete API endpoint URL for this flow

    # Request payload configuration
    payload = {"output_type": "chat", "input_type": "chat", "input_value": question}

    # Request headers
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,  # Authentication key from environment variable
    }

    try:
        # Send API request
        response = requests.request("POST", url, json=payload, headers=headers)
        response.raise_for_status()  # Raise exception for bad status codes

        # Print response
        return extract_response_text(response.json())

    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
    except ValueError as e:
        print(f"Error parsing response: {e}")

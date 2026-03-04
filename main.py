# import os module to access environment variables
import os

from dotenv import load_dotenv

load_dotenv()


def main():
    print("Hello from langchain-1!")
    # print the OPENAI_API_KEY environment variable
    # print(os.environ.get("OPENAI_API_KEY"))


if __name__ == "__main__":
    main()

# Hotel Booking Chatbot

## Overview

This project is a chatbot application built using Cohere AI. The chatbot assists users in making hotel bookings by providing information and answering queries related to hotel reservations.
Hotel booking bot for Sheraton hotel. Made with cohere AI

<img src="https://github.com/4N1Z/hotel-booking-bot/assets/91843271/31cb54e7-9db1-4251-a93b-693726e73e48" alt="Screenshot 2023-09-24 115029" width="50%">

## Check [blog](https://aniz.hashnode.dev/chatbot-with-rag-memory-cohere-ai-streamlit-langchain-qdrant) to know more about the project
## Features

- **Chat Interface**: Users can interact with the chatbot through a user-friendly chat interface.

- **Hotel Information**: The chatbot can provide details about available hotels, room types, and amenities.

- **Booking Assistance**: Users can inquire about room availability, rates, and make reservations through the chatbot.
  
- **DISCLAIMER**:
- - We are using a short knowledge base and an trail api key for cohere api. Therefore the responses may vary

## Deployment

The chatbot is deployed on Streamlit Cloud and can be accessed at the following URL: [Chatbot App](https://streamlit.io/sharing)

## Prerequisites

Before running this project locally, ensure that you have the following installed:

- Python 3.10.5
- Streamlit
- Other necessary Python libraries (specified in `requirements.txt`)

## Getting Started
For clear instruction refer to the blog

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/4N1Z/hotel-booking-bot
   cd hotel-booking-bot
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```
3. Add your Cohere API key and Qdrant API_key in secrets.toml 

   ```bash
   COHERE_API_KEY = "<add_your_api_key>"
   QDRANT_HOST = "<QDRANT_URL>"
   QDRANT_API_KEY = "QDRANT_API_KEY" 
     
   ```
  4. Run the database file first
     ```bash
     python run dbCheck.py

     ```
   
3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

4. Access the chatbot in your web browser at `http://localhost:8501`.

## Usage

- Start a conversation with the chatbot by typing your questions or commands into the chat input field.

- The chatbot will respond with relevant information and assistance related to hotel bookings.

## Contributing

Contributions are welcome! If you have any ideas, bug fixes, or improvements to the chatbot, please feel free to submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE.md).

## Acknowledgments

- Special thanks to the Langchain, Qdrant, Cohere and Streamlit team for providing a user-friendly framework for building web applications with Python.




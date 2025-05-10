# YouTube Engagement Chatbot

## Introduction

The YouTube Engagement Chatbot is a user-friendly application designed to enhance interaction with YouTube content. It accepts user inputs such as likes or questions and provides automated responses, including generating question-answer pairs, replying to questions, or summarizing prompts. The chatbot features an intuitive UI, making it accessible and engaging for users to interact with YouTube-related content seamlessly.

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Node.js 14.x or higher (for the frontend)
- Git
- A YouTube Data API key (optional, for advanced features)

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/youtube-engagement-chatbot.git
   cd youtube-engagement-chatbot
   ```

2. **Install** Requirements

   - Create a virtual environment

     ```bash
     python -m venv venv
     ```
   - Activate the virtual environment

     ```bash
     venv\Scripts\activate
     ```
   - Install Python dependencies:

     ```bash
     pip install -r requirements.txt
     ```

### Running the Application

1. **Run the endpoints**

   - Run the `endpoints.ipynb` file, each cell in a different Google account, and change the ngrok link in the server.py file.

2. **Run Server**

   - Run the `server.py` file:

     ```bash
     uvicorn server:app --reload
     ```
   - The frontend will run on `http://localhost:8000`.

3. **Access the Chatbot**

   - Open your browser and navigate to `http://localhost:8000` to interact with the chatbot.

## Usage

- **Interacting with the Chatbot**:

  - **Ask Questions**: Type a question related to a YouTube video or topic, and the chatbot will provide a relevant answer.
  - **Generate Q&A**: Input a prompt, and the chatbot will create question-answer pairs.
  - **Summarize Prompts**: Provide a lengthy prompt, and the chatbot will summarize it concisely.
  - **Like Content**: Use the like button to express appreciation for content, and the chatbot will acknowledge it.

## Contributing

We welcome contributions to improve Our Chatbot! To contribute, follow these steps:

1. **Fork the Repository**

   - Click the "Fork" button on the GitHub repository page.

2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add your feature description"
   ```

4. **Push to Your Fork**

   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request**

   - Go to the original repository and create a pull request with a detailed description of your changes.

### Contribution Guidelines

- Ensure your code follows PEP 8 (Python) and ESLint (JavaScript) standards.
- Write clear commit messages and provide thorough documentation for new features.
- Test your changes locally before submitting a pull request.
- Respect the project’s code of conduct (see `CODE_OF_CONDUCT.md`).

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
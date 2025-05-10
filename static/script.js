let selectedAction = 'summarize'; // Default action
let loadingInterval = null;
let contextForAnswer = ''; // Store context for "answer" action

// Function to set the selected action based on the button clicked
function setAction(action) {
    selectedAction = action;
    updateActiveButton();
    // Reset input visibility
    document.getElementById('main-input').style.display = 'flex';
    document.getElementById('question-input').style.display = 'none';
}

// Function to update the active button style
function updateActiveButton() {
    const buttons = document.querySelectorAll('.option-btn');
    buttons.forEach(button => {
        if (button.id === selectedAction) {
            button.classList.add('active');
        } else {
            button.classList.remove('active');
        }
    });
}

function sendMessage() {
    const input = document.getElementById("user-input");
    const chatBox = document.getElementById("chat-box");
    const text = input.value.trim();

    if (text === "") return;

    appendMessage("user", text);
    input.value = "";

    if (selectedAction === 'answer') {
        // Store context and prompt for question
        contextForAnswer = text;
        appendMessage("bot", "Please enter your question.");
        // Hide main input and show question input
        document.getElementById('main-input').style.display = 'none';
        document.getElementById('question-input').style.display = 'flex';
    } else {
        // Handle other actions (summarize, qna)
        const waitingMsg = appendMessage("bot", `waiting for ${selectedAction} <span id="dots">.</span>`);
        startLoadingDots();

        fetch('/process', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: new URLSearchParams({
                'user_input': text,
                'action': selectedAction
            })
        })
        .then(response => response.text())
        .then(data => {
            stopLoadingDots();
            waitingMsg.innerHTML = parseResponse(data); // Parse and display response
        })
        .catch(error => {
            console.error('Error:', error);
            stopLoadingDots();
            waitingMsg.innerHTML = "Sorry, there was an error processing your request.";
        });
    }
}

function sendQuestion() {
    const questionInput = document.getElementById("question-text");
    const question = questionInput.value.trim();

    if (question === "") return;

    appendMessage("user", question);
    questionInput.value = "";

    // Hide question input and show main input
    document.getElementById('main-input').style.display = 'flex';
    document.getElementById('question-input').style.display = 'none';

    // Send context and question to server
    const waitingMsg = appendMessage("bot", `waiting for answer <span id="dots">.</span>`);
    startLoadingDots();

    fetch('/process', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
            'user_input': contextForAnswer,
            'question': question,
            'action': selectedAction
        })
    })
    .then(response => response.text())
    .then(data => {
        stopLoadingDots();
        waitingMsg.innerHTML = parseResponse(data); // Parse and display response
    })
    .catch(error => {
        console.error('Error:', error);
        stopLoadingDots();
        waitingMsg.innerHTML = "Sorry, there was an error processing your request.";
    });
}

// Parse HTML response to extract output
function parseResponse(html) {
    const parser = new DOMParser();
    const doc = parser.parseFromString(html, 'text/html');
    const outputElement = doc.querySelector('#output');
    return outputElement ? outputElement.textContent : "Error: No output found in response.";
}

function appendMessage(sender, text) {
    const message = document.createElement("div");
    message.className = `chat-message ${sender}`; // Fixed syntax
    message.innerHTML = text; // Allow <span> dots
    const chatBox = document.getElementById("chat-box");
    chatBox.appendChild(message);
    chatBox.scrollTop = chatBox.scrollHeight;
    return message; // Return the message div
}

function startLoadingDots() {
    stopLoadingDots(); // Clear any existing interval
    let dots = document.getElementById('dots');
    let count = 1;
    loadingInterval = setInterval(() => {
        count = (count % 3) + 1;
        dots.textContent = '.'.repeat(count);
    }, 500);
}

function stopLoadingDots() {
    if (loadingInterval) {
        clearInterval(loadingInterval);
        loadingInterval = null;
    }
}

// Update active button when the page loads
window.onload = updateActiveButton;
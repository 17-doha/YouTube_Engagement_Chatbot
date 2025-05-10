let selectedAction = 'summarize'; // Default action

// Function to set the selected action based on the button clicked
function setAction(action) {
    selectedAction = action;
    updateActiveButton();
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

    // Send user input to the FastAPI server using a POST request
    fetch('/process', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams({
            'user_input': text,
            'action': selectedAction // Pass the selected action
        })
    })
    .then(response => response.text())
    .then(data => {
        // Append the actual response from the server
        appendMessage("bot", data);
    })
    .catch(error => {
        console.error('Error:', error);
        appendMessage("bot", "Sorry, there was an error processing your request.");
    });
}

function appendMessage(sender, text) {
    const message = document.createElement("div");
    message.className = `chat-message ${sender}`;
    message.innerText = text;

    const chatBox = document.getElementById("chat-box");
    chatBox.appendChild(message);
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Update the active button style when the page loads
window.onload = updateActiveButton;

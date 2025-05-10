let selectedAction = 'summarize'; // Default action
let loadingInterval = null;

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

    // Append waiting message with dots
    const waitingMsg = appendMessage("bot", `Waiting for ${selectedAction}<span id="dots">.</span>`);

    // Start loading dots animation
    startLoadingDots();

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
        stopLoadingDots();
        waitingMsg.innerHTML = data; // Replace waiting message with response
    })
    .catch(error => {
        console.error('Error:', error);
        stopLoadingDots();
        waitingMsg.innerHTML = "Sorry, there was an error processing your request.";
    });
}

function appendMessage(sender, text) {
    const message = document.createElement("div");
    message.className = `chat-message ${sender}`;
    message.innerHTML = text;  // Changed to innerHTML to allow <span> dots
    const chatBox = document.getElementById("chat-box");
    chatBox.appendChild(message);
    chatBox.scrollTop = chatBox.scrollHeight;
    return message; // Return the message div (so we can later update it)
}

function startLoadingDots() {
    let dots = document.getElementById('dots');
    let count = 1;
    loadingInterval = setInterval(() => {
        count = (count % 3) + 1;
        dots.textContent = '.'.repeat(count);
    }, 500);
}

function stopLoadingDots() {
    clearInterval(loadingInterval);
}

// Update the active button style when the page loads
window.onload = updateActiveButton;

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('qa-form');
    const submitBtn = document.getElementById('submit-btn');
    const resultContainer = document.getElementById('result-container');
    const answerOutput = document.getElementById('answer-output');

    resultContainer.style.display = 'none';

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const context = document.getElementById('context').value.trim();
        const question = document.getElementById('question').value.trim();

        if (!context || !question) {
            alert('Please enter both context and question.');
            return;
        }

        submitBtn.disabled = true;
        submitBtn.textContent = 'Processing...';
        resultContainer.style.display = 'none';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ context, question })
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            
            answerOutput.textContent = data.answer || "No answer found in the context.";
            resultContainer.style.display = 'block';

        } catch (error) {
            console.error('There was a problem with the fetch operation:', error);
            answerOutput.textContent = 'An error occurred while fetching the answer.';
            resultContainer.style.display = 'block';
        } finally {
            submitBtn.disabled = false;
            submitBtn.textContent = 'Get Answer';
        }
    });
});
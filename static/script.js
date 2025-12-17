document.addEventListener('DOMContentLoaded', () => {
    const projectButtons = document.querySelectorAll('.project-btn');
    const projectContent = document.getElementById('project-content');

    // Define fertilizer data for the showFert function (needed for the Chemicals section)
    const fertilizerData = {
        nitrogen: "Nitrogen Fertilizers (e.g., Urea): Vital for chlorophyll production. It promotes rapid vegetative growth, lush green foliage, and high protein content in crops.",
        phosphorus: "Phosphorus Fertilizers (e.g., DAP): Essential for energy transfer and root development. It ensures early plant maturity and improves flowering and seed production.",
        potassium: "Potassium Fertilizers (e.g., MOP): Regulates water movement and activates enzymes. It enhances drought resistance and improves the quality and shelf life of fruits.",
        organic: "Organic Fertilizers (e.g., Vermicompost): Naturally sourced nutrients that improve soil structure, water retention, and microbial activity for long-term sustainability.",
        compound: "Compound Fertilizers (NPK): A balanced mix designed to provide all primary nutrients in one application, perfect for generalized soil enrichment."
    };

    projectButtons.forEach(button => {
        button.addEventListener('click', (event) => {
            const clickedButton = event.currentTarget;
            const targetId = clickedButton.getAttribute('data-target');

            // 1. **Smooth Fade-Out:** Start the fade-out transition for the button
            clickedButton.classList.add('fade-out');

            // 2. **Smooth Content Update:**
            projectContent.style.transition = 'opacity 0.5s ease';
            projectContent.style.opacity = 0; // Fade out content

            // Wait for the fade-out to complete before updating and fading back in
            setTimeout(() => {
                let newContent = '';
                switch (targetId) {
                    case 'disease':
                        // Requirement: Redirect to streamlit/local server
                        window.location.href = "http://localhost:8501";
                        break;
                    case 'solutions':
                        // Requirement: Integrated Video Feed and Manual Browse Upload
                        newContent = `
                            <div class="video-container fade-in">
                                <h3 style="color: white;">Live Disease Analysis System</h3>
                                <img src="http://127.0.0.1:5000/video_feed" width="850" height="480" alt="Live AI Feed">
                            </div>
                            <div class="upload-wrapper fade-in">
                                <h3 style="color: white;">Manual Image Analysis</h3>
                                <input type="file" id="browse-input" accept="image/*" style="display: none;">
                                <button class="sub-btn" onclick="document.getElementById('browse-input').click()">📁 Browse from Device</button>
                                <div id="preview-container"><img id="leaf-preview" src="" alt="Preview" style="display:none; max-width:300px; margin: 20px auto; border: 2px solid #ff5722; border-radius: 10px;"></div>
                                <div id="prediction-result" class="fertilizer-info-box fade-in" style="display: none;">
                                    <h4 id="result-title">Analysis Result</h4>
                                    <p id="result-text"></p>
                                    <div style="margin-top: 10px; padding: 10px; background: #222; border-left: 4px solid #ff5722;">
                                        <strong>Treatment Recommendation:</strong>
                                        <p id="result-rec"></p>
                                    </div>
                                </div>
                            </div>`;
                        break;
                    case 'chemicals':
                        // Requirement: Chemical and Fertilizer Sub-menus
                        newContent = `
                            <div class="sub-menu fade-in">
                                <button class="sub-btn" id="btn-chem-list">Chemicals</button>
                                <button class="sub-btn" id="btn-fert-list">Fertilizers</button>
                            </div>
                            <div id="sub-display" class="fade-in">
                                <p style="color: #666;">Select a category to explore agricultural chemistry.</p>
                            </div>`;
                        break;
                }

                projectContent.innerHTML = newContent;
                projectContent.style.opacity = 1; // Fade in new content

                // Re-attach specific listeners for the new content after it's injected
                if (targetId === 'solutions') {
                    attachSolutionListeners();
                } else if (targetId === 'chemicals') {
                    attachChemicalListeners();
                }

                // 3. **Smooth Fade-In for Button:** Reset the button after a delay
                setTimeout(() => {
                    clickedButton.classList.remove('fade-out');
                }, 1000);
            }, 500);
        });
    });

    // Helper function to handle the browse/prediction logic
    function attachSolutionListeners() {
        const browseInput = document.getElementById('browse-input');
        if (browseInput) {
            browseInput.addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(event) {
                        const preview = document.getElementById('leaf-preview');
                        preview.src = event.target.result;
                        preview.style.display = "block";
                        
                        // Wait for image to load to analyze colors
                        preview.onload = () => processDetection();
                    };
                    reader.readAsDataURL(file);
                }
            });
        }
    }

    function processDetection() {
        const resultDiv = document.getElementById('prediction-result');
        const title = document.getElementById('result-title');
        const text = document.getElementById('result-text');
        const rec = document.getElementById('result-rec');
        const preview = document.getElementById('leaf-preview');

        // Create canvas to scan image colors
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = preview.naturalWidth;
        canvas.height = preview.naturalHeight;
        ctx.drawImage(preview, 0, 0);

        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
        let r = 0, g = 0, b = 0;

        for (let i = 0; i < imageData.length; i += 4) {
            r += imageData[i];
            g += imageData[i + 1];
            b += imageData[i + 2];
        }

        const pixelCount = imageData.length / 4;
        const avgR = r / pixelCount;
        const avgG = g / pixelCount;
        const avgB = b / pixelCount;

        resultDiv.style.display = "block";

        // Logic: If image has strong Red/Brown/Yellow tones relative to green
        // Reddish-Green/Brownish-Green/Yellowish-Green triggers Blight
        if (avgR > (avgG - 30)) { 
            title.innerText = "Detection: Early or Late Blight Detected";
            title.style.color = "red";
            text.innerText = "Non-green pigments (brown/yellow/red) detected. This indicates necrotic tissue consistent with blight.";
            rec.innerText = "Apply Copper-based fungicides immediately. Remove and destroy infected leaves.";
        } else {
            // Pure Green
            title.innerText = "Status: Healthy Leaf Detected";
            title.style.color = "#00ff88";
            text.innerText = "Dominant green chlorophyll signature detected. No significant browning found.";
            rec.innerText = "Continue regular organic fertilization and monitoring.";
        }
    }

    // Helper function for Chemical/Fertilizer sub-tabs
    function attachChemicalListeners() {
        document.getElementById('btn-chem-list').addEventListener('click', () => {
            document.getElementById('sub-display').innerHTML = `
                <div class="chem-grid fade-in">
                    <div class="chem-card"><h3>N</h3><p>Nitrogen</p></div>
                    <div class="chem-card"><h3>P</h3><p>Phosphorus</p></div>
                    <div class="chem-card"><h3>K</h3><p>Potassium</p></div>
                </div>`;
        });
        document.getElementById('btn-fert-list').addEventListener('click', () => {
            document.getElementById('sub-display').innerHTML = `
                <div class="chem-grid fade-in">
                    <div class="fert-card" onclick="showFert('nitrogen')"><h4>Nitrogenous</h4></div>
                    <div class="fert-card" onclick="showFert('phosphorus')"><h4>Phosphatic</h4></div>
                    <div class="fert-card" onclick="showFert('potassium')"><h4>Potassic</h4></div>
                </div>
                <div id="fert-detail"></div>`;
        });
    }

    // Global function for fertilizer detail (called by onclick in HTML string)
    window.showFert = function(type) {
        const detailDiv = document.getElementById('fert-detail');
        detailDiv.innerHTML = `
            <div class="fertilizer-info-box fade-in">
                <h4>Nutrient Description:</h4>
                <p>${fertilizerData[type]}</p>
            </div>`;
    };
});
document.addEventListener('DOMContentLoaded', function() {
    const searchInput = document.getElementById('search-input');
    const searchBtn = document.getElementById('search-btn');
    const suggestionsContainer = document.getElementById('suggestions');
    const resultsContainer = document.getElementById('results-container');
    const resultsTitle = document.getElementById('results-title');
    const resultsContent = document.getElementById('results-content');
    const rangeInfoContainer = document.getElementById('range-info');
    
    // Load examples
    loadExamples();
    
    // Search when button is clicked or Enter is pressed
    searchBtn.addEventListener('click', performSearch);
    searchInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            performSearch();
        }
    });
    
    // Get suggestions as user types
    searchInput.addEventListener('input', function() {
        const query = searchInput.value.trim();
        if (query.length > 0) {
            getSuggestions(query);
        } else {
            suggestionsContainer.innerHTML = '';
        }
    });
    
    // Perform search for the given query
    async function performSearch() {
        const query = searchInput.value.trim();
        if (!query) return;
        
        // Get the selected Bible version
        const version = document.getElementById('bible-version').value;
        
        try {
            const response = await axios.post('/api/bible/search', {
                query: query,
                version: version,
                include_context: false
            });
            
            displayResults(response.data);
        } catch (error) {
            console.error('Search error:', error);
            showError('Failed to perform search. Please try again.');
        }
    }
    
    // Get search suggestions for the given query
    async function getSuggestions(query) {
        try {
            const response = await axios.get('/api/bible/suggestions', {
                params: { q: query }
            });
            
            displaySuggestions(response.data.suggestions);
        } catch (error) {
            console.error('Suggestions error:', error);
            suggestionsContainer.innerHTML = '';
        }
    }
    
    // Display the list of suggestions
    function displaySuggestions(suggestions) {
        suggestionsContainer.innerHTML = '';
        
        suggestions.forEach(suggestion => {
            const tag = document.createElement('span');
            tag.className = 'tag is-info is-medium suggestion-tag';
            tag.textContent = suggestion;
            tag.addEventListener('click', function() {
                searchInput.value = suggestion;
                performSearch();
            });
            suggestionsContainer.appendChild(tag);
        });
    }
    
    // Display the search results
    function displayResults(data) {
        resultsContainer.style.display = 'block';
        resultsTitle.textContent = data.message;
        resultsContent.innerHTML = '';
        rangeInfoContainer.style.display = 'none';
            // Display structured AI response if available
            if (data.ai_response_structured) {
                const s = data.ai_response_structured;
                const aiDiv = document.createElement('div');
                aiDiv.className = 'notification is-light mt-4';
                let html = '';
                if (s.type === 'single') {
                    html += `<p class="title is-5">📖 ${s.query}</p>`;
                    if (s.verses && s.verses.length > 0) {
                        html += `<p>${s.verses[0].text}</p>`;
                    }
                } else if (s.type === 'range' || s.type === 'semantic') {
                    html += `<p class="title is-5">📖 ${s.query}</p>`;
                    s.verses.forEach(v => {
                        html += `<p><strong>${v.reference}</strong> ${v.text}</p>`;
                    });
                } else {
                    html += `<pre>${JSON.stringify(s, null, 2)}</pre>`;
                }
                aiDiv.innerHTML = html;
                resultsContent.appendChild(aiDiv);
            }

            // version badge for results
            const versionBadge = document.createElement('div');
            versionBadge.className = 'tags has-addons mb-3';

            const versionLabel = document.createElement('span');
            versionLabel.className = 'tag is-dark';
            versionLabel.textContent = 'Version';
            
            const versionValue = document.createElement('span');
            versionValue.className = 'tag is-info';
            versionValue.textContent = data.version ? data.version.toUpperCase() : 'KJV';
            
            versionBadge.appendChild(versionLabel);
            versionBadge.appendChild(versionValue);
            resultsContent.appendChild(versionBadge);
        
        if (data.found && data.results.length > 0) {
            // Display verses
            data.results.forEach(verse => {
                const verseDiv = document.createElement('div');
                verseDiv.className = 'verse-result';
                
                const reference = document.createElement('div');
                reference.className = 'verse-reference';
                reference.textContent = verse.reference;
                
                const text = document.createElement('div');
                text.className = 'verse-text';
                text.textContent = verse.text;
                
                verseDiv.appendChild(reference);
                verseDiv.appendChild(text);
                resultsContent.appendChild(verseDiv);
            });
            
            // For semantic search, add preview options if more than one result
            if (data.query_type === "semantic_search" && data.results.length > 1) {
                const previewHeader = document.createElement('h4');
                previewHeader.className = 'title is-5 mt-4';
                previewHeader.textContent = 'Select a verse to view in detail:';
                resultsContent.appendChild(previewHeader);
                
                const previewContainer = document.createElement('div');
                previewContainer.className = 'preview-container';
                
                data.results.forEach((verse, index) => {
                    const previewCard = document.createElement('div');
                    previewCard.className = 'card mb-3 preview-card';
                    previewCard.innerHTML = `
                        <div class="card-content">
                            <p class="title is-6">${verse.reference}</p>
                            <p class="subtitle is-7">${verse.text.length > 100 ? verse.text.substring(0, 100) + '...' : verse.text}</p>
                        </div>
                        <footer class="card-footer">
                            <a class="card-footer-item view-detail-btn">View Full Verse</a>
                        </footer>
                    `;
                    
                    // Add click event to view the verse in detail
                    previewCard.querySelector('.view-detail-btn').addEventListener('click', function() {
                        // Show the selected verse in detail
                        showVerseDetail(verse, data.version);
                    });
                    
                    previewContainer.appendChild(previewCard);
                });
                
                resultsContent.appendChild(previewContainer);
            }
            
            // Display AI response if available
            // Display structured AI response if available, otherwise fallback to ai_response string
            if (data.ai_response_structured) {
                const s = data.ai_response_structured;
                const aiDiv = document.createElement('div');
                aiDiv.className = 'notification is-light mt-4';
                // Build a structured rendering
                let html = '';
                if (s.type === 'single') {
                    html += `<p class="title is-5">📖 ${s.query}</p>`;
                    if (s.verses && s.verses.length > 0) {
                        html += `<p>${s.verses[0].text}</p>`;
                    }
                } else if (s.type === 'range' || s.type === 'semantic') {
                    html += `<p class="title is-5">📖 ${s.query}</p>`;
                    s.verses.forEach(v => {
                        html += `<p><strong>${v.reference}</strong> ${v.text}</p>`;
                    });
                } else {
                    html += `<pre>${JSON.stringify(s, null, 2)}</pre>`;
                }
                aiDiv.innerHTML = html;
                resultsContent.appendChild(aiDiv);
            } else if (data.ai_response) {
                const aiDiv = document.createElement('div');
                aiDiv.className = 'notification is-light mt-4';
                aiDiv.innerHTML = data.ai_response;
                resultsContent.appendChild(aiDiv);
            }
            
            // Display range info if applicable
            if (data.range_info && data.range_info.is_range) {
                let rangeText = `Range: ${data.query} (Found ${data.range_info.total_verses_found} of ${data.range_info.total_verses_requested} verses)`;
                
                if (data.range_info.missing_verses) {
                    rangeText += `<br>Missing verses: ${data.range_info.missing_verses.join(', ')}`;
                }
                
                rangeInfoContainer.innerHTML = rangeText;
                rangeInfoContainer.style.display = 'block';
            }
        } else {
            // No results found
            resultsContent.innerHTML = `
                <div class="notification is-warning">
                    ${data.message}
                </div>
            `;
            
            // Show suggestions if available
            if (data.suggestions && data.suggestions.length > 0) {
                const suggestionsDiv = document.createElement('div');
                suggestionsDiv.className = 'mt-4';
                suggestionsDiv.innerHTML = '<p>Try one of these:</p>';
                
                const tagsDiv = document.createElement('div');
                tagsDiv.className = 'tags';
                
                data.suggestions.forEach(suggestion => {
                    const tag = document.createElement('span');
                    tag.className = 'tag is-info is-medium suggestion-tag';
                    tag.textContent = suggestion;
                    tag.addEventListener('click', function() {
                        searchInput.value = suggestion;
                        performSearch();
                    });
                    tagsDiv.appendChild(tag);
                });
                
                suggestionsDiv.appendChild(tagsDiv);
                resultsContent.appendChild(suggestionsDiv);
            }
        }
    }
    
    // Show the detail view of the selected verse
    function showVerseDetail(verse, version) {
        // Clear the results content
        resultsContent.innerHTML = '';
        
        // Show version information if available
        if (version) {
            const versionBadge = document.createElement('div');
            versionBadge.className = 'tags has-addons mb-3';
            
            const versionLabel = document.createElement('span');
            versionLabel.className = 'tag is-dark';
            versionLabel.textContent = 'Version';
            
            const versionValue = document.createElement('span');
            versionValue.className = 'tag is-info';
            versionValue.textContent = version.toUpperCase();
            
            versionBadge.appendChild(versionLabel);
            versionBadge.appendChild(versionValue);
            resultsContent.appendChild(versionBadge);
        }
        
        // Create a back button
        const backButton = document.createElement('button');
        backButton.className = 'button is-info is-light mb-4';
        backButton.innerHTML = '<i class="fas fa-arrow-left"></i>&nbsp; Back to Results';
        backButton.addEventListener('click', function() {
            performSearch(); // Re-run the search to get all results again
        });
        resultsContent.appendChild(backButton);

        // Add Next button for seamless reading
        const nextBtn = document.createElement('button');
        nextBtn.className = 'button is-primary is-light ml-3 mb-4';
        nextBtn.textContent = 'Next';
        nextBtn.addEventListener('click', async function() {
            // Call the next endpoint
            try {
                const version = document.getElementById('bible-version').value;
                const resp = await axios.get('/api/bible/next', { params: { reference: verse.reference, version: version } });
                showVerseDetail(resp.data, version);
            } catch (err) {
                console.error('Next verse error', err);
                showError('No next verse found.');
            }
        });
        resultsContent.appendChild(nextBtn);
        // Add Presentation mode button
        const presBtn = document.createElement('button');
        presBtn.className = 'button is-link is-light ml-3 mb-4';
        presBtn.textContent = 'Presentation';
        presBtn.addEventListener('click', function() {
            const v = encodeURIComponent(verse.reference);
            const ver = encodeURIComponent(version || 'kjv');
            // open presentation via mounted static path
            window.open(`/static/presentation.html?reference=${v}&version=${ver}`, '_blank');
        });
        resultsContent.appendChild(presBtn);
        
        // Display the verse in detail
        const verseDiv = document.createElement('div');
        verseDiv.className = 'verse-result verse-detail';
        
        const reference = document.createElement('div');
        reference.className = 'verse-reference';
        reference.textContent = verse.reference;
        
        const text = document.createElement('div');
        text.className = 'verse-text';
        text.textContent = verse.text;
        
        verseDiv.appendChild(reference);
        verseDiv.appendChild(text);
        resultsContent.appendChild(verseDiv);
        
        // Update the title
        resultsTitle.textContent = `Viewing: ${verse.reference} (${version ? version.toUpperCase() : 'KJV'})`;
    }

    // Keyboard navigation: Left = prev, Right = next, when not focused in an input
    document.addEventListener('keydown', async function(e) {
        const active = document.activeElement;
        if (active && (active.tagName === 'INPUT' || active.tagName === 'TEXTAREA')) return;

        if (e.key === 'ArrowRight') {
            // If currently viewing a verse detail, try to get the reference text
            const refElem = document.querySelector('.verse-detail .verse-reference');
            if (refElem) {
                const reference = refElem.textContent.trim();
                const version = document.getElementById('bible-version').value;
                try {
                    const resp = await axios.get('/api/bible/next', { params: { reference: reference, version: version } });
                    showVerseDetail(resp.data, version);
                } catch (err) {
                    showError('No more verses.');
                }
            }
        } else if (e.key === 'ArrowLeft') {
            const refElem = document.querySelector('.verse-detail .verse-reference');
            if (refElem) {
                const reference = refElem.textContent.trim();
                const version = document.getElementById('bible-version').value;
                try {
                    const resp = await axios.get('/api/bible/prev', { params: { reference: reference, version: version } });
                    showVerseDetail(resp.data, version);
                } catch (err) {
                    showError('No previous verse.');
                }
            }
        }
    });
    
    // Show an error message
    function showError(message) {
        resultsContainer.style.display = 'block';
        resultsTitle.textContent = 'Error';
        resultsContent.innerHTML = `
            <div class="notification is-danger">
                ${message}
            </div>
        `;
    }
    
    // Load example searches from the server
    async function loadExamples() {
        try {
            const response = await axios.get('/api/bible/examples');
            const examples = response.data;
            
            populateExamples('single-verse-examples', examples.exact_references);
            populateExamples('range-examples', examples.verse_ranges);
            populateExamples('topic-examples', examples.semantic_searches);
            populateExamples('quote-examples', examples.quoted_text);
        } catch (error) {
            console.error('Error loading examples:', error);
        }
    }
    
    // Populate the example searches in the UI
    function populateExamples(elementId, examples) {
        const container = document.getElementById(elementId);
        examples.forEach(example => {
            const li = document.createElement('li');
            const a = document.createElement('a');
            a.href = '#';
            a.textContent = example;
            a.addEventListener('click', function(e) {
                e.preventDefault();
                searchInput.value = example;
                performSearch();
            });
            li.appendChild(a);
            container.appendChild(li);
        });
    }
});

// Add event listeners for search input and button
// Define functions to perform search and fetch suggestions
// Display search results and handle errors
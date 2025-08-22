// [app.js] loaded - debug marker
console.log('[app.js] loaded', new Date().toISOString());

const api = axios.create({ baseURL: '/api' });

let currentReference = null;
let lastQuery = null;
// Holds the Bible version detected by a semantic search (set by /api/bible/semantic response)
let semanticDetectedVersion = null;

// UI: show detected version badge with Accept/Change
function showDetectedVersionBadge(version) {
  try {
    let badge = document.getElementById('detected-version-badge');
    const container = document.getElementById('controls') || document.body;
    if (!badge) {
      badge = document.createElement('div');
      badge.id = 'detected-version-badge';
      badge.style.position = 'absolute';
      badge.style.right = '1rem';
      badge.style.top = '1rem';
      badge.style.background = '#fff8dc';
      badge.style.border = '1px solid #e0c070';
      badge.style.padding = '0.4rem 0.6rem';
      badge.style.borderRadius = '6px';
      badge.style.boxShadow = '0 2px 6px rgba(0,0,0,0.08)';
      badge.style.zIndex = 1200;
      badge.style.fontSize = '0.95rem';
      container.appendChild(badge);
    }

    badge.innerHTML = '';

    const txt = document.createElement('span');
    txt.textContent = `Detected version: ${version}`;
    txt.style.marginRight = '0.6rem';
    badge.appendChild(txt);

    const accept = document.createElement('button');
    accept.textContent = 'Accept';
    accept.style.marginRight = '0.4rem';
    accept.addEventListener('click', () => {
      try { document.getElementById('bible-version').value = version; } catch (e) {}
      badge.remove();
    });
    badge.appendChild(accept);

    const change = document.createElement('button');
    change.textContent = 'Change';
    change.addEventListener('click', () => {
      try { document.getElementById('bible-version').focus(); } catch (e) {}
    });
    badge.appendChild(change);
  } catch (e) {
    console.warn('[detected-badge] error', e);
  }
}

function updateSyncStatus(ok, msg) {
  const el = document.getElementById('syncStatus');
  if (!el) return;
  el.textContent = msg || (ok ? 'online' : 'offline');
  el.classList.toggle('online', ok);
  el.classList.toggle('offline', !ok);
}

function clearVerseContainer() {
  const verseEl = document.getElementById('verse');
  verseEl.innerHTML = '';
}

function renderPagedVerses(verses, perPage) {
  const verseEl = document.getElementById('verse');
  clearVerseContainer();

  const total = verses.length;
  const pages = Math.max(1, Math.ceil(total / perPage));

  for (let p = 0; p < pages; p++) {
    const pageDiv = document.createElement('div');
    pageDiv.className = 'verse-page';
    pageDiv.dataset.page = p + 1;

    const start = p * perPage;
    const end = Math.min(start + perPage, total);

    const fragment = document.createDocumentFragment();
    for (let i = start; i < end; i++) {
      const v = verses[i];
      const pEl = document.createElement('p');
      pEl.style.margin = '0.6rem 0';
      pEl.innerHTML = `<strong>${v.verse}.</strong> ${v.text}`;
      fragment.appendChild(pEl);
    }
    pageDiv.appendChild(fragment);
    verseEl.appendChild(pageDiv);
  }

  // add pagination controls
  const controls = document.createElement('div');
  controls.className = 'verse-pagination-controls';
  // leave extra bottom margin so these controls don't sit behind the fixed bottom bar
  controls.style.marginBottom = '2rem';

  const prevBtn = document.createElement('button');
  prevBtn.textContent = 'Prev page';
  const nextBtn = document.createElement('button');
  nextBtn.textContent = 'Next page';
  const pageLabel = document.createElement('span');
  pageLabel.style.opacity = '0.9';

  controls.appendChild(prevBtn);
  controls.appendChild(pageLabel);
  controls.appendChild(nextBtn);
  verseEl.appendChild(controls);

  let current = 1;
  function showPage(n) {
    const pagesEls = verseEl.querySelectorAll('.verse-page');
    pagesEls.forEach(el => el.classList.toggle('active', Number(el.dataset.page) === n));
    pageLabel.textContent = `Page ${n} / ${pages}`;
    current = n;
  }

  prevBtn.addEventListener('click', () => showPage(Math.max(1, current - 1)));
  nextBtn.addEventListener('click', () => showPage(Math.min(pages, current + 1)));

  showPage(1);
}

async function loadRef(ref, version) {
  if (!ref) return;
  
  const searchMode = document.getElementById('search-mode').value;
  
  if (searchMode === 'semantic') {
    await loadSemanticSearch(ref, version);
  } else {
    await loadReferenceSearch(ref, version);
  }
}

// Global variables for smart chunking
let originalPassage = null; // Store the full passage info
let currentChunk = 0; // Current chunk index
let chunkSize = 3; // Default chunk size

async function loadReferenceSearch(ref, version) {
  lastQuery = { query: ref, version };
  try {
    const res = await api.post('/bible/search', lastQuery);
    console.log('[loadReferenceSearch] response', res.data);
    if (!res.data || !res.data.results || res.data.results.length === 0) {
      document.getElementById('reference').textContent = ref;
      document.getElementById('verse').innerHTML = '<em>Not found</em>';
      updateSyncStatus(false, 'not found');
      return;
    }

    // Normalize response: some endpoints return an array of individual verse objects
    // while others return a single result with a .verses array. Handle both.
    const rawResults = res.data.results || [];
    // Build a consistent `verses` array of objects { verse, text, book, chapter }
    let verses = [];
    let refInfo = { book: null, chapter: null };

    if (rawResults.length > 0 && rawResults[0].verse !== undefined) {
      // Each element is a single verse entry
      verses = rawResults.map(r => ({ verse: r.verse, text: r.text, book: r.book || null, chapter: r.chapter || null, reference: r.reference }));
      refInfo.book = rawResults[0].book || (rawResults[0].reference ? rawResults[0].reference.split(' ')[0] : null);
      refInfo.chapter = rawResults[0].chapter || null;
    } else if (res.data.ai_response_structured && res.data.ai_response_structured.verses) {
      // Fallback to structured AI response
      const structured = res.data.ai_response_structured.verses;
      verses = structured.map(v => {
        const m = (v.reference || '').match(/:(\d+)$/);
        return { verse: m ? Number(m[1]) : null, text: v.text, reference: v.reference };
      });
      // Try to parse book/chapter from first reference
      if (verses.length > 0 && verses[0].reference) {
        const refParts = verses[0].reference.split(':')[0].split(' ');
        refInfo.chapter = Number(refParts[refParts.length - 1]) || null;
        refInfo.book = refParts.slice(0, -1).join(' ');
      }
    } else if (rawResults.length > 0 && rawResults[0].verses) {
      // Older shape: single result with a .verses array
      verses = rawResults[0].verses.map(v => ({ verse: v.verse, text: v.text }));
      refInfo.book = rawResults[0].reference?.book || null;
      refInfo.chapter = rawResults[0].reference?.chapter || null;
    }

    // Compute a canonical display reference when verses are available
    let displayRef = null;
    if (verses && verses.length > 0) {
      const book = refInfo.book || (res.data.results && res.data.results[0] && res.data.results[0].book) || null;
      const chapter = refInfo.chapter || (res.data.results && res.data.results[0] && res.data.results[0].chapter) || null;
      if (book && chapter) {
        if (verses.length === 1) {
          displayRef = `${book} ${chapter}:${verses[0].verse}`;
        } else {
          displayRef = `${book} ${chapter}:${verses[0].verse}-${verses[verses.length-1].verse}`;
        }
      }
    }

    // Fallbacks if we couldn't build a canonical ref
    if (!displayRef) {
      displayRef = (res.data.results && res.data.results[0] && (res.data.results[0].reference?.compact || res.data.results[0].reference?.display)) || res.data.ai_response_structured?.query || ref;
    }
    document.getElementById('reference').textContent = displayRef;

    const perPageSelect = document.getElementById('verses-per-page');
    // default/fallback to 4 verses per page to keep presentation lean
    let perPage = 4;
    try {
      const v = perPageSelect ? Number(perPageSelect.value) : NaN;
      if (Number.isInteger(v) && v > 0) perPage = v;
    } catch (e) {}

    if (verses && verses.length > 1) {
      // Check if this is a range that needs smart chunking (any range with more than 3 verses)
      if (verses.length > 3) {
        // Initialize smart chunking for ranges with 4+ verses
        originalPassage = {
          book: refInfo.book || (res.data.results && res.data.results[0] && res.data.results[0].book) || null,
          chapter: refInfo.chapter || (res.data.results && res.data.results[0] && res.data.results[0].chapter) || null,
          verses: verses,
          startVerse: verses[0].verse,
          endVerse: verses[verses.length - 1].verse,
          isRange: true
        };
        currentChunk = 0;

        // Display first chunk
        displayCurrentChunk();
        console.log(`[loadReferenceSearch] Range detected: ${verses.length} verses, using smart chunking`);
      } else {
        // For shorter ranges (2-3 verses), use existing logic
        originalPassage = null;
        const lastVerse = verses[verses.length - 1];
        currentReference = `${refInfo.book || (res.data.results && res.data.results[0] && res.data.results[0].book) || ''} ${refInfo.chapter || (res.data.results && res.data.results[0] && res.data.results[0].chapter) || ''}:${lastVerse.verse}`;

  const verseEl = document.getElementById('verse');
  verseEl.innerHTML = '';
  // Render verses only — the canonical reference is shown in the top `#reference` element
  renderPagedVerses(verses, perPage);
        verseEl.classList.add('multi-verse');

        console.log(`[loadReferenceSearch] Short range: Set currentReference to last verse: ${currentReference}`);
      }
    } else {
      // For single verses, use the original reference
      originalPassage = null;
      // If the API returned a single-verse object, use that. Otherwise fall back to r.reference or input
      if (rawResults.length === 1 && rawResults[0].reference) {
        currentReference = rawResults[0].reference;
        const verseEl = document.getElementById('verse');
        verseEl.classList.remove('multi-verse');
        verseEl.textContent = rawResults[0].text || '';
      } else if (res.data.results && res.data.results[0] && res.data.results[0].text) {
        currentReference = res.data.results[0].reference || ref;
        const verseEl = document.getElementById('verse');
        verseEl.classList.remove('multi-verse');
        verseEl.textContent = res.data.results[0].text || '';
      } else {
        currentReference = ref;
        const verseEl = document.getElementById('verse');
        verseEl.classList.remove('multi-verse');
        verseEl.textContent = '';
      }
      const verseEl = document.getElementById('verse');
      
    }

    updateSyncStatus(true, 'synced');
  } catch (err) {
    console.error('[loadRef] error', err);
    updateSyncStatus(false, 'error');
  }
}

function displayCurrentChunk() {
  if (!originalPassage) return;
  
  const startIdx = currentChunk * chunkSize;
  const endIdx = Math.min(startIdx + chunkSize, originalPassage.verses.length);
  const chunkVerses = originalPassage.verses.slice(startIdx, endIdx);
  
  if (chunkVerses.length === 0) return;
  
  // Set current reference to the last verse in this chunk
  const lastVerseInChunk = chunkVerses[chunkVerses.length - 1];
  currentReference = `${originalPassage.book} ${originalPassage.chapter}:${lastVerseInChunk.verse}`;
  
  // Update display
  const verseEl = document.getElementById('verse');
  verseEl.innerHTML = '';
  verseEl.classList.add('multi-verse');
  
  // Display the chunk verses
  const container = document.createElement('div');
  container.style.marginTop = '1rem';
  
  chunkVerses.forEach((verse, index) => {
    const verseDiv = document.createElement('div');
    verseDiv.style.margin = '0.8rem 0';
    verseDiv.style.lineHeight = '1.6';
    verseDiv.innerHTML = `<strong>${verse.verse}.</strong> ${verse.text}`;
    container.appendChild(verseDiv);
  });
  
  verseEl.appendChild(container);
  
    console.log(`[displayCurrentChunk] Showing chunk ${currentChunk + 1}/${Math.ceil(originalPassage.verses.length / chunkSize)}`);
}

async function loadSemanticSearch(query, version) {
  console.log('[semantic] Starting search for:', query);
  
  // Show loading state
  document.getElementById('reference').textContent = 'Searching...';
  document.getElementById('verse').innerHTML = '<em>Searching for similar verses...</em>';
  
  try {
    // Make direct API call - prefer the provided `version` argument, otherwise use the UI selector; default to 'auto'
    const thresholdControl = document.getElementById('similarity-threshold');
    const thresh = thresholdControl && thresholdControl.value ? Number(thresholdControl.value) : null;
    const thrParam = (thresh !== null && !Number.isNaN(thresh)) ? `&threshold=${encodeURIComponent(thresh)}` : '';

    // Determine which version to send: prefer explicit function arg -> UI selector -> 'auto'
    const uiVersionEl = document.getElementById('bible-version');
    const versionToUse = (typeof version !== 'undefined' && version) ? version : (uiVersionEl ? uiVersionEl.value : 'auto');
    const versionParam = encodeURIComponent(versionToUse || 'auto');

    const response = await fetch(`/api/bible/semantic?query=${encodeURIComponent(query)}&version=${versionParam}&limit=3${thrParam}`, {
      method: 'POST'
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    
    const data = await response.json();
    console.log('[semantic] Response:', data);
    // Capture any detected version so navigation (Next/Prev) uses the proper version
    semanticDetectedVersion = data && data.version ? data.version : null;
    if (semanticDetectedVersion) {
      try { document.getElementById('bible-version').value = semanticDetectedVersion; } catch(e) {}
      // Remember last query with the detected version for refresh/navigation
      lastQuery = { query, version: semanticDetectedVersion };
    }
    
    // Simple check - if we have results, show them
    if (data && data.found && data.results && data.results.length > 0) {
      console.log('[semantic] Displaying', data.results.length, 'results - detected version:', data.version);
      // Capture and show detected version badge
      semanticDetectedVersion = data.version || semanticDetectedVersion;
      if (semanticDetectedVersion) showDetectedVersionBadge(semanticDetectedVersion);
      displaySemanticResults(data.results, query, data.version);
    } else {
      console.log('[semantic] No results found');
      document.getElementById('reference').textContent = 'Semantic Search';
      document.getElementById('verse').innerHTML = '<em>No similar verses found</em>';
    }
    
  } catch (error) {
    console.error('[semantic] Error:', error);
    document.getElementById('reference').textContent = 'Semantic Search';
    document.getElementById('verse').innerHTML = '<em>Search error</em>';
  }
}

function displaySemanticResults(results, query, detectedVersion) {
  document.getElementById('reference').textContent = `Semantic Search: "${query}"`;
  
  const verseEl = document.getElementById('verse');
  verseEl.innerHTML = '';
  verseEl.classList.remove('multi-verse');
  
  const container = document.createElement('div');
  container.className = 'semantic-results';
  
  const instructions = document.createElement('div');
  instructions.style.marginBottom = '1.5rem';
  instructions.style.fontSize = '1.1rem';
  instructions.style.opacity = '0.8';
  instructions.textContent = 'Click on a verse to select and use with Next/Prev navigation:';
  container.appendChild(instructions);
  
  results.forEach((result, index) => {
    const item = document.createElement('div');
    item.className = 'semantic-result-item';
    // Reference row with version badge
    const header = document.createElement('div');
    header.style.display = 'flex';
    header.style.justifyContent = 'space-between';
    header.style.alignItems = 'center';

    const ref = document.createElement('div');
    ref.className = 'reference';
    ref.textContent = result.reference;
    ref.style.fontWeight = '600';
    
    const meta = document.createElement('div');
    meta.style.display = 'flex';
    meta.style.alignItems = 'center';
    meta.style.gap = '0.6rem';

    const verBadge = document.createElement('span');
    verBadge.className = 'version-badge';
    verBadge.textContent = (detectedVersion || semanticDetectedVersion) ? (detectedVersion || semanticDetectedVersion).toUpperCase() : 'unknown';
    verBadge.style.background = '#eef6ff';
    verBadge.style.color = '#0656a6';
    verBadge.style.padding = '0.15rem 0.4rem';
    verBadge.style.borderRadius = '4px';
    verBadge.style.fontSize = '0.85rem';

    const sim = document.createElement('span');
    sim.className = 'similarity';
    sim.textContent = `Sim ${(result.similarity_score * 100).toFixed(1)}%`;
    sim.style.opacity = '0.85';

    meta.appendChild(verBadge);
    meta.appendChild(sim);
    header.appendChild(ref);
    header.appendChild(meta);
    item.appendChild(header);

    const text = document.createElement('div');
    text.className = 'text';
    text.textContent = result.text;
    text.style.marginTop = '0.4rem';
    item.appendChild(text);

    // Action row
    const actions = document.createElement('div');
    actions.style.marginTop = '0.6rem';
    actions.style.display = 'flex';
    actions.style.gap = '0.5rem';

    const useBtn = document.createElement('button');
    useBtn.textContent = 'Use';
    useBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      // set detected version and select this verse
      semanticDetectedVersion = detectedVersion || semanticDetectedVersion;
      try { document.getElementById('bible-version').value = semanticDetectedVersion; } catch(e) {}
      selectSemanticResult(result);
    });

    const previewBtn = document.createElement('button');
    previewBtn.textContent = 'Preview';
    previewBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      // just show the verse in the pane without changing mode
      document.getElementById('reference').textContent = result.reference;
      document.getElementById('verse').textContent = result.text;
      document.getElementById('verse').classList.remove('multi-verse');
    });

    actions.appendChild(useBtn);
    actions.appendChild(previewBtn);
    item.appendChild(actions);

    // Click handler still selects the verse
    item.addEventListener('click', () => selectSemanticResult(result));
    
    container.appendChild(item);
  });
  
  verseEl.appendChild(container);
}

async function selectSemanticResult(result) {
  // Set the current reference and switch to reference mode
  currentReference = result.reference;
  document.getElementById('search-mode').value = 'reference';
  document.getElementById('search-input').value = result.reference;
  
  // Display the selected verse directly without additional API calls
  // to avoid conflicts with the original search endpoint
  document.getElementById('reference').textContent = result.reference;
  document.getElementById('verse').textContent = result.text;
  document.getElementById('verse').classList.remove('multi-verse');
  
  // If the semantic search detected a best version, set the UI version so nav uses it
  if (semanticDetectedVersion) {
    try { document.getElementById('bible-version').value = semanticDetectedVersion; } catch(e) {}
    lastQuery = { query: result.reference, version: semanticDetectedVersion };
  } else {
    // Fallback: set lastQuery using currently selected version
    try { lastQuery = { query: result.reference, version: document.getElementById('bible-version').value }; } catch(e) { lastQuery = { query: result.reference, version: 'kjv' }; }
  }

  updateSyncStatus(true, 'synced');
}

async function nav(dir) {
  try {
    console.log(`[nav] requesting ${dir}, currentReference:`, currentReference);
    
    // Check if we're in chunked mode (any range with 4+ verses)
    if (originalPassage && originalPassage.verses && originalPassage.verses.length > 3) {
      const totalChunks = Math.ceil(originalPassage.verses.length / chunkSize);
      
      if (dir === 'next') {
        if (currentChunk < totalChunks - 1) {
          // Move to next chunk within current passage
          currentChunk++;
          displayCurrentChunk();
          return;
        } else {
          // We're at the last chunk, continue navigation beyond original range
          // Use the last verse of the passage for navigation
          const lastVerse = originalPassage.verses[originalPassage.verses.length - 1];
          const params = { 
            reference: `${originalPassage.book} ${originalPassage.chapter}:${lastVerse.verse}`, 
            version: document.getElementById('bible-version').value 
          };
          const res = await api.get(`/bible/${dir}`, { params });
          if (res.data && res.data.reference) {
            const ref = res.data.reference.display || res.data.reference.compact || res.data.reference;
            // Reset chunking state and load new reference
            originalPassage = null;
            currentChunk = 0;
            await loadRef(ref, document.getElementById('bible-version').value);
            return;
          }
        }
      } else if (dir === 'prev') {
        if (currentChunk > 0) {
          // Move to previous chunk within current passage
          currentChunk--;
          displayCurrentChunk();
          return;
        } else {
          // We're at the first chunk, continue navigation before original range
          // Use the first verse of the passage for navigation
          const firstVerse = originalPassage.verses[0];
          const params = { 
            reference: `${originalPassage.book} ${originalPassage.chapter}:${firstVerse.verse}`, 
            version: document.getElementById('bible-version').value 
          };
          const res = await api.get(`/bible/${dir}`, { params });
          if (res.data && res.data.reference) {
            const ref = res.data.reference.display || res.data.reference.compact || res.data.reference;
            // Reset chunking state and load new reference
            originalPassage = null;
            currentChunk = 0;
            await loadRef(ref, document.getElementById('bible-version').value);
            return;
          }
        }
      }
    }
    
    // Regular navigation for non-chunked content
    if (!currentReference) {
      console.warn('[nav] no current reference set');
      updateSyncStatus(false, 'no current reference');
      return;
    }
    
    const params = { 
      reference: currentReference, 
      version: document.getElementById('bible-version').value 
    };
    const res = await api.get(`/bible/${dir}`, { params });
    console.log(`[nav] ${dir} response`, res.data);
    if (res.data && res.data.reference) {
      const ref = res.data.reference.display || res.data.reference.compact || res.data.reference;
      await loadRef(ref, document.getElementById('bible-version').value);
    } else {
      console.warn('[nav] no reference in response');
      updateSyncStatus(false, 'no next/prev');
    }
  } catch (err) {
    console.error('[nav] error', err);
    updateSyncStatus(false, 'error');
  }
}

document.addEventListener('DOMContentLoaded', () => {
  const input = document.getElementById('search-input');
  const btn = document.getElementById('search-btn');
  // Microphone / voice capture button (added dynamically)
  const micBtn = document.createElement('button');
  micBtn.id = 'voice-btn';
  micBtn.type = 'button';
  micBtn.textContent = '🎤';
  micBtn.title = 'Start voice capture';
  micBtn.style.marginLeft = '0.5rem';
  micBtn.style.cursor = 'pointer';
  micBtn.style.padding = '0.45rem 0.6rem';
  micBtn.style.fontSize = '1.1rem';
  btn.parentNode && btn.parentNode.insertBefore(micBtn, btn.nextSibling);
  const micStatus = document.createElement('span');
  micStatus.id = 'voice-status';
  micStatus.style.marginLeft = '0.6rem';
  micStatus.style.opacity = '0.8';
  micStatus.style.fontSize = '0.95rem';
  micStatus.textContent = '';
  btn.parentNode && btn.parentNode.insertBefore(micStatus, micBtn.nextSibling);
  const prev = document.getElementById('prevBtn');
  const next = document.getElementById('nextBtn');
  const refresh = document.getElementById('refreshBtn');
  const fsBtn = document.getElementById('toggleFs');
  const perPageSelect = document.getElementById('verses-per-page');
  const searchModeSelect = document.getElementById('search-mode');
  
  // Fullscreen controls
  const fsVersionSelect = document.getElementById('fullscreen-bible-version');
  const fsSearchModeSelect = document.getElementById('fullscreen-search-mode');
  const fsPrev = document.getElementById('fullscreen-prevBtn');
  const fsNext = document.getElementById('fullscreen-nextBtn');
  const fsExit = document.getElementById('fullscreen-exitBtn');

  // Update placeholder text based on search mode
  function updatePlaceholder() {
    const mode = searchModeSelect.value;
    if (mode === 'semantic') {
      input.placeholder = 'Enter text to search (e.g. "love your enemies")';
    } else {
      input.placeholder = 'Enter reference (e.g. John 3:16)';
    }
  }

  // Initialize placeholder
  updatePlaceholder();
  
  // Update placeholder when mode changes
  searchModeSelect.addEventListener('change', (e) => {
    // If switching to semantic, default the version selector to 'auto' but remember previous
    const mode = e.target.value;
    const versionEl = document.getElementById('bible-version');
    if (mode === 'semantic') {
      // Save previous version so we can restore when returning to reference mode
      try {
        if (!versionEl.dataset._prevVersion) versionEl.dataset._prevVersion = versionEl.value || '';
        versionEl.value = 'auto';
      } catch (e) {}
    } else {
      // restore previous version if present
      try {
        if (versionEl.dataset._prevVersion) {
          versionEl.value = versionEl.dataset._prevVersion;
          delete versionEl.dataset._prevVersion;
        }
      } catch (e) {}
    }
    updatePlaceholder();
  });

  // Sync version selectors
  function syncVersions() {
    const mainVersion = document.getElementById('bible-version').value;
    if (fsVersionSelect) fsVersionSelect.value = mainVersion;
  }
  
  // Sync search mode selectors
  function syncSearchModes() {
    const mainMode = searchModeSelect.value;
    if (fsSearchModeSelect) fsSearchModeSelect.value = mainMode;
  }
  
  // Sync versions and modes when main selectors change
  document.getElementById('bible-version').addEventListener('change', syncVersions);
  searchModeSelect.addEventListener('change', syncSearchModes);
  
  // Handle fullscreen version change
  if (fsVersionSelect) {
    fsVersionSelect.addEventListener('change', () => {
      document.getElementById('bible-version').value = fsVersionSelect.value;
      // Reload current reference with new version
      if (lastQuery) loadRef(lastQuery.query, fsVersionSelect.value);
    });
  }
  
  // Handle fullscreen search mode change
  if (fsSearchModeSelect) {
    fsSearchModeSelect.addEventListener('change', () => {
      searchModeSelect.value = fsSearchModeSelect.value;
      updatePlaceholder();
    });
  }

  btn.addEventListener('click', () => {
    const q = input.value.trim();
    const v = document.getElementById('bible-version').value;
    loadRef(q, v);
  });

  // Voice capture: use Web Speech API (webkitSpeechRecognition fallback)
  let recognition = null;
  let recognizing = false;
  function supportsSpeech() {
    return ('SpeechRecognition' in window) || ('webkitSpeechRecognition' in window);
  }

  if (supportsSpeech()) {
    const Rec = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new Rec();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;
    recognition.continuous = false;

    recognition.onstart = () => {
      recognizing = true;
      micBtn.textContent = '●';
      micBtn.style.color = 'crimson';
      micStatus.textContent = 'listening...';
    };

    recognition.onerror = (e) => {
      console.error('[voice] error', e);
      micStatus.textContent = 'error';
      recognizing = false;
      micBtn.textContent = '🎤';
      micBtn.style.color = '';
    };

    recognition.onend = () => {
      recognizing = false;
      micBtn.textContent = '🎤';
      micBtn.style.color = '';
      if (micStatus.textContent === 'listening...') micStatus.textContent = '';
    };

    recognition.onresult = (event) => {
      const transcript = Array.from(event.results).map(r => r[0].transcript).join(' ').trim();
      console.log('[voice] transcript', transcript);
      micStatus.textContent = '';
      input.value = transcript;

      // Simple heuristic to detect reference-like input
      const refLike = /\b[A-Za-z]+\s+\d+(?::\d+(-\d+)?)?(?:-\d+)?\b/.test(transcript) || /:\d+/.test(transcript);
      const version = document.getElementById('bible-version').value;

      if (refLike) {
        // Ensure search mode set to reference and load reference
        const searchModeSelect = document.getElementById('search-mode');
        if (searchModeSelect) searchModeSelect.value = 'reference';
        loadRef(transcript, version);
      } else {
        // Use semantic search for natural language
        const searchModeSelect = document.getElementById('search-mode');
        if (searchModeSelect) searchModeSelect.value = 'semantic';
        loadSemanticSearch(transcript, version);
      }
    };

    micBtn.addEventListener('click', () => {
      if (!recognition) return;
      if (recognizing) {
        recognition.stop();
        recognizing = false;
        micBtn.textContent = '🎤';
        micStatus.textContent = '';
      } else {
        try {
          recognition.start();
        } catch (e) {
          console.warn('[voice] start error', e);
        }
      }
    });
  } else {
    micBtn.title = 'Voice not supported in this browser';
    micBtn.disabled = true;
    micBtn.style.opacity = '0.45';
  }

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') btn.click();
  });

  prev.addEventListener('click', async () => {
    console.log('[ui] prev clicked');
    await nav('prev');
  });

  next.addEventListener('click', async () => {
    console.log('[ui] next clicked');
    await nav('next');
  });

  refresh.addEventListener('click', async () => {
    console.log('[ui] refresh clicked', lastQuery);
    if (lastQuery) await loadRef(lastQuery.query, lastQuery.version);
  });

  // Fullscreen navigation
  if (fsPrev) {
    fsPrev.addEventListener('click', async () => {
      console.log('[ui] fullscreen prev clicked');
      // Update the main version selector to match fullscreen version before nav
      const fsVersion = fsVersionSelect ? fsVersionSelect.value : null;
      if (fsVersion) document.getElementById('bible-version').value = fsVersion;
      await nav('prev');
    });
  }

  if (fsNext) {
    fsNext.addEventListener('click', async () => {
      console.log('[ui] fullscreen next clicked');
      // Update the main version selector to match fullscreen version before nav
      const fsVersion = fsVersionSelect ? fsVersionSelect.value : null;
      if (fsVersion) document.getElementById('bible-version').value = fsVersion;
      await nav('next');
    });
  }

  if (fsExit) {
    fsExit.addEventListener('click', () => {
      if (document.fullscreenElement) document.exitFullscreen();
    });
  }

  fsBtn.addEventListener('click', () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      syncVersions(); // Ensure versions are synced when entering fullscreen
      syncSearchModes(); // Ensure search modes are synced when entering fullscreen
    } else {
      document.exitFullscreen();
    }
  });

  window.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') prev.click();
    if (e.key === 'ArrowRight') next.click();
  });

  // Handle fullscreen changes (including ESC key exit)
  document.addEventListener('fullscreenchange', () => {
    if (document.fullscreenElement) {
      syncVersions(); // Sync versions when entering fullscreen
      syncSearchModes(); // Sync search modes when entering fullscreen
    }
  });

  // persist per-page selection
  try {
    const stored = localStorage.getItem('verses-per-page');
    if (stored && perPageSelect) perPageSelect.value = stored;
    if (perPageSelect) perPageSelect.addEventListener('change', () => {
      localStorage.setItem('verses-per-page', perPageSelect.value);
    });
  } catch (e) {}

  // Add similarity threshold control (persisted)
  const thresholdControl = document.getElementById('similarity-threshold');
  try {
    const storedT = localStorage.getItem('similarity-threshold');
    if (storedT && thresholdControl) thresholdControl.value = storedT;
    if (thresholdControl) thresholdControl.addEventListener('change', () => {
      localStorage.setItem('similarity-threshold', thresholdControl.value);
    });
  } catch (e) {}

  // initial health check
  (async function() {
    try {
      const h = await axios.get('/api/bible/health');
      if (h && h.data && h.data.status === 'healthy') updateSyncStatus(true, 'Backend online');
      else updateSyncStatus(false, 'Backend unexpected response');
    } catch (e) { updateSyncStatus(false, 'Backend offline'); }
  })();
});

// Expose for debugging
window._app = { loadRef, nav };
console.log('[app.js] loaded', new Date().toISOString());

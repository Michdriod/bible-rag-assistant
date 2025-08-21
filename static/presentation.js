(function(){
  const params = new URLSearchParams(window.location.search);
  let reference = params.get('reference') || '';
  let version = params.get('version') || 'kjv';

  const refEl = document.getElementById('reference');
  const verseEl = document.getElementById('verse');
  const prevBtn = document.getElementById('prevBtn');
  const nextBtn = document.getElementById('nextBtn');
  const toggleFs = document.getElementById('toggleFs');

  async function loadRef(ref) {
    if (!ref) return;
    try {
      const resp = await axios.get('/api/bible/prev', { params: { reference: ref, version: version } });
      // We asked for prev to ensure existence when navigating backward; for initial load, try search endpoint first
    } catch (e) {
      // ignore
    }

    try {
      const resp = await axios.get('/api/bible/next', { params: { reference: ref, version: version } });
      // If next exists we'll show it; but primary is to call the search endpoint for exact verse
    } catch (e) {
      // ignore
    }

    // Primary: fetch via search endpoint to get the exact verse data
    try {
      const r = await axios.post('/api/bible/search', { query: ref, version: version, include_context: false });
      if (r.data && r.data.results && r.data.results.length > 0) {
        const v = r.data.results[0];
        refEl.textContent = v.reference;
        verseEl.textContent = v.text;
        reference = v.reference;
      } else {
        refEl.textContent = 'Not found';
        verseEl.textContent = '';
      }
    } catch (err) {
      refEl.textContent = 'Error';
      verseEl.textContent = '';
    }
  }

  prevBtn.addEventListener('click', async function(){
    try {
      const resp = await axios.get('/api/bible/prev', { params: { reference: reference, version: version } });
      if (resp.data && resp.data.reference) {
        reference = resp.data.reference;
        await loadRef(reference);
      }
    } catch (e) {
      // ignore
    }
  });

  nextBtn.addEventListener('click', async function(){
    try {
      const resp = await axios.get('/api/bible/next', { params: { reference: reference, version: version } });
      if (resp.data && resp.data.reference) {
        reference = resp.data.reference;
        await loadRef(reference);
      }
    } catch (e) {
      // ignore
    }
  });

  toggleFs.addEventListener('click', function(){
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
    } else {
      document.exitFullscreen();
    }
  });

  // Keyboard navigation in presentation
  document.addEventListener('keydown', async function(e){
    if (e.key === 'ArrowRight') {
      nextBtn.click();
    } else if (e.key === 'ArrowLeft') {
      prevBtn.click();
    }
  });

  // initial load
  if (reference) {
    loadRef(reference);
  } else {
    refEl.textContent = 'No reference provided';
  }
})();

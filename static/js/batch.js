/* Batch automation client.
 *
 * Generation runs a chunk at a time rather than in one request. A hundred
 * rows means a hundred model calls; asked for in one go the browser waits
 * several minutes with nothing on screen and the request may time out.
 * Chunking keeps each response quick, fills the table as it goes, and lets
 * the run be stopped part way.
 */
document.addEventListener('DOMContentLoaded', () => {
  // Read from the data attributes rendered by batch.html, so no server
  // template syntax ever appears inside a script block.
  const config = document.getElementById('batch-config');
  const CHUNK = Number(config?.dataset.chunkSize) || 5;
  const SMTP_READY = config?.dataset.smtpReady === 'true';

  const $ = (id) => document.getElementById(id);
  const show = (el) => el && el.classList.remove('is-hidden');
  const hide = (el) => el && el.classList.add('is-hidden');

  const state = {
    sessionId: null,
    rowCount: 0,
    results: [],
    generating: false,
    cancelled: false,
  };

  const setStatus = (el, message, kind) => {
    if (!el) return;
    el.textContent = message || '';
    el.dataset.kind = kind || '';
  };

  const escapeHtml = (value) =>
    String(value == null ? '' : value)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');

  const post = async (url, formData) => {
    const response = await fetch(url, { method: 'POST', body: formData });
    let payload = {};
    try { payload = await response.json(); } catch (err) { payload = {}; }
    if (!response.ok) {
      throw new Error(payload.detail || 'The request failed.');
    }
    return payload;
  };

  /* ---------------- Step 1: read the files ---------------- */

  $('inspect-btn').addEventListener('click', async () => {
    const cv = $('cv-file').files[0];
    const workbook = $('workbook-file').files[0];

    if (!cv || !workbook) {
      setStatus($('inspect-status'), 'Choose both a CV and a job spreadsheet.', 'error');
      return;
    }

    const button = $('inspect-btn');
    button.disabled = true;
    setStatus($('inspect-status'), 'Reading the CV and spreadsheet…');

    const formData = new FormData();
    formData.append('resume_file', cv);
    formData.append('workbook_file', workbook);

    try {
      const data = await post('/batch/inspect', formData);
      if (data.needs_mapping) {
        renderMapping(data, true);
        setStatus($('inspect-status'), data.message, 'error');
      } else {
        state.sessionId = data.session_id;
        state.rowCount = data.row_count;
        renderMapping(data, false);
        prepareOptions(data);
        setStatus($('inspect-status'), `Read ${data.row_count} job rows.`, 'ok');
      }
    } catch (err) {
      setStatus($('inspect-status'), err.message, 'error');
    } finally {
      button.disabled = false;
    }
  });

  /* ---------------- Step 2: columns ---------------- */

  function renderMapping(data, needsPicker) {
    const detected = data.detected || {};
    const columns = data.columns || [];
    const byIndex = {};
    columns.forEach((column) => { byIndex[column.index] = column; });

    const rows = Object.entries(detected.mapping || {}).map(([field, index]) => {
      const column = byIndex[index] || {};
      const sample = (column.samples || [])[0] || '';
      return `<tr>
        <td>${escapeHtml(detected.labels[field] || field)}</td>
        <td><code>${escapeHtml(column.name || '')}</code></td>
        <td class="muted">${escapeHtml(sample.slice(0, 70))}</td>
      </tr>`;
    }).join('');

    $('mapping-table').innerHTML = rows
      ? `<div class="batch-table-wrap"><table class="batch-table">
           <thead><tr><th>Field</th><th>Column</th><th>Example</th></tr></thead>
           <tbody>${rows}</tbody></table></div>`
      : '';

    $('mapping-summary').textContent = needsPicker
      ? 'Some required columns could not be identified.'
      : `Detected automatically from ${columns.length} columns.`;

    show($('step-mapping'));

    if (!needsPicker) {
      hide($('mapping-picker'));
      return;
    }

    // The server refused to guess. Offer every column for the fields it could
    // not resolve, rather than picking one and being confidently wrong.
    const required = detected.required_fields || ['role', 'description'];
    const options = columns.map(
      (column) => `<option value="${column.index}">${escapeHtml(column.name)}</option>`
    ).join('');

    $('mapping-selects').innerHTML = required.map((field) => `
      <div class="field">
        <label for="map-${field}">${escapeHtml(detected.labels[field] || field)}</label>
        <select id="map-${field}" data-field="${field}">
          <option value="">- choose a column -</option>${options}
        </select>
      </div>`).join('');

    show($('mapping-picker'));

    $('confirm-mapping-btn').onclick = async () => {
      const mapping = {};
      let incomplete = false;
      required.forEach((field) => {
        const value = $(`map-${field}`).value;
        if (value === '') incomplete = true;
        else mapping[field] = Number(value);
      });
      if (incomplete) {
        setStatus($('inspect-status'), 'Choose a column for every required field.', 'error');
        return;
      }

      const formData = new FormData();
      formData.append('resume_path', data.resume_path);
      formData.append('workbook_path', data.workbook_path);
      formData.append('sheet', data.sheet || '');
      formData.append('header_row', data.header_row || 1);
      formData.append('mapping_json', JSON.stringify(mapping));
      formData.append('cv_filename', $('cv-file').files[0]?.name || '');
      formData.append('workbook_name', $('workbook-file').files[0]?.name || '');

      try {
        const confirmed = await post('/batch/confirm-mapping', formData);
        state.sessionId = confirmed.session_id;
        state.rowCount = confirmed.row_count;
        hide($('mapping-picker'));
        prepareOptions(confirmed);
        setStatus($('inspect-status'), `Read ${confirmed.row_count} job rows.`, 'ok');
      } catch (err) {
        setStatus($('inspect-status'), err.message, 'error');
      }
    };
  }

  /* ---------------- Step 3: options ---------------- */

  function prepareOptions(data) {
    show($('step-options'));
    const count = $('job-count');
    count.max = data.row_count;
    count.value = Math.min(5, data.row_count);
    $('row-hint').textContent =
      `${data.row_count} rows available. Start with a few before running the whole file.`;

    if (data.has_row_emails) show($('row-email-option'));
    if (!SMTP_READY) {
      setStatus($('generate-status'),
        'Note: email sending is not configured, so drafts can be generated and reviewed but not sent.');
    }
  }

  document.querySelectorAll('input[name="recipient-mode"]').forEach((radio) => {
    radio.addEventListener('change', () => {
      const useRow = radio.value === 'row' && radio.checked;
      $('demo-recipient').disabled = useRow;
      $('demo-recipient').placeholder = useRow
        ? "Taken from each row" : 'Recipient email address';
    });
  });

  /* ---------------- Generation, one chunk at a time ---------------- */

  $('generate-btn').addEventListener('click', async () => {
    if (state.generating) {           // second click stops the run
      state.cancelled = true;
      return;
    }
    if (!state.sessionId) {
      setStatus($('generate-status'), 'Read the files first.', 'error');
      return;
    }

    const mode = document.querySelector('input[name="batch-mode"]:checked').value;
    const useRowEmail = document.querySelector('input[name="recipient-mode"]:checked').value === 'row';
    const recipient = $('demo-recipient').value.trim();
    const wanted = Math.max(1, Math.min(Number($('job-count').value) || 1, state.rowCount));

    if (!useRowEmail && !/^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$/.test(recipient)) {
      setStatus($('generate-status'), 'Enter a valid recipient email address.', 'error');
      return;
    }

    state.generating = true;
    state.cancelled = false;
    state.results = [];
    $('results-body').innerHTML = '';
    $('generate-btn').textContent = 'Stop';
    show($('progress-wrap'));
    show($('step-results'));
    hide($('step-send'));
    setStatus($('generate-status'), '');

    let offset = 0;
    try {
      while (offset < wanted && !state.cancelled) {
        const formData = new FormData();
        formData.append('session_id', state.sessionId);
        formData.append('mode', mode);
        formData.append('offset', offset);
        formData.append('limit', CHUNK);
        formData.append('total', wanted);
        formData.append('demo_recipient', recipient);
        formData.append('use_row_email', useRowEmail ? 'true' : 'false');

        const data = await post('/batch/generate', formData);
        (data.results || []).forEach(appendResultRow);
        state.results = state.results.concat(data.results || []);
        offset = data.next_offset || wanted;
        updateProgress(Math.min(offset, wanted), wanted, data.summary);
        if (data.done) break;
      }

      const generated = state.results.filter((item) => item.status === 'generated').length;
      setStatus($('generate-status'),
        state.cancelled ? `Stopped after ${state.results.length} rows.` :
        `Finished. ${generated} of ${state.results.length} rows produced an email.`,
        'ok');
      if (generated > 0) prepareSend(generated, useRowEmail, recipient);
    } catch (err) {
      setStatus($('generate-status'), err.message, 'error');
    } finally {
      state.generating = false;
      $('generate-btn').textContent = 'Generate emails';
    }
  });

  function updateProgress(done, total, summary) {
    $('progress-fill').style.setProperty('--fill', `${Math.round((done / total) * 100)}%`);
    $('progress-text').textContent = `${done} of ${total} processed`;
    if (summary) {
      $('batch-tally').innerHTML = `
        <span class="tally-pill score-good">${summary.generated} generated</span>
        <span class="tally-pill score-average">${summary.skipped} skipped</span>
        <span class="tally-pill score-poor">${summary.failed} failed</span>`;
    }
  }

  function appendResultRow(item) {
    const pill = item.status === 'generated' ? 'score-good'
               : item.status === 'skipped' ? 'score-average' : 'score-poor';
    const detail = item.status === 'generated'
      ? `<button class="link-btn" data-row="${item.row}" type="button">View</button>`
      : `<span class="muted">${escapeHtml(item.error || '')}</span>`;

    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${escapeHtml(item.row)}</td>
      <td>${escapeHtml((item.role || '').slice(0, 46))}</td>
      <td>${escapeHtml((item.company || '').slice(0, 24))}</td>
      <td class="muted">${escapeHtml(item.recipient || '')}</td>
      <td><span class="score-pill ${pill}">${escapeHtml(item.status)}</span></td>
      <td>${detail}</td>`;
    $('results-body').appendChild(row);

    const button = row.querySelector('.link-btn');
    if (button) button.addEventListener('click', () => openDraft(item));
  }

  /* ---------------- Draft viewer ---------------- */

  function openDraft(item) {
    $('modal-role').textContent = item.role || 'Generated email';
    $('modal-subject').value = item.subject || '';
    $('modal-body').value = item.body || '';
    show($('email-modal'));
  }
  $('modal-close').addEventListener('click', () => hide($('email-modal')));
  $('email-modal').addEventListener('click', (event) => {
    if (event.target === $('email-modal')) hide($('email-modal'));
  });

  /* ---------------- Step 5: send ---------------- */

  function prepareSend(generated, useRowEmail, recipient) {
    show($('step-send'));
    $('send-summary').innerHTML = `
      <p class="decision-rationale"><strong>${generated}</strong> email(s) ready to send.</p>
      <ul class="decision-reasons">
        <li>Recipient mode: ${useRowEmail ? "each row's own address" : 'single address (demo)'}</li>
        <li>Recipient: ${escapeHtml(useRowEmail ? 'from the spreadsheet' : recipient)}</li>
      </ul>
      <p class="decision-basis">Nothing has been sent yet. Sending is a separate, confirmed step.</p>`;
    $('send-btn').disabled = !SMTP_READY;
    if (!SMTP_READY) {
      setStatus($('send-status'),
        'Email sending is not configured. Add the SMTP settings to .env and restart.', 'error');
    }
  }

  const runSend = async (dryRun) => {
    if (!dryRun) {
      const generated = state.results.filter((item) => item.status === 'generated').length;
      // Sending is irreversible, so it takes an explicit yes.
      if (!window.confirm(`Send ${generated} email(s) now? This cannot be undone.`)) return;
    }

    $('send-btn').disabled = true;
    $('dry-run-btn').disabled = true;
    setStatus($('send-status'), dryRun ? 'Running through without sending…' : 'Sending…');

    const formData = new FormData();
    formData.append('session_id', state.sessionId);
    formData.append('confirm', 'true');
    formData.append('dry_run', dryRun ? 'true' : 'false');

    try {
      const data = await post('/batch/send', formData);
      $('send-results-body').innerHTML = (data.results || []).map((item) => {
        const pill = item.status === 'sent' ? 'score-good'
                   : item.status === 'dry-run' ? 'score-average' : 'score-poor';
        return `<tr>
          <td>${escapeHtml(item.row)}</td>
          <td class="muted">${escapeHtml(item.recipient)}</td>
          <td>${escapeHtml((item.subject || '').slice(0, 52))}</td>
          <td><span class="score-pill ${pill}">${escapeHtml(item.status)}</span></td>
          <td class="muted">${escapeHtml(item.error || '')}</td>
        </tr>`;
      }).join('');
      show($('send-results-wrap'));
      setStatus($('send-status'),
        dryRun
          ? `Dry run complete: ${data.results.length} would be sent. Nothing left the server.`
          : `Sent ${data.sent}, skipped ${data.skipped}, failed ${data.failed}.`,
        'ok');
    } catch (err) {
      setStatus($('send-status'), err.message, 'error');
    } finally {
      $('send-btn').disabled = !SMTP_READY;
      $('dry-run-btn').disabled = false;
    }
  };

  $('send-btn').addEventListener('click', () => runSend(false));
  $('dry-run-btn').addEventListener('click', () => runSend(true));
});